#    Copyright 2025 TianYe
#    
#    Licensed under the Apache License, Version 2.0 (the "License");
#
#    Landmark Grounding Head for NaVid-VLN-CE
#    Conservatively improves navigation by injecting instruction-grounded landmark tokens

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class LandmarkGroundingHead(nn.Module):
    """
    指令驱动的地标定位头 (Landmark Grounding Head)
    
    目标：通过注入 K 个指令相关的地标标记到视觉标记流中来保守地改进 VLN-CE 导航，
    同时保持基础规划器不变。
    """
    
    def __init__(
        self,
        vision_dim: int = 1408,  # EVA-ViT-G 特征维度 
        instruction_dim: int = 4096,  # LLaMA token embedding 维度
        num_landmark_queries: int = 8,  # M 个学习查询
        num_landmark_tokens: int = 4,  # K 每帧地标标记数
        confidence_threshold: float = 0.2,  # 置信度门槛
        temperature: float = 1.0,  # Softmax 温度 (eval时为0)
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.instruction_dim = instruction_dim
        self.num_landmark_queries = num_landmark_queries
        self.num_landmark_tokens = num_landmark_tokens
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.dropout = dropout
        
        # 学习的查询向量，由指令嵌入调制
        self.landmark_queries = nn.Parameter(
            torch.randn(num_landmark_queries, vision_dim) * 0.02
        )
        
        # 指令池化和投影
        self.instruction_pooler = nn.Sequential(
            nn.Linear(instruction_dim, instruction_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(instruction_dim // 2, vision_dim)
        )
        
        # 交叉注意力层：查询来自指令短语嵌入，键值来自视觉补丁
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 归一化层
        self.norm_pre = nn.LayerNorm(vision_dim)
        self.norm_post = nn.LayerNorm(vision_dim)
        
        # 分数预测头 (指针分布)
        self.scoring_head = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim // 2, 1)
        )
        
        # 小投影层 (如果需要维度匹配)
        self.output_projection = nn.Identity()
        if vision_dim != vision_dim:  # 可配置输出维度
            self.output_projection = nn.Linear(vision_dim, vision_dim)
        
        # 置信度门控
        self.confidence_gate = nn.Sequential(
            nn.Linear(vision_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visual_patches: torch.Tensor,  # [B, T, N, Dv] N个补丁每帧，EVA-ViT-G后
        instr_emb: torch.Tensor,       # [B, L, Dt] 指令token嵌入
        mask: Optional[torch.Tensor] = None,  # [B, T, N] 可选mask
        eval_mode: bool = False        # 评估模式（确定性）
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            visual_patches: [B, T, N, Dv] 视觉补丁特征 
            instr_emb: [B, L, Dt] 指令嵌入
            mask: [B, T, N] 可选注意力mask
            eval_mode: 评估模式（确定性top-k选择）
            
        Returns:
            Dict包含:
            - landmark_tokens: [B, T, K, Dv] K个地标标记每帧
            - landmark_scores: [B, T, N] 选择分数over N个补丁
            - selected_idx: [B, T, K] 选中的补丁索引
            - confidence: [B, T] 每帧置信度
            - gate_mask: [B, T] 门控mask (低置信度为False)
        """
        B, T, N, Dv = visual_patches.shape
        L = instr_emb.shape[1]
        
        # 1. 池化指令嵌入
        pooled_instr = instr_emb.mean(dim=1)  # [B, Dt] 平均池化
        modulated_queries = self.instruction_pooler(pooled_instr)  # [B, Dv]
        
        # 2. 调制学习查询 
        # 广播并调制：[B, M, Dv] = [1, M, Dv] + [B, 1, Dv]
        queries = self.landmark_queries.unsqueeze(0) + modulated_queries.unsqueeze(1)
        queries = self.norm_pre(queries)  # [B, M, Dv]
        
        # 3. 处理每个时间步
        all_landmark_tokens = []
        all_landmark_scores = []
        all_selected_idx = []
        all_confidence = []
        all_gate_mask = []
        
        for t in range(T):
            # 当前帧的视觉补丁: [B, N, Dv]
            frame_patches = visual_patches[:, t, :, :]
            frame_patches = self.norm_pre(frame_patches)
            
            # 交叉注意力: queries来自指令，keys/values来自视觉补丁
            attn_output, attn_weights = self.cross_attention(
                query=queries,      # [B, M, Dv]
                key=frame_patches,  # [B, N, Dv]
                value=frame_patches, # [B, N, Dv]
                key_padding_mask=mask[:, t, :] if mask is not None else None
            )
            
            # 4. 计算每个补丁的选择分数
            # 使用注意力输出计算分数
            combined_features = attn_output.mean(dim=1, keepdim=True)  # [B, 1, Dv]
            combined_features = combined_features.expand(-1, N, -1)  # [B, N, Dv]
            
            # 元素级相似度 + MLP打分
            patch_scores = torch.sum(
                combined_features * frame_patches, dim=-1
            ) / (Dv ** 0.5)  # [B, N] 缩放点积
            
            # 使用MLP进一步细化分数
            mlp_scores = self.scoring_head(frame_patches).squeeze(-1)  # [B, N]
            final_scores = patch_scores + mlp_scores  # [B, N]
            
            # 应用温度缩放
            temp = 0.0 if eval_mode else self.temperature
            if temp > 0:
                final_scores = final_scores / temp
            
            # 5. 计算置信度和门控
            confidence = self.confidence_gate(combined_features.mean(dim=1))  # [B, 1]
            confidence = confidence.squeeze(-1)  # [B]
            gate_mask = confidence > self.confidence_threshold  # [B]
            
            # 6. Top-K 选择 (确定性在eval模式)
            if eval_mode:
                # 确定性选择：固定种子 + top-k with tie-breaker
                # 使用torch.topk确保确定性
                top_scores, top_indices = torch.topk(
                    final_scores, k=self.num_landmark_tokens, dim=-1, sorted=True
                )  # [B, K] sorted=True确保确定性
            else:
                # 训练时：保持可微性的soft selection
                softmax_scores = F.softmax(final_scores, dim=-1)
                top_scores, top_indices = torch.topk(
                    softmax_scores, k=self.num_landmark_tokens, dim=-1
                )
            
            # 7. 收集选中的地标标记
            # 使用gather收集选中的补丁
            selected_patches = torch.gather(
                frame_patches,  # [B, N, Dv]
                dim=1,
                index=top_indices.unsqueeze(-1).expand(-1, -1, Dv)  # [B, K, Dv]
            )
            
            # 应用门控：低置信度时返回零标记
            zero_tokens = torch.zeros_like(selected_patches)
            gated_patches = torch.where(
                gate_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1]
                selected_patches,
                zero_tokens
            )
            
            # 应用输出投影
            landmark_tokens = self.output_projection(gated_patches)  # [B, K, Dv]
            landmark_tokens = self.norm_post(landmark_tokens)
            
            # 存储结果
            all_landmark_tokens.append(landmark_tokens)
            all_landmark_scores.append(final_scores)
            all_selected_idx.append(top_indices)
            all_confidence.append(confidence)
            all_gate_mask.append(gate_mask)
        
        # 8. 堆叠时间维度
        result = {
            'landmark_tokens': torch.stack(all_landmark_tokens, dim=1),  # [B, T, K, Dv]
            'landmark_scores': torch.stack(all_landmark_scores, dim=1),  # [B, T, N] 
            'selected_idx': torch.stack(all_selected_idx, dim=1),        # [B, T, K]
            'confidence': torch.stack(all_confidence, dim=1),           # [B, T]
            'gate_mask': torch.stack(all_gate_mask, dim=1),            # [B, T]
            'contrastive_logits': final_scores.unsqueeze(1).expand(-1, T, -1)  # [B, T, N] for loss
        }
        
        return result

    def set_eval_mode(self, eval_mode: bool = True):
        """设置评估模式以确保确定性选择"""
        self.eval()
        # 固定随机种子以确保确定性
        if eval_mode:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

    def get_num_parameters(self):
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
