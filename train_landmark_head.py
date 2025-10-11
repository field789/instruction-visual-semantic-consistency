#!/usr/bin/env python3
#    Copyright 2025 TianYe
#    
#    Licensed under the Apache License, Version 2.0 (the "License");
#
#    Lightweight trainer for Landmark Grounding Head

import argparse
import gzip
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navid.model.builder import load_pretrained_model
from navid.modules.landmark_head import LandmarkGroundingHead
from navid.utils.instruction_spans import extract_landmark_phrases, prepare_instruction_embeddings


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandmarkTrainingDataset(Dataset):
    """
    Lightweight dataset for training landmark head on R2R data without env stepping
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        max_frames: int = 10,
        max_instructions: int = 100
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_frames = max_frames
        
        # Load R2R training data
        logger.info(f"Loading training data from {data_path}")
        if data_path.endswith('.gz'):
            with gzip.open(data_path, 'rt') as f:
                self.data = json.load(f)
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        # Extract episodes from R2R format
        if isinstance(self.data, dict) and 'episodes' in self.data:
            self.data = self.data['episodes']
        
        # Limit data size if specified
        if max_instructions > 0:
            self.data = self.data[:max_instructions]
            
        logger.info(f"Loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns synthetic training example:
        - Random visual patches (simulating EVA-ViT-G features)
        - Instruction text
        - Landmark phrases (for contrastive learning)
        """
        item = self.data[idx]
        
        # Extract instruction text from R2R format
        if 'instruction' in item:
            if isinstance(item['instruction'], dict):
                instruction = item['instruction'].get('instruction_text', '')
            else:
                instruction = item['instruction']
        elif 'instructions' in item:
            instruction = item['instructions'][0] if item['instructions'] else ''
        else:
            instruction = ''
        
        # Generate synthetic visual patches [T, N, D] with fixed T for batching
        T = 5  # Fixed number of frames to enable batching
        N = 256  # EVA-ViT-G patch count
        D = 1408  # EVA-ViT-G feature dim
        
        # Create realistic visual features (normally distributed)
        visual_patches = torch.randn(T, N, D) * 0.02
        
        # Extract landmark phrases
        landmark_phrases = extract_landmark_phrases(instruction)
        
        # Convert instruction to token embeddings (simplified)
        instruction_tokens = self.tokenizer.encode(instruction, add_special_tokens=False)
        if len(instruction_tokens) > 77:  # CLIP limit
            instruction_tokens = instruction_tokens[:77]
        
        # Pad to fixed length
        max_len = 77
        if len(instruction_tokens) < max_len:
            instruction_tokens += [self.tokenizer.pad_token_id] * (max_len - len(instruction_tokens))
        
        return {
            'visual_patches': visual_patches,  # [T, N, D]
            'instruction': instruction,
            'instruction_tokens': torch.tensor(instruction_tokens),
            'landmark_phrases': landmark_phrases[:5] + [''] * (5 - min(len(landmark_phrases), 5)),  # Fixed length
            'episode_id': item.get('episode_id', str(idx))
        }


class LandmarkTrainer:
    """Trainer for Landmark Grounding Head"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        config: Dict
    ):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model components
        logger.info("Loading pretrained model...")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "navid", device="cuda"
        )
        
        # Initialize landmark head
        self.landmark_head = LandmarkGroundingHead(
            vision_dim=1408,
            instruction_dim=4096,
            num_landmark_queries=config.get('num_landmark_queries', 8),
            num_landmark_tokens=config.get('landmark_k', 4),
            confidence_threshold=config.get('confidence_threshold', 0.2),
            temperature=config.get('temperature', 1.0),
            dropout=config.get('dropout', 0.1)
        ).cuda()
        
        # Freeze backbone and main planner
        if config.get('freeze_backbone', True):
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Frozen backbone and main planner")
        
        # Only train landmark head parameters
        trainable_params = list(self.landmark_head.parameters())
        logger.info(f"Training {sum(p.numel() for p in trainable_params)} parameters")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('lr_head', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 10),
            eta_min=config.get('lr_head', 1e-4) * 0.1
        )
        
        # Loss weights
        self.loss_weights = {
            'infonce': config.get('loss_infonce', 1.0),
            'diversity': config.get('loss_diversity', 0.10),
            'gate': config.get('loss_gate', 0.05)
        }
        self.warmup_epochs = config.get('warmup_epochs', 2)
        
        # Add projection layer to align landmark and instruction features
        landmark_dim = self.landmark_head.landmark_queries.shape[-1]  # Should be 1408
        instruction_dim = self.model.get_model().embed_tokens.embedding_dim  # Should be 4096
        
        self.instruction_proj = nn.Linear(instruction_dim, landmark_dim).cuda()
        
        # Add projection layer parameters to optimizer
        proj_params = list(self.instruction_proj.parameters())
        trainable_params.extend(proj_params)
        
        # Re-initialize optimizer with projection layer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('lr_head', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        
    def compute_contrastive_loss(
        self,
        landmark_tokens: torch.Tensor,  # [B, T, K, D]
        landmark_phrases: List[List[str]],
        instruction_embeddings: torch.Tensor  # [B, L, D]
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between landmark tokens and phrase embeddings
        """
        B, T, K, D = landmark_tokens.shape
        
        # Pool landmark tokens across time and spatial dimensions
        pooled_landmarks = landmark_tokens.mean(dim=[1, 2])  # [B, D]
        
        # Pool and project instruction/phrase embeddings
        pooled_instructions = instruction_embeddings.mean(dim=1)  # [B, D_instr]
        pooled_instructions = self.instruction_proj(pooled_instructions)  # [B, D_landmark]
        
        # Normalize features
        pooled_landmarks = F.normalize(pooled_landmarks, dim=1)
        pooled_instructions = F.normalize(pooled_instructions, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(pooled_landmarks, pooled_instructions.T)  # [B, B]
        logits = logits / 0.07  # Temperature
        
        # InfoNCE loss (positive pairs on diagonal)
        labels = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_diversity_loss(self, landmark_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encourage landmark tokens within a frame to be different (cosine repulsion)
        """
        B, T, K, D = landmark_tokens.shape
        
        if K <= 1:
            return torch.tensor(0.0, device=landmark_tokens.device)
        
        total_loss = 0.0
        count = 0
        
        for b in range(B):
            for t in range(T):
                frame_tokens = landmark_tokens[b, t]  # [K, D]
                frame_tokens = F.normalize(frame_tokens, dim=1)
                
                # Compute pairwise cosine similarities
                similarities = torch.matmul(frame_tokens, frame_tokens.T)  # [K, K]
                
                # Mask out diagonal (self-similarity)
                mask = torch.eye(K, device=similarities.device, dtype=torch.bool)
                off_diagonal = similarities[~mask]
                
                # Penalize high similarities (encourage diversity)
                diversity_loss = torch.mean(torch.relu(off_diagonal))
                total_loss += diversity_loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def compute_gating_loss(
        self,
        confidence: torch.Tensor,  # [B, T]
        gate_mask: torch.Tensor,   # [B, T]
        target_sparsity: float = 0.7
    ) -> torch.Tensor:
        """
        Encourage sparsity and high confidence when gate is open
        """
        # Sparsity loss: encourage gates to be closed when confidence is low
        gate_open_rate = gate_mask.float().mean()
        sparsity_loss = F.mse_loss(gate_open_rate, torch.tensor(target_sparsity, device=confidence.device))
        
        # Confidence loss: encourage high confidence when gate is open
        gated_confidence = confidence * gate_mask.float()
        confidence_loss = -gated_confidence.mean()
        
        return sparsity_loss + confidence_loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.landmark_head.train()
        
        total_loss = 0.0
        total_infonce = 0.0
        total_diversity = 0.0
        total_gate = 0.0
        total_confidence = 0.0
        total_replacement_ratio = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # Move to device and ensure consistent dtype
            visual_patches = batch['visual_patches'].cuda().float()  # [B, T, N, D]
            instruction_tokens = batch['instruction_tokens'].cuda()
            
            # Convert instruction tokens to embeddings using model's embedding layer
            with torch.no_grad():
                instruction_embeddings = self.model.get_model().embed_tokens(instruction_tokens).float()  # [B, L, D]
            
            # Forward pass through landmark head
            landmark_result = self.landmark_head(
                visual_patches,
                instruction_embeddings,
                eval_mode=False
            )
            
            landmark_tokens = landmark_result['landmark_tokens']
            confidence = landmark_result['confidence']
            gate_mask = landmark_result['gate_mask']
            
            # Compute losses
            infonce_loss = self.compute_contrastive_loss(
                landmark_tokens, batch['landmark_phrases'], instruction_embeddings
            )
            
            diversity_loss = self.compute_diversity_loss(landmark_tokens)
            gate_loss = self.compute_gating_loss(confidence, gate_mask)
            
            # Apply loss weights and warmup
            if epoch < self.warmup_epochs:
                # Warmup: only InfoNCE loss
                total_batch_loss = self.loss_weights['infonce'] * infonce_loss
            else:
                # Full loss
                total_batch_loss = (
                    self.loss_weights['infonce'] * infonce_loss +
                    self.loss_weights['diversity'] * diversity_loss +
                    self.loss_weights['gate'] * gate_loss
                )
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.landmark_head.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_infonce += infonce_loss.item()
            total_diversity += diversity_loss.item()
            total_gate += gate_loss.item()
            total_confidence += confidence.mean().item()
            total_replacement_ratio += gate_mask.float().mean().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'infonce': f'{infonce_loss.item():.4f}',
                'conf': f'{confidence.mean().item():.3f}',
                'gate': f'{gate_mask.float().mean().item():.3f}'
            })
        
        # Compute epoch averages
        num_batches = len(dataloader)
        metrics = {
            'train/loss': total_loss / num_batches,
            'train/infonce_loss': total_infonce / num_batches,
            'train/diversity_loss': total_diversity / num_batches,
            'train/gate_loss': total_gate / num_batches,
            'train/avg_confidence': total_confidence / num_batches,
            'train/replacement_ratio': total_replacement_ratio / num_batches,
            'train/lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation epoch"""
        self.landmark_head.eval()
        
        total_loss = 0.0
        total_infonce = 0.0
        total_diversity = 0.0
        total_gate = 0.0
        total_confidence = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device and ensure consistent dtype
                visual_patches = batch['visual_patches'].cuda().float()  # [B, T, N, D]
                instruction_tokens = batch['instruction_tokens'].cuda()
                
                # Convert instruction tokens to embeddings
                instruction_embeddings = self.model.get_model().embed_tokens(instruction_tokens).float()
                
                # Forward pass
                landmark_result = self.landmark_head(
                    visual_patches,
                    instruction_embeddings,
                    eval_mode=True  # Use eval mode for validation
                )
                
                landmark_tokens = landmark_result['landmark_tokens']
                confidence = landmark_result['confidence']
                gate_mask = landmark_result['gate_mask']
                
                # Compute losses
                infonce_loss = self.compute_contrastive_loss(
                    landmark_tokens, batch['landmark_phrases'], instruction_embeddings
                )
                diversity_loss = self.compute_diversity_loss(landmark_tokens)
                gate_loss = self.compute_gating_loss(confidence, gate_mask)
                
                # Total loss
                total_batch_loss = (
                    self.loss_weights['infonce'] * infonce_loss +
                    self.loss_weights['diversity'] * diversity_loss +
                    self.loss_weights['gate'] * gate_loss
                )
                
                # Accumulate metrics
                total_loss += total_batch_loss.item()
                total_infonce += infonce_loss.item()
                total_diversity += diversity_loss.item()
                total_gate += gate_loss.item()
                total_confidence += confidence.mean().item()
                num_batches += 1
        
        # Calculate averages
        metrics = {
            'val/loss': total_loss / num_batches,
            'val/infonce_loss': total_infonce / num_batches,
            'val/diversity_loss': total_diversity / num_batches,
            'val/gate_loss': total_gate / num_batches,
            'val/avg_confidence': total_confidence / num_batches,
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'landmark_head_state_dict': self.landmark_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'landmark_head.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the lowest loss
        best_path = os.path.join(self.output_dir, 'landmark_head_best.pt')
        if not os.path.exists(best_path) or metrics['train/loss'] < self.best_loss:
            torch.save(checkpoint, best_path)
            self.best_loss = metrics['train/loss']
            logger.info(f"Saved best checkpoint at epoch {epoch}")
    
    def train(self, train_dataloader: DataLoader, epochs: int, val_dataloader=None, patience: int = 3):
        """Main training loop with validation and early stopping"""
        logger.info(f"Starting training for {epochs} epochs")
        self.best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation epoch if validation data is provided
            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
                
                # Check for early stopping based on validation loss
                current_val_loss = val_metrics['val/loss']
                if current_val_loss < self.best_loss:
                    self.best_loss = current_val_loss
                    patience_counter = 0
                    # Save best checkpoint
                    self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
                    logger.info(f"Saved best checkpoint at epoch {epoch} with val_loss={current_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                        break
                
                # Log combined progress
                logger.info(
                    f"Epoch {epoch}: Train_Loss={train_metrics['train/loss']:.4f}, "
                    f"Val_Loss={val_metrics['val/loss']:.4f}, "
                    f"InfoNCE={train_metrics['train/infonce_loss']:.4f}, "
                    f"Confidence={train_metrics['train/avg_confidence']:.3f}, "
                    f"Gate Rate={train_metrics['train/replacement_ratio']:.3f}"
                )
            else:
                # No validation - use training loss for checkpointing
                if train_metrics['train/loss'] < self.best_loss:
                    self.best_loss = train_metrics['train/loss']
                    self.save_checkpoint(epoch, train_metrics)
                    logger.info(f"Saved best checkpoint at epoch {epoch}")
                
                logger.info(
                    f"Epoch {epoch}: Loss={train_metrics['train/loss']:.4f}, "
                    f"InfoNCE={train_metrics['train/infonce_loss']:.4f}, "
                    f"Confidence={train_metrics['train/avg_confidence']:.3f}, "
                    f"Gate Rate={train_metrics['train/replacement_ratio']:.3f}"
                )
            
            # Update scheduler
            self.scheduler.step()
            
            # Log training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(key, value, epoch)
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: Loss={train_metrics['train/loss']:.4f}, "
                f"InfoNCE={train_metrics['train/infonce_loss']:.4f}, "
                f"Confidence={train_metrics['train/avg_confidence']:.3f}, "
                f"Gate Rate={train_metrics['train/replacement_ratio']:.3f}"
            )
        
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Landmark Grounding Head")
    parser.add_argument('--model_path', required=True, help="Path to pretrained NaVid model")
    parser.add_argument('--data_path', required=True, help="Path to R2R training data")
    parser.add_argument('--val_path', default=None, help="Path to R2R validation data")
    parser.add_argument('--output_dir', default='./landmark_head_output', help="Output directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")
    parser.add_argument('--lr_head', type=float, default=1e-4, help="Learning rate for landmark head")
    parser.add_argument('--overfit_small', type=int, default=0, help="Overfit on small subset for debugging")
    parser.add_argument('--max_frames', type=int, default=10, help="Maximum video frames")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Configuration
    config = {
        'num_landmark_queries': 8,
        'landmark_k': 4,
        'confidence_threshold': 0.2,
        'temperature': 1.0,
        'dropout': 0.1,
        'freeze_backbone': True,
        'lr_head': args.lr_head,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'loss_infonce': 1.0,
        'loss_diversity': 0.10,
        'loss_gate': 0.05,
        'warmup_epochs': 2
    }
    
    # Initialize trainer
    trainer = LandmarkTrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config
    )
    
    # Create dataset and dataloader
    max_instructions = args.overfit_small if args.overfit_small > 0 else -1
    train_dataset = LandmarkTrainingDataset(
        data_path=args.data_path,
        tokenizer=trainer.tokenizer,
        image_processor=trainer.image_processor,
        max_frames=args.max_frames,
        max_instructions=max_instructions
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid collate issues
        pin_memory=True
    )
    
    # Create validation dataloader if validation path is provided
    val_dataloader = None
    if args.val_path:
        val_dataset = LandmarkTrainingDataset(
            data_path=args.val_path,
            tokenizer=trainer.tokenizer,
            image_processor=trainer.image_processor,
            max_frames=args.max_frames,
            max_instructions=-1  # Use all validation data
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        logger.info(f"Loaded {len(val_dataset)} validation samples")
    
    # Start training
    trainer.train(train_dataloader, args.epochs, val_dataloader, args.patience)


if __name__ == "__main__":
    main()
