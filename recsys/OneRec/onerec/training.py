import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import os
from tqdm import tqdm
import numpy as np
from .model import OneRec
from .config import config


class OneRecDataset(Dataset):
    """
    Dataset class for OneRec training data
    """
    def __init__(self, 
                 user_features_list: List[Dict[str, torch.Tensor]],
                 short_term_features_list: List[Dict[str, torch.Tensor]],
                 pos_feedback_features_list: List[Dict[str, torch.Tensor]],
                 lifelong_features_list: List[Dict[str, torch.Tensor]],
                 target_semantic_ids_list: List[torch.Tensor],
                 user_repr_list: Optional[List[torch.Tensor]] = None,
                 item_repr_list: Optional[List[torch.Tensor]] = None,
                 industrial_metrics_list: Optional[List[torch.Tensor]] = None,
                 legal_ids_mask_list: Optional[List[torch.Tensor]] = None):
        
        self.user_features_list = user_features_list
        self.short_term_features_list = short_term_features_list
        self.pos_feedback_features_list = pos_feedback_features_list
        self.lifelong_features_list = lifelong_features_list
        self.target_semantic_ids_list = target_semantic_ids_list
        
        # Optional reward-related data
        self.user_repr_list = user_repr_list
        self.item_repr_list = item_repr_list
        self.industrial_metrics_list = industrial_metrics_list
        self.legal_ids_mask_list = legal_ids_mask_list
    
    def __len__(self):
        return len(self.user_features_list)
    
    def __getitem__(self, idx):
        item = {
            'user_features': {k: v[idx] for k, v in self.user_features_list[idx].items()},
            'short_term_features': {k: v[idx] for k, v in self.short_term_features_list[idx].items()},
            'pos_feedback_features': {k: v[idx] for k, v in self.pos_feedback_features_list[idx].items()},
            'lifelong_features': {k: v[idx] for k, v in self.lifelong_features_list[idx].items()},
            'target_semantic_ids': self.target_semantic_ids_list[idx]
        }

        # Add optional reward-related data if available
        if self.user_repr_list is not None:
            item['user_repr'] = self.user_repr_list[idx]
        if self.item_repr_list is not None:
            item['item_repr'] = self.item_repr_list[idx]
        if self.industrial_metrics_list is not None:
            item['industrial_metrics'] = self.industrial_metrics_list[idx]
        if self.legal_ids_mask_list is not None:
            item['legal_ids_mask'] = self.legal_ids_mask_list[idx]

        return item


class OneRecTrainer:
    """
    Training framework for OneRec model with pre-training and post-training
    """
    def __init__(self,
                 model: OneRec,
                 pretrain_lr: float = 1e-4,
                 posttrain_lr: float = 5e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 total_steps: int = 100000,
                 grad_clip: float = 1.0):
        
        self.model = model
        self.grad_clip = grad_clip
        
        # Optimizers for pre-training and post-training
        self.pretrain_optimizer = optim.AdamW(
            model.parameters(), 
            lr=pretrain_lr, 
            weight_decay=weight_decay
        )
        
        self.posttrain_optimizer = optim.AdamW(
            model.parameters(), 
            lr=posttrain_lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate schedulers
        self.pretrain_scheduler = optim.lr_scheduler.LinearLR(
            self.pretrain_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        
        self.posttrain_scheduler = optim.lr_scheduler.LinearLR(
            self.posttrain_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        
        self.global_step = 0
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def pretrain_step(self, 
                     user_features: Dict[str, torch.Tensor],
                     short_term_features: Dict[str, torch.Tensor],
                     pos_feedback_features: Dict[str, torch.Tensor],
                     lifelong_features: Dict[str, torch.Tensor],
                     target_semantic_ids: torch.Tensor) -> Dict[str, float]:
        """
        Single pre-training step (supervised learning with NLL loss)
        """
        self.model.train()
        self.pretrain_optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            user_features, short_term_features, pos_feedback_features, lifelong_features,
            target_semantic_ids
        )
        
        # Compute loss
        loss = outputs['nll_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update parameters
        self.pretrain_optimizer.step()
        self.pretrain_scheduler.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'lr': self.pretrain_scheduler.get_last_lr()[0] if hasattr(self.pretrain_scheduler, 'get_last_lr') else self.pretrain_optimizer.param_groups[0]['lr']
        }
    
    def posttrain_step(self,
                      user_features: Dict[str, torch.Tensor],
                      short_term_features: Dict[str, torch.Tensor],
                      pos_feedback_features: Dict[str, torch.Tensor],
                      lifelong_features: Dict[str, torch.Tensor],
                      target_semantic_ids: torch.Tensor,
                      industrial_metrics: Optional[torch.Tensor] = None,
                      legal_ids_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single post-training step (reinforcement learning with reward system)
        """
        self.model.train()
        self.posttrain_optimizer.zero_grad()

        # Compute reward and combined loss
        outputs = self.model.compute_reward_and_loss(
            user_features, short_term_features, pos_feedback_features, lifelong_features,
            target_semantic_ids, industrial_metrics, legal_ids_mask
        )

        # Use combined loss for optimization
        loss = outputs['combined_loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Update parameters
        self.posttrain_optimizer.step()
        self.posttrain_scheduler.step()

        self.global_step += 1

        return {
            'loss': loss.item(),
            'nll_loss': outputs['nll_loss'].item(),
            'pg_loss': outputs['pg_loss'].item(),
            'total_reward': outputs['total_reward'].mean().item(),
            'lr': self.posttrain_scheduler.get_last_lr()[0] if hasattr(self.posttrain_scheduler, 'get_last_lr') else self.posttrain_optimizer.param_groups[0]['lr']
        }
    
    def pretrain_epoch(self, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
        """
        Run one pre-training epoch
        """
        self.model.to(device)
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Pre-training")
        for batch in pbar:
            # Move data to device
            user_features = {k: v.to(device) for k, v in batch['user_features'].items()}
            short_term_features = {k: v.to(device) for k, v in batch['short_term_features'].items()}
            pos_feedback_features = {k: v.to(device) for k, v in batch['pos_feedback_features'].items()}
            lifelong_features = {k: v.to(device) for k, v in batch['lifelong_features'].items()}
            target_semantic_ids = batch['target_semantic_ids'].to(device)
            
            # Run pre-training step
            step_outputs = self.pretrain_step(
                user_features, short_term_features, pos_feedback_features, lifelong_features,
                target_semantic_ids
            )
            
            total_loss += step_outputs['loss']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{step_outputs['loss']:.4f}",
                'lr': f"{step_outputs['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return {'avg_loss': avg_loss}
    
    def posttrain_epoch(self, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
        """
        Run one post-training epoch
        """
        self.model.to(device)
        total_loss = 0.0
        total_nll_loss = 0.0
        total_pg_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Post-training")
        for batch in pbar:
            # Move data to device
            user_features = {k: v.to(device) for k, v in batch['user_features'].items()}
            short_term_features = {k: v.to(device) for k, v in batch['short_term_features'].items()}
            pos_feedback_features = {k: v.to(device) for k, v in batch['pos_feedback_features'].items()}
            lifelong_features = {k: v.to(device) for k, v in batch['lifelong_features'].items()}
            target_semantic_ids = batch['target_semantic_ids'].to(device)

            # Handle optional reward-related data
            user_repr = batch.get('user_repr', None)
            item_repr = batch.get('item_repr', None)
            industrial_metrics = batch.get('industrial_metrics', None)
            legal_ids_mask = batch.get('legal_ids_mask', None)

            if user_repr is not None:
                user_repr = user_repr.to(device)
            if item_repr is not None:
                item_repr = item_repr.to(device)
            if industrial_metrics is not None:
                industrial_metrics = industrial_metrics.to(device)
            if legal_ids_mask is not None:
                legal_ids_mask = legal_ids_mask.to(device)

            # Run post-training step
            step_outputs = self.posttrain_step(
                user_features, short_term_features, pos_feedback_features, lifelong_features,
                target_semantic_ids, user_repr, item_repr, industrial_metrics, legal_ids_mask
            )

            total_loss += step_outputs['loss']
            total_nll_loss += step_outputs['nll_loss']
            total_pg_loss += step_outputs['pg_loss']
            total_reward += step_outputs['total_reward']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{step_outputs['loss']:.4f}",
                'nll': f"{step_outputs['nll_loss']:.4f}",
                'pg': f"{step_outputs['pg_loss']:.4f}",
                'reward': f"{step_outputs['total_reward']:.4f}",
                'lr': f"{step_outputs['lr']:.2e}"
            })

        return {
            'avg_loss': total_loss / num_batches,
            'avg_nll_loss': total_nll_loss / num_batches,
            'avg_pg_loss': total_pg_loss / num_batches,
            'avg_reward': total_reward / num_batches
        }
    
    def pretrain(self, 
                train_dataloader: DataLoader, 
                num_epochs: int, 
                device: torch.device,
                save_path: Optional[str] = None) -> None:
        """
        Run pre-training phase
        """
        print("Starting pre-training...")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_metrics = self.pretrain_epoch(train_dataloader, device)
            print(f"Epoch {epoch + 1} - Average Loss: {epoch_metrics['avg_loss']:.4f}")
            
            # Save checkpoint
            if save_path:
                checkpoint_path = f"{save_path}_pretrain_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.pretrain_optimizer.state_dict(),
                    'loss': epoch_metrics['avg_loss']
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    def posttrain(self,
                 train_dataloader: DataLoader,
                 num_epochs: int,
                 device: torch.device,
                 save_path: Optional[str] = None) -> None:
        """
        Run post-training phase with reinforcement learning
        """
        print("Starting post-training...")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_metrics = self.posttrain_epoch(train_dataloader, device)
            print(f"Epoch {epoch + 1} - "
                  f"Average Loss: {epoch_metrics['avg_loss']:.4f}, "
                  f"NLL: {epoch_metrics['avg_nll_loss']:.4f}, "
                  f"PG: {epoch_metrics['avg_pg_loss']:.4f}, "
                  f"Reward: {epoch_metrics['avg_reward']:.4f}")
            
            # Save checkpoint
            if save_path:
                checkpoint_path = f"{save_path}_posttrain_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.posttrain_optimizer.state_dict(),
                    'loss': epoch_metrics['avg_loss']
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self,
             pretrain_dataloader: DataLoader,
             posttrain_dataloader: DataLoader,
             pretrain_epochs: int,
             posttrain_epochs: int,
             device: torch.device,
             save_path: Optional[str] = None) -> None:
        """
        Complete training pipeline: pre-training followed by post-training
        """
        # Pre-training phase
        self.pretrain(pretrain_dataloader, pretrain_epochs, device, save_path)
        
        # Post-training phase
        self.posttrain(posttrain_dataloader, posttrain_epochs, device, save_path)
        
        print("Training completed!")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the OneRec model using default config values
    model = OneRec()
    
    # Initialize trainer
    trainer = OneRecTrainer(
        model=model,
        pretrain_lr=1e-4,
        posttrain_lr=5e-5,
        weight_decay=0.01
    )
    
    # Create dummy data for testing
    batch_size = 2
    seq_len_short = 20
    seq_len_pos = 256
    seq_len_life = 2000  # Before compression
    
    # Create sample data for dataset
    user_features_list = []
    short_term_features_list = []
    pos_feedback_features_list = []
    lifelong_features_list = []
    target_semantic_ids_list = []
    user_repr_list = []
    item_repr_list = []
    industrial_metrics_list = []
    legal_ids_mask_list = []
    
    for i in range(10):  # 10 samples for testing
        # User static features
        user_features = {
            'uid': torch.randint(0, 1000000, (batch_size,)),
            'gender': torch.randint(0, 3, (batch_size,)),
            'age': torch.randint(18, 80, (batch_size,))
        }
        
        # Short-term features
        short_term_features = {
            'vid': torch.randint(0, 10000000, (batch_size, seq_len_short)),
            'aid': torch.randint(0, 5000000, (batch_size, seq_len_short)),
            'tag': torch.randn(batch_size, seq_len_short, 100),
            'ts': torch.randn(batch_size, seq_len_short),
            'playtime': torch.randn(batch_size, seq_len_short),
            'dur': torch.randn(batch_size, seq_len_short),
            'label': torch.randn(batch_size, seq_len_short, 10)
        }
        
        # Positive feedback features
        pos_feedback_features = {
            'vid': torch.randint(0, 10000000, (batch_size, seq_len_pos)),
            'aid': torch.randint(0, 5000000, (batch_size, seq_len_pos)),
            'tag': torch.randn(batch_size, seq_len_pos, 100),
            'ts': torch.randn(batch_size, seq_len_pos),
            'playtime': torch.randn(batch_size, seq_len_pos),
            'dur': torch.randn(batch_size, seq_len_pos),
            'label': torch.randn(batch_size, seq_len_pos, 10)
        }
        
        # Lifelong features
        lifelong_features = {
            'vid': torch.randint(0, 10000000, (batch_size, seq_len_life)),
            'aid': torch.randint(0, 5000000, (batch_size, seq_len_life)),
            'tag': torch.randn(batch_size, seq_len_life, 100),
            'ts': torch.randn(batch_size, seq_len_life),
            'playtime': torch.randn(batch_size, seq_len_life),
            'dur': torch.randn(batch_size, seq_len_life),
            'label': torch.randn(batch_size, seq_len_life, 10)
        }
        
        target_semantic_ids = torch.randint(0, 256, (batch_size, 3))
        user_repr = torch.randn(batch_size, 512)
        item_repr = torch.randn(batch_size, 512)
        industrial_metrics = torch.randn(batch_size, 3)
        legal_ids_mask = torch.ones(batch_size, 3, 256).bool()
        
        user_features_list.append(user_features)
        short_term_features_list.append(short_term_features)
        pos_feedback_features_list.append(pos_feedback_features)
        lifelong_features_list.append(lifelong_features)
        target_semantic_ids_list.append(target_semantic_ids)
        user_repr_list.append(user_repr)
        item_repr_list.append(item_repr)
        industrial_metrics_list.append(industrial_metrics)
        legal_ids_mask_list.append(legal_ids_mask)
    
    # Create datasets
    pretrain_dataset = OneRecDataset(
        user_features_list=user_features_list,
        short_term_features_list=short_term_features_list,
        pos_feedback_features_list=pos_feedback_features_list,
        lifelong_features_list=lifelong_features_list,
        target_semantic_ids_list=target_semantic_ids_list
    )
    
    posttrain_dataset = OneRecDataset(
        user_features_list=user_features_list,
        short_term_features_list=short_term_features_list,
        pos_feedback_features_list=pos_feedback_features_list,
        lifelong_features_list=lifelong_features_list,
        target_semantic_ids_list=target_semantic_ids_list,
        user_repr_list=user_repr_list,
        item_repr_list=item_repr_list,
        industrial_metrics_list=industrial_metrics_list,
        legal_ids_mask_list=legal_ids_mask_list
    )
    
    # Create data loaders
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=2, shuffle=True)
    # Test post-training step with internal representations
    batch = next(iter(posttrain_dataloader))
    user_features = {k: v.to(device) for k, v in batch['user_features'].items()}
    short_term_features = {k: v.to(device) for k, v in batch['short_term_features'].items()}
    pos_feedback_features = {k: v.to(device) for k, v in batch['pos_feedback_features'].items()}
    lifelong_features = {k: v.to(device) for k, v in batch['lifelong_features'].items()}
    target_semantic_ids = batch['target_semantic_ids'].to(device)
    
    # Get optional data if available
    industrial_metrics = batch.get('industrial_metrics', None)
    legal_ids_mask = batch.get('legal_ids_mask', None)
    
    if industrial_metrics is not None:
        industrial_metrics = industrial_metrics.to(device)
    if legal_ids_mask is not None:
        legal_ids_mask = legal_ids_mask.to(device)
    
    posttrain_outputs = trainer.posttrain_step(
        user_features, short_term_features, pos_feedback_features, lifelong_features,
        target_semantic_ids, industrial_metrics, legal_ids_mask
    )
    print(f"Post-training step - Loss: {posttrain_outputs['loss'].item():.4f}, "
          f"NLL: {posttrain_outputs['nll_loss'].item():.4f}, "
          f"PG: {posttrain_outputs['pg_loss'].item():.4f}, "
          f"Reward: {posttrain_outputs['total_reward'].mean().item():.4f}")
