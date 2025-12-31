import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from .config import config


class PreferenceScoreTower(nn.Module):
    """
    Preference Score Tower for learning personalized fusion score
    """
    def __init__(self, 
                 user_dim: int = 512,
                 item_dim: int = 512,
                 hidden_dim: int = 1024,
                 num_objectives: int = 5,  # ctr, lvtr, ltr, vtr, etc.
                 tower_hidden_dim: int = 512):
        super().__init__()
        
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.num_objectives = num_objectives
        
        # Separate towers for different objectives
        self.objective_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(user_dim + item_dim, tower_hidden_dim),
                nn.ReLU(),
                nn.Linear(tower_hidden_dim, tower_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(tower_hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_objectives)
        ])
        
        # Final MLP to combine tower outputs
        self.final_mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim + num_objectives * tower_hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_repr: User representation [batch_size, user_dim]
            item_repr: Item representation [batch_size, item_dim]
        Returns:
            P-Score [batch_size, 1]
        """
        batch_size = user_repr.size(0)
        
        # Concatenate user and item representations
        user_item_repr = torch.cat([user_repr, item_repr], dim=-1)  # [batch_size, user_dim + item_dim]
        
        # Process through objective towers
        tower_outputs = []
        tower_intermediates = []
        
        for tower in self.objective_towers:
            # Each tower takes user_item_repr and produces intermediate features and final output
            tower_input = user_item_repr
            intermediate = tower_input
            for i, layer in enumerate(tower):
                intermediate = layer(intermediate)
                if i == len(tower) - 2:  # Second to last layer (before sigmoid)
                    tower_intermediates.append(intermediate)
        
        # Concatenate user, item, and intermediate tower outputs
        final_input = torch.cat([user_repr, item_repr] + tower_intermediates, dim=-1)
        
        # Final MLP to get P-Score
        p_score = self.final_mlp(final_input)
        
        return p_score


class EarlyClippedGRPO(nn.Module):
    """
    Early Clipped Group Relative Policy Optimization
    """
    def __init__(self, epsilon: float = 0.2, delta: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
    
    def forward(self, 
                old_policy_probs: torch.Tensor,
                new_policy_probs: torch.Tensor,
                advantages: torch.Tensor,
                old_policy_probs_clipped: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            old_policy_probs: Old policy probabilities [batch_size, vocab_size]
            new_policy_probs: New policy probabilities [batch_size, vocab_size]
            advantages: Advantages [batch_size]
            old_policy_probs_clipped: Clipped old policy probabilities (optional)
        Returns:
            ECPO loss
        """
        # Calculate policy ratio
        ratio = new_policy_probs / (old_policy_probs + 1e-8)
        
        # If advantages are positive, use standard PPO clipping
        # If advantages are negative, use early clipping with delta
        advantages_expanded = advantages.unsqueeze(-1).expand_as(ratio)
        
        # Calculate clipped ratio based on advantages
        pos_adv_mask = advantages_expanded > 0
        neg_adv_mask = advantages_expanded <= 0
        
        # For positive advantages: standard clipping
        clipped_ratio_pos = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        
        # For negative advantages: early clipping with delta
        if old_policy_probs_clipped is None:
            old_policy_probs_clipped = torch.clamp(old_policy_probs, min=1e-8)
        clipped_ratio_neg = torch.clamp(ratio, 1 - self.epsilon - self.delta, 1 + self.epsilon + self.delta)
        
        # Apply masks
        clipped_ratio = torch.where(
            pos_adv_mask, 
            clipped_ratio_pos, 
            clipped_ratio_neg
        )
        
        # Calculate unclipped and clipped surrogate objectives
        unclipped_surrogate = ratio * advantages_expanded
        clipped_surrogate = clipped_ratio * advantages_expanded
        
        # Return minimum of both (as in PPO)
        surrogate_loss = torch.min(unclipped_surrogate, clipped_surrogate)
        
        # Return negative for maximization
        return -surrogate_loss.mean()


class FormatReward(nn.Module):
    """
    Format reward for ensuring legal generation of semantic IDs
    """
    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
    
    def forward(self, generated_ids: torch.Tensor, legal_ids_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated_ids: Generated semantic IDs [batch_size, num_rq_layers]
            legal_ids_mask: Boolean mask indicating legal IDs [batch_size, num_rq_layers, vocab_size]
        Returns:
            Format rewards [batch_size]
        """
        batch_size, num_layers = generated_ids.size()
        
        # Calculate legality for each generated ID
        legality_scores = []
        for i in range(num_layers):
            layer_ids = generated_ids[:, i]  # [batch_size]
            layer_legal_mask = legal_ids_mask[:, i, :]  # [batch_size, vocab_size]
            
            # Get legality for each ID in the batch
            batch_legal = layer_legal_mask[torch.arange(batch_size), layer_ids]  # [batch_size]
            legality_scores.append(batch_legal.float())
        
        # Average legality across layers
        avg_legality = torch.stack(legality_scores, dim=1).mean(dim=1)  # [batch_size]
        
        return avg_legality


class IndustrialReward(nn.Module):
    """
    Industrial scenario alignment reward
    """
    def __init__(self, num_industrial_objectives: int = 3):
        super().__init__()
        self.num_industrial_objectives = num_industrial_objectives
        
        # Simple linear combination of industrial objectives
        self.reward_weights = nn.Parameter(torch.ones(num_industrial_objectives))
    
    def forward(self, industrial_metrics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            industrial_metrics: Industrial metrics [batch_size, num_industrial_objectives]
        Returns:
            Industrial rewards [batch_size]
        """
        # Weighted sum of industrial objectives
        weighted_rewards = industrial_metrics * self.reward_weights.unsqueeze(0)
        total_reward = weighted_rewards.sum(dim=1)  # [batch_size]
        
        return total_reward


class OneRecRewardSystem(nn.Module):
    """
    Complete reward system for OneRec with preference alignment, format reward, and industrial reward
    """
    def __init__(self,
                 user_dim: int = None,
                 item_dim: int = None,
                 vocab_size: int = None,
                 num_rq_layers: int = None,
                 num_objectives: int = None,
                 num_industrial_objectives: int = None):
        super().__init__()

        # Use config values with fallbacks to maintain backward compatibility
        user_dim = user_dim or config.user_dim
        item_dim = item_dim or config.item_dim
        vocab_size = vocab_size or config.codebook_size
        num_rq_layers = num_rq_layers or config.num_rq_layers
        num_objectives = num_objectives or config.num_objectives
        num_industrial_objectives = num_industrial_objectives or config.num_industrial_objectives

        # Preference reward components
        self.preference_tower = PreferenceScoreTower(
            user_dim=user_dim,
            item_dim=item_dim,
            num_objectives=num_objectives
        )

        # Format reward
        self.format_reward = FormatReward(vocab_size=vocab_size)

        # Industrial reward
        self.industrial_reward = IndustrialReward(num_industrial_objectives=num_industrial_objectives)

        # ECPO optimizer
        self.ecpo_optimizer = EarlyClippedGRPO()

        # Weight parameters for combining different rewards
        self.preference_weight = nn.Parameter(torch.tensor(1.0))
        self.format_weight = nn.Parameter(torch.tensor(0.5))
        self.industrial_weight = nn.Parameter(torch.tensor(0.3))
    
    def compute_preference_reward(self, user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute preference reward using P-Score
        Args:
            user_repr: User representation [batch_size, user_dim]
            item_repr: Item representation [batch_size, item_dim]
        Returns:
            Preference rewards [batch_size]
        """
        p_scores = self.preference_tower(user_repr, item_repr)
        return p_scores.squeeze(-1)  # [batch_size]
    
    def compute_format_reward(self, generated_ids: torch.Tensor, legal_ids_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute format reward for legal generation
        Args:
            generated_ids: Generated semantic IDs [batch_size, num_rq_layers]
            legal_ids_mask: Legal IDs mask [batch_size, num_rq_layers, vocab_size]
        Returns:
            Format rewards [batch_size]
        """
        return self.format_reward(generated_ids, legal_ids_mask)
    
    def compute_industrial_reward(self, industrial_metrics: torch.Tensor) -> torch.Tensor:
        """
        Compute industrial reward
        Args:
            industrial_metrics: Industrial metrics [batch_size, num_industrial_objectives]
        Returns:
            Industrial rewards [batch_size]
        """
        return self.industrial_reward(industrial_metrics)
    
    def compute_total_reward(self,
                           user_repr: torch.Tensor,
                           item_repr: torch.Tensor,
                           generated_ids: torch.Tensor,
                           legal_ids_mask: Optional[torch.Tensor] = None,
                           industrial_metrics: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total reward combining all components
        Args:
            user_repr: User representation [batch_size, user_dim]
            item_repr: Item representation [batch_size, item_dim]
            generated_ids: Generated semantic IDs [batch_size, num_rq_layers]
            legal_ids_mask: Legal IDs mask [batch_size, num_rq_layers, vocab_size]
            industrial_metrics: Industrial metrics [batch_size, num_industrial_objectives]
        Returns:
            Total rewards [batch_size] and individual reward components
        """
        # Compute individual rewards
        preference_reward = self.compute_preference_reward(user_repr, item_repr)
        format_reward = self.compute_format_reward(generated_ids, legal_ids_mask) if legal_ids_mask is not None else torch.zeros_like(preference_reward)
        industrial_reward = self.compute_industrial_reward(industrial_metrics) if industrial_metrics is not None else torch.zeros_like(preference_reward)

        # Combine rewards with learned weights
        total_reward = (
            self.preference_weight * preference_reward +
            self.format_weight * format_reward +
            self.industrial_weight * industrial_reward
        )

        reward_components = {
            'preference': preference_reward,
            'format': format_reward,
            'industrial': industrial_reward
        }

        return total_reward, reward_components
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages from rewards (mean-centering)
        Args:
            rewards: Rewards [batch_size]
        Returns:
            Advantages [batch_size]
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        return advantages
    
    def compute_ecpo_loss(self,
                         old_policy_probs: torch.Tensor,
                         new_policy_probs: torch.Tensor,
                         advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute ECPO loss
        Args:
            old_policy_probs: Old policy probabilities [batch_size, vocab_size]
            new_policy_probs: New policy probabilities [batch_size, vocab_size]
            advantages: Advantages [batch_size]
        Returns:
            ECPO loss
        """
        return self.ecpo_optimizer(old_policy_probs, new_policy_probs, advantages)


# Example usage and testing
if __name__ == "__main__":
    # Initialize reward system using default config values
    reward_system = OneRecRewardSystem()
    
    # Create dummy inputs
    batch_size = 4
    user_repr = torch.randn(batch_size, 512)
    item_repr = torch.randn(batch_size, 512)
    generated_ids = torch.randint(0, 256, (batch_size, 3))
    legal_ids_mask = torch.ones(batch_size, 3, 256).bool()
    industrial_metrics = torch.randn(batch_size, 3)
    
    # Compute total reward
    total_reward, reward_components = reward_system.compute_total_reward(
        user_repr, item_repr, generated_ids, legal_ids_mask, industrial_metrics
    )
    
    print(f"Total reward shape: {total_reward.shape}")
    print(f"Total reward mean: {total_reward.mean().item():.4f}")
    print(f"Preference reward mean: {reward_components['preference'].mean().item():.4f}")
    print(f"Format reward mean: {reward_components['format'].mean().item():.4f}")
    print(f"Industrial reward mean: {reward_components['industrial'].mean().item():.4f}")
    
    # Compute advantages
    advantages = reward_system.compute_advantages(total_reward)
    print(f"Advantages mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
    
    # Example ECPO loss computation
    old_policy_probs = torch.softmax(torch.randn(batch_size, 256), dim=-1)
    new_policy_probs = torch.softmax(torch.randn(batch_size, 256), dim=-1)
    ecpo_loss = reward_system.compute_ecpo_loss(old_policy_probs, new_policy_probs, advantages)
    print(f"ECPO loss: {ecpo_loss.item():.4f}")