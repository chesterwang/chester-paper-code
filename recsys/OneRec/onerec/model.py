import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# Import components from other modules
from .tokenizer import OneRecTokenizer
from .encoder import OneRecEncoder
from .decoder import OneRecDecoder
from .reward_system import OneRecRewardSystem
from .config import OneRecConfig


class OneRec(nn.Module):
    """
    Complete OneRec model architecture combining tokenizer, encoder, decoder, and reward system
    """
    def __init__(self):
        
        super().__init__()

        # Get singleton config instance
        config = OneRecConfig.get_instance()

        # Initialize tokenizer
        self.tokenizer = OneRecTokenizer(
        )

        # Initialize encoder
        self.encoder = OneRecEncoder(
        )

        # Initialize decoder
        self.decoder = OneRecDecoder(
        )

        # Initialize reward system
        self.reward_system = OneRecRewardSystem(
        )
        
        # Cross-attention compatibility: ensure encoder and decoder have same model dim
        assert config.encoder_model_dim == config.decoder_model_dim, \
            f"Encoder model dim ({config.encoder_model_dim}) must match decoder model dim ({config.decoder_model_dim})"
        
        self.model_dim = config.encoder_model_dim
        
        # Loss function for next token prediction
        self.nll_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self,
                user_features: Dict[str, torch.Tensor],
                short_term_features: Dict[str, torch.Tensor],
                pos_feedback_features: Dict[str, torch.Tensor],
                lifelong_features: Dict[str, torch.Tensor],
                target_semantic_ids: Optional[torch.Tensor] = None,
                multimodal_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the OneRec model
        Args:
            user_features: Dict with user static features
            short_term_features: Dict with short-term behavior features
            pos_feedback_features: Dict with positive feedback features
            lifelong_features: Dict with lifelong behavior features
            target_semantic_ids: Target semantic IDs for training [batch_size, num_rq_layers]
            multimodal_features: Multimodal features for tokenization [batch_size, seq_len, multimodal_hidden_dim]
        Returns:
            Dictionary with model outputs
        """
        outputs = {}
        
        # Encode user behavior
        encoder_output = self.encoder(
            user_features, short_term_features, pos_feedback_features, lifelong_features
        )
        outputs['encoder_output'] = encoder_output
        
        if target_semantic_ids is not None:
            # Training mode: compute next token prediction loss
            logits = self.decoder(target_semantic_ids, encoder_output)
            outputs['logits'] = logits
            
            # Compute NLL loss for next token prediction
            # For each RQ layer, we predict the next semantic ID
            # logits shape: [batch_size, num_rq_layers, vocab_size]
            # target_semantic_ids shape: [batch_size, num_rq_layers]

            # Reshape for loss computation
            logits_flat = logits.reshape(-1, logits.size(-1))  # [batch_size * num_rq_layers, vocab_size]
            targets_flat = target_semantic_ids.reshape(-1)  # [batch_size * num_rq_layers]

            nll_loss = self.nll_loss(logits_flat, targets_flat)
            outputs['nll_loss'] = nll_loss
        else:
            # Inference mode: generate semantic IDs
            # This would require a more complex generation process
            outputs['generated_ids'] = self.decoder.generate(encoder_output)
        
        return outputs
    
    def compute_reward_and_loss(self,
                               user_features: Dict[str, torch.Tensor],
                               short_term_features: Dict[str, torch.Tensor],
                               pos_feedback_features: Dict[str, torch.Tensor],
                               lifelong_features: Dict[str, torch.Tensor],
                               target_semantic_ids: torch.Tensor,
                               industrial_metrics: Optional[torch.Tensor] = None,
                               legal_ids_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute reward and combined loss for reinforcement learning
        Args:
            user_features: Dict with user static features
            short_term_features: Dict with short-term behavior features
            pos_feedback_features: Dict with positive feedback features
            lifelong_features: Dict with lifelong behavior features
            target_semantic_ids: Target semantic IDs [batch_size, num_rq_layers]
            industrial_metrics: Industrial metrics [batch_size, num_industrial_objectives]
            legal_ids_mask: Legal IDs mask [batch_size, num_rq_layers, vocab_size]
        Returns:
            Dictionary with reward and loss components
        """
        # Forward pass to get encoder output and logits
        model_outputs = self.forward(
            user_features, short_term_features, pos_feedback_features, lifelong_features,
            target_semantic_ids
        )

        # Extract user representation from encoder output
        # Take the representation corresponding to user static pathway (first position)
        encoder_output = model_outputs['encoder_output']  # [batch_size, seq_len, model_dim]
        user_repr = encoder_output[:, 0, :]  # [batch_size, model_dim]

        # Generate item representation from target semantic IDs
        # Use the average of semantic ID embeddings
        item_repr = torch.zeros_like(user_repr)  # [batch_size, model_dim]
        for layer_idx in range(OneRecConfig.get_instance().num_rq_layers):
            layer_ids = target_semantic_ids[:, layer_idx]  # [batch_size]
            layer_emb = self.decoder.semantic_id_embeddings[layer_idx](layer_ids)  # [batch_size, model_dim]
            item_repr = item_repr + layer_emb
        item_repr = item_repr / OneRecConfig.get_instance().num_rq_layers  # Average across layers

        # Compute total reward
        total_reward, reward_components = self.reward_system.compute_total_reward(
            user_repr, item_repr, target_semantic_ids, legal_ids_mask, industrial_metrics
        )

        # Compute advantages
        advantages = self.reward_system.compute_advantages(total_reward)

        # Compute ECPO loss
        # For this example, we'll use the model's output probabilities
        logits = model_outputs['logits']  # [batch_size, num_rq_layers, vocab_size]
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_rq_layers, vocab_size]

        # Expand advantages to match logits shape
        advantages_expanded = advantages.unsqueeze(1).unsqueeze(2).expand_as(logits)  # [batch_size, num_rq_layers, vocab_size]

        # Compute policy gradient loss
        log_probs = F.log_softmax(logits, dim=-1)
        pg_loss = -(log_probs * advantages_expanded).mean()

        # Combine NLL loss and policy gradient loss
        combined_loss = model_outputs['nll_loss'] + pg_loss

        results = {
            'nll_loss': model_outputs['nll_loss'],
            'pg_loss': pg_loss,
            'combined_loss': combined_loss,
            'total_reward': total_reward,
            'advantages': advantages,
            **{f'{k}_reward': v for k, v in reward_components.items()}
        }

        return results
    
    def tokenize_video(self, video_features: torch.Tensor) -> torch.Tensor:
        """
        Tokenize a video using the tokenizer component
        Args:
            video_features: Video features [seq_len, multimodal_hidden_dim]
        Returns:
            Semantic IDs [num_query_tokens, num_rq_layers]
        """
        return self.tokenizer.tokenize_video(video_features)
    
    def generate_recommendations(self,
                                user_features: Dict[str, torch.Tensor],
                                short_term_features: Dict[str, torch.Tensor],
                                pos_feedback_features: Dict[str, torch.Tensor],
                                lifelong_features: Dict[str, torch.Tensor],
                                num_recommendations: int = 10) -> torch.Tensor:
        """
        Generate recommendations for a user
        Args:
            user_features: Dict with user static features
            short_term_features: Dict with short-term behavior features
            pos_feedback_features: Dict with positive feedback features
            lifelong_features: Dict with lifelong behavior features
            num_recommendations: Number of recommendations to generate
        Returns:
            Generated semantic IDs [num_recommendations, num_rq_layers]
        """
        # Encode user behavior
        encoder_output = self.encoder(
            user_features, short_term_features, pos_feedback_features, lifelong_features
        )
        
        # Generate semantic IDs
        generated_ids = self.decoder.generate(encoder_output, max_length=OneRecConfig.get_instance().num_rq_layers)
        
        return generated_ids


# Example usage and testing
if __name__ == "__main__":
    # Initialize the complete OneRec model using default config values
    model = OneRec()
    
    # Create dummy features for testing
    batch_size = 2
    seq_len_short = 20
    seq_len_pos = 256
    seq_len_life = 2000  # Before compression
    
    # User static features
    user_features = {
        'uid': torch.randint(0, 1000000, (batch_size,)),
        'gender': torch.randint(0, 3, (batch_size,)),
        'age': torch.randint(18, 80, (batch_size,))
    }

    # - vid: Video ID - 视频的唯一标识符
    # - aid: Author ID - 视频作者或UP主的唯一标识符
    # - tag: Tag features - 视频的标签特征，用100维向量表示
    # - ts: Timestamp - 用户观看视频的时间戳
    # - playtime: Play time - 用户观看视频的实际时长
    # - dur: Duration - 视频的总时长
    # - label: User interaction labels - 用户对视频的交互行为标签，如点赞、关注、转发、不喜欢、评论、进入个人主页等，用10维向量表示
    
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
    
    # Lifelong features (compressed to 2000 before QFormer)
    lifelong_features = {
        'vid': torch.randint(0, 10000000, (batch_size, seq_len_life)),
        'aid': torch.randint(0, 5000000, (batch_size, seq_len_life)),
        'tag': torch.randn(batch_size, seq_len_life, 100),
        'ts': torch.randn(batch_size, seq_len_life),
        'playtime': torch.randn(batch_size, seq_len_life),
        'dur': torch.randn(batch_size, seq_len_life),
        'label': torch.randn(batch_size, seq_len_life, 10)
    }
    
    # Target semantic IDs for training
    target_semantic_ids = torch.randint(0, 256, (batch_size, 3))  # 3 RQ layers
    
    # Additional data for reward computation
    industrial_metrics = torch.randn(batch_size, 3)
    legal_ids_mask = torch.ones(batch_size, 3, 256).bool()

    # Forward pass in training mode
    outputs = model(
        user_features, short_term_features, pos_feedback_features, lifelong_features,
        target_semantic_ids
    )

    print(f"Training mode - NLL Loss: {outputs['nll_loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # Compute reward and loss - now using internal representations
    reward_outputs = model.compute_reward_and_loss(
        user_features, short_term_features, pos_feedback_features, lifelong_features,
        target_semantic_ids, industrial_metrics, legal_ids_mask
    )
    
    print(f"Combined loss: {reward_outputs['combined_loss'].item():.4f}")
    print(f"Total reward mean: {reward_outputs['total_reward'].mean().item():.4f}")
    print(f"PG loss: {reward_outputs['pg_loss'].item():.4f}")
    
    # Forward pass in inference mode
    outputs_infer = model(
        user_features, short_term_features, pos_feedback_features, lifelong_features
    )
    
    print(f"Inference mode - Generated IDs shape: {outputs_infer['generated_ids'].shape}")