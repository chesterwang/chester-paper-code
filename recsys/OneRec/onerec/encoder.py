import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from .config import OneRecConfig


class UserStaticPathway(nn.Module):
    """
    User static pathway: processes user static features like uid, age, gender
    """
    def __init__(self ):
        super().__init__()
        
        # Get config values
        config = OneRecConfig.get_instance()
        
        self.uid_embedding = nn.Embedding(config.uid_vocab_size, config.user_dim)  # Assuming 1M unique users
        self.gender_embedding = nn.Embedding(3, config.gender_dim)  # 0: unknown, 1: male, 2: female
        self.age_embedding = nn.Embedding(100, config.age_dim)  # 0-99 age range
        
        # Combine all static features
        total_static_dim = config.user_dim + config.gender_dim + config.age_dim
        self.projection = nn.Sequential(
            nn.Linear(total_static_dim, config.encoder_model_dim),
            nn.LeakyReLU(),
            nn.Linear(config.encoder_model_dim, config.encoder_model_dim)
        )
    
    def forward(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            user_features: Dict containing 'uid', 'gender', 'age'
        Returns:
            h_u: User static representation [batch_size, 1, model_dim]
        """
        uid_emb = self.uid_embedding(user_features['uid'])
        gender_emb = self.gender_embedding(user_features['gender'])
        age_emb = self.age_embedding(user_features['age'])
        
        # Concatenate all embeddings
        combined = torch.cat([uid_emb, gender_emb, age_emb], dim=-1)
        
        # Project to model dimension
        h_u = self.projection(combined).unsqueeze(1)  # Add sequence dimension
        
        return h_u


class ShortTermPathway(nn.Module):
    """
    Short-term behavior pathway: processes recent user interactions (L_s = 20)
    """
    def __init__(self ):
        super().__init__()
        
        config = OneRecConfig.get_instance()
        
        # Embedding layers
        self.vid_embedding = nn.Embedding(config.vid_vocab_size, config.vid_dim)  # Assuming 10M unique videos
        self.aid_embedding = nn.Embedding(config.aid_vocab_size, config.aid_dim)   # Assuming 5M unique authors
        self.tag_embedding = nn.Linear(100, config.tag_dim)  # Assuming 100-dimensional tag features
        self.ts_embedding = nn.Linear(1, config.ts_dim)       # Timestamp embedding
        self.playtime_embedding = nn.Linear(1, config.playtime_dim)
        self.dur_embedding = nn.Linear(1, config.dur_dim)
        self.label_embedding = nn.Linear(10, config.label_dim)  # Assuming 10 different labels/interactions
        
        # Projection to model dimension
        total_dim = config.vid_dim + config.aid_dim + config.tag_dim + \
            config.ts_dim + config.playtime_dim + config.dur_dim + config.label_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, config.encoder_model_dim),
            nn.LeakyReLU(),
            nn.Linear(config.encoder_model_dim, config.encoder_model_dim)
        )
    
    def forward(self, short_term_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            short_term_features: Dict containing 'vid', 'aid', 'tag', 'ts', 'playtime', 'dur', 'label'
        Returns:
            h_s: Short-term behavior representation [batch_size, seq_len, model_dim]
        """
        batch_size = short_term_features['vid'].size(0)
        
        # Process each feature type
        vid_emb = self.vid_embedding(short_term_features['vid'])  # [batch_size, seq_len, vid_dim]
        aid_emb = self.aid_embedding(short_term_features['aid'])
        tag_emb = self.tag_embedding(short_term_features['tag'])
        ts_emb = self.ts_embedding(short_term_features['ts'].unsqueeze(-1))
        playtime_emb = self.playtime_embedding(short_term_features['playtime'].unsqueeze(-1))
        dur_emb = self.dur_embedding(short_term_features['dur'].unsqueeze(-1))
        label_emb = self.label_embedding(short_term_features['label'])
        
        # Concatenate all features
        combined = torch.cat([ vid_emb, aid_emb, tag_emb, ts_emb, playtime_emb, dur_emb, label_emb ], dim=-1)
        
        # Project to model dimension
        h_s = self.projection(combined)
        
        return h_s


class PositiveFeedbackPathway(nn.Module):
    """
    Positive-feedback behavior pathway: processes high-engagement interactions (L_p = 256)
    """
    def __init__(self ):
        super().__init__()
        
        # Get config values
        config = OneRecConfig.get_instance()
        
        # Embedding layers (same as short-term but for longer sequences)
        self.vid_embedding = nn.Embedding(config.vid_vocab_size, config.vid_dim)
        self.aid_embedding = nn.Embedding(config.aid_vocab_size, config.aid_dim)
        self.tag_embedding = nn.Linear(100, config.tag_dim)
        self.ts_embedding = nn.Linear(1, config.ts_dim)
        self.playtime_embedding = nn.Linear(1, config.playtime_dim)
        self.dur_embedding = nn.Linear(1, config.dur_dim)
        self.label_embedding = nn.Linear(10, config.label_dim)
        
        # Projection to model dimension
        total_dim = config.vid_dim + config.aid_dim + config.tag_dim + config.ts_dim + config.playtime_dim + config.dur_dim + config.label_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, config.encoder_model_dim),
            nn.LeakyReLU(),
            nn.Linear(config.encoder_model_dim, config.encoder_model_dim)
        )
    
    def forward(self, pos_feedback_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pos_feedback_features: Dict containing 'vid', 'aid', 'tag', 'ts', 'playtime', 'dur', 'label'
        Returns:
            h_p: Positive feedback representation [batch_size, seq_len, model_dim]
        """
        # Process each feature type (same as short-term pathway)
        vid_emb = self.vid_embedding(pos_feedback_features['vid'])
        aid_emb = self.aid_embedding(pos_feedback_features['aid'])
        tag_emb = self.tag_embedding(pos_feedback_features['tag'])
        ts_emb = self.ts_embedding(pos_feedback_features['ts'].unsqueeze(-1))
        playtime_emb = self.playtime_embedding(pos_feedback_features['playtime'].unsqueeze(-1))
        dur_emb = self.dur_embedding(pos_feedback_features['dur'].unsqueeze(-1))
        label_emb = self.label_embedding(pos_feedback_features['label'])
        
        # Concatenate all features
        combined = torch.cat([
            vid_emb, aid_emb, tag_emb, ts_emb, 
            playtime_emb, dur_emb, label_emb
        ], dim=-1)
        
        # Project to model dimension
        h_p = self.projection(combined)
        
        return h_p


class LifelongPathway(nn.Module):
    """
    Lifelong behavior pathway: processes ultra-long user interaction histories
    Uses hierarchical compression with QFormer
    """
    def __init__(self):
        super().__init__()
        
        config = OneRecConfig.get_instance()
        # Embedding layers
        self.vid_embedding = nn.Embedding(config.uid_vocab_size, config.vid_dim)
        self.aid_embedding = nn.Embedding(config.aid_vocab_size, config.aid_dim)
        self.tag_embedding = nn.Linear(100, config.tag_dim)
        self.ts_embedding = nn.Linear(1, config.ts_dim)
        self.playtime_embedding = nn.Linear(1, config.playtime_dim)
        self.dur_embedding = nn.Linear(1, config.dur_dim)
        self.label_embedding = nn.Linear(10, config.label_dim)
        
        # Projection to model dimension
        total_dim = config.vid_dim + config.aid_dim + config.tag_dim + config.ts_dim + config.playtime_dim + config.dur_dim + config.label_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, config.encoder_model_dim),
            nn.LeakyReLU(),
            nn.Linear(config.encoder_model_dim, config.encoder_model_dim)
        )
        
        # QFormer for compression
        self.qformer = QFormer(
            num_query_tokens=config.num_query_tokens,
            num_layers=config.num_encoder_layers,
            hidden_dim=config.encoder_model_dim
        )
    
    def forward(self, lifelong_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            lifelong_features: Dict containing 'vid', 'aid', 'tag', 'ts', 'playtime', 'dur', 'label'
        Returns:
            h_l: Compressed lifelong behavior representation [batch_size, num_query_tokens, model_dim]
        """
        # Process each feature type
        vid_emb = self.vid_embedding(lifelong_features['vid'])
        aid_emb = self.aid_embedding(lifelong_features['aid'])
        tag_emb = self.tag_embedding(lifelong_features['tag'])
        ts_emb = self.ts_embedding(lifelong_features['ts'].unsqueeze(-1))
        playtime_emb = self.playtime_embedding(lifelong_features['playtime'].unsqueeze(-1))
        dur_emb = self.dur_embedding(lifelong_features['dur'].unsqueeze(-1))
        label_emb = self.label_embedding(lifelong_features['label'])
        
        # Concatenate all features
        combined = torch.cat([
            vid_emb, aid_emb, tag_emb, ts_emb, 
            playtime_emb, dur_emb, label_emb
        ], dim=-1)
        
        # Project to model dimension
        v_l = self.projection(combined)
        
        # Apply QFormer to compress the long sequence
        h_l = self.qformer(v_l)
        
        return h_l


class QFormer(nn.Module):
    """
    Querying Transformer (QFormer) module for compressing long sequences
    """
    def __init__(self, num_query_tokens: int = 128, num_layers: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim))
        
        # Cross-attention and FFN layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8) for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, long_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            long_sequence: Tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            Compressed tokens of shape [batch_size, num_query_tokens, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = long_sequence.size()
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        for i in range(self.num_layers):
            # Cross-attention: query tokens attend to long sequence
            query_tokens_t = query_tokens.transpose(0, 1)  # [num_query_tokens, batch_size, hidden_dim]
            long_seq_t = long_sequence.transpose(0, 1)     # [seq_len, batch_size, hidden_dim]
            
            attn_output, _ = self.cross_attn_layers[i](
                query_tokens_t, long_seq_t, long_seq_t
            )
            attn_output = attn_output.transpose(0, 1)  # [batch_size, num_query_tokens, hidden_dim]
            
            # Add & Norm
            query_tokens = query_tokens + attn_output
            query_tokens = self.ln_layers[i](query_tokens)
            
            # FFN
            ffn_output = self.ffn_layers[i](query_tokens)
            query_tokens = query_tokens + ffn_output
            query_tokens = self.ln_layers[i](query_tokens)
        
        return query_tokens


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward network
    """
    def __init__(self):
        super().__init__()

        config = OneRecConfig.get_instance()

        self.self_attn = nn.MultiheadAttention(embed_dim=config.encoder_model_dim, num_heads=config.encoder_num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(config.encoder_model_dim, config.encoder_ff_dim),
            nn.GELU(),
            nn.Linear(config.encoder_ff_dim, config.encoder_model_dim)
        )
        self.norm1 = nn.LayerNorm(config.encoder_model_dim)
        self.norm2 = nn.LayerNorm(config.encoder_model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        
        return x


class OneRecEncoder(nn.Module):
    """
    Complete OneRec encoder with multi-scale feature engineering
    """
    def __init__(self):
        super().__init__()

        config = OneRecConfig.get_instance()
        # Use config values with fallbacks to maintain backward compatibility

        self.model_dim = config.encoder_model_dim
        self.num_encoder_layers = config.num_encoder_layers

        # Multi-scale pathways
        self.user_static_pathway = UserStaticPathway(
                                                     )
        self.short_term_pathway = ShortTermPathway(
        )
        self.positive_feedback_pathway = PositiveFeedbackPathway()
        self.lifelong_pathway = LifelongPathway(
                                                )

        # Positional embeddings
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.encoder_model_dim)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([ 
            TransformerEncoderLayer() for _ in range(config.num_encoder_layers)
        ])

        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.encoder_model_dim)
    
    def forward(self, 
                user_features: Dict[str, torch.Tensor],
                short_term_features: Dict[str, torch.Tensor],
                pos_feedback_features: Dict[str, torch.Tensor],
                lifelong_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            user_features: Dict with user static features
            short_term_features: Dict with short-term behavior features
            pos_feedback_features: Dict with positive feedback features
            lifelong_features: Dict with lifelong behavior features
        Returns:
            z_enc: Encoded representation [batch_size, total_seq_len, model_dim]
        """
        batch_size = user_features['uid'].size(0)
        
        # Process each pathway
        h_u = self.user_static_pathway(user_features)  # [batch_size, 1, model_dim]
        h_s = self.short_term_pathway(short_term_features)  # [batch_size, 20, model_dim]
        h_p = self.positive_feedback_pathway(pos_feedback_features)  # [batch_size, 256, model_dim]
        h_l = self.lifelong_pathway(lifelong_features)  # [batch_size, 128, model_dim]
        
        # Concatenate all pathway outputs
        z_1 = torch.cat([h_u, h_s, h_p, h_l], dim=1)  # [batch_size, total_seq_len, model_dim]
        
        # Add positional embeddings
        total_seq_len = z_1.size(1)
        positions = torch.arange(0, total_seq_len, device=z_1.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        z_1 = z_1 + pos_emb
        
        # Apply transformer encoder layers
        z_enc = z_1
        for layer in self.encoder_layers:
            z_enc = layer(z_enc)
        
        # Apply final layer norm
        z_enc = self.layer_norm(z_enc)
        
        return z_enc


# Example usage and testing
if __name__ == "__main__":
    # Initialize encoder using default config values
    encoder = OneRecEncoder()
    
    # Create dummy features for testing
    batch_size = 2
    seq_len_short = 20
    seq_len_pos = 256
    seq_len_life = 2000  # Before compression
    
    # Get config values
    config = OneRecConfig.get_instance()

    # User static features
    user_features = {
        'uid': torch.randint(0, config.uid_vocab_size, (batch_size,)),
        'gender': torch.randint(0, 3, (batch_size,)),
        'age': torch.randint(18, 80, (batch_size,))
    }
    
    # Short-term features
    short_term_features = {
        'vid': torch.randint(0, config.vid_vocab_size, (batch_size, seq_len_short)),
        'aid': torch.randint(0, config.aid_vocab_size, (batch_size, seq_len_short)),
        'tag': torch.randn(batch_size, seq_len_short, 100),
        'ts': torch.randn(batch_size, seq_len_short),
        'playtime': torch.randn(batch_size, seq_len_short),
        'dur': torch.randn(batch_size, seq_len_short),
        'label': torch.randn(batch_size, seq_len_short, 10)
    }
    
    # Positive feedback features
    pos_feedback_features = {
        'vid': torch.randint(0, config.vid_vocab_size, (batch_size, seq_len_pos)),
        'aid': torch.randint(0, config.aid_vocab_size, (batch_size, seq_len_pos)),
        'tag': torch.randn(batch_size, seq_len_pos, 100),
        'ts': torch.randn(batch_size, seq_len_pos),
        'playtime': torch.randn(batch_size, seq_len_pos),
        'dur': torch.randn(batch_size, seq_len_pos),
        'label': torch.randn(batch_size, seq_len_pos, 10)
    }
    
    # Lifelong features (compressed to 2000 before QFormer)
    lifelong_features = {
        'vid': torch.randint(0, config.vid_vocab_size, (batch_size, seq_len_life)),
        'aid': torch.randint(0, config.aid_vocab_size, (batch_size, seq_len_life)),
        'tag': torch.randn(batch_size, seq_len_life, 100),
        'ts': torch.randn(batch_size, seq_len_life),
        'playtime': torch.randn(batch_size, seq_len_life),
        'dur': torch.randn(batch_size, seq_len_life),
        'label': torch.randn(batch_size, seq_len_life, 10)
    }
    
    # Encode
    z_enc = encoder(user_features, short_term_features, pos_feedback_features, lifelong_features)
    
    print(f"Encoded representation shape: {z_enc.shape}")
    print(f"Expected shape: [{batch_size}, {1 + seq_len_short + seq_len_pos + 128}, 512]")