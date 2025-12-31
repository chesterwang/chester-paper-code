import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from .config import config


class QFormer(nn.Module):
    """
    Querying Transformer (QFormer) module for compressing multimodal tokens
    """
    def __init__(self, num_query_tokens: int = 4, num_layers: int = 4, hidden_dim: int = 512):
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
    
    def forward(self, multimodal_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multimodal_tokens: Tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            Compressed tokens of shape [batch_size, num_query_tokens, hidden_dim]
        """
        batch_size = multimodal_tokens.size(0)
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        for i in range(self.num_layers):
            # Cross-attention: query tokens attend to multimodal tokens
            query_tokens = query_tokens.transpose(0, 1)  # [num_query_tokens, batch_size, hidden_dim]
            multimodal_tokens_t = multimodal_tokens.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
            
            attn_output, _ = self.cross_attn_layers[i](
                query_tokens, multimodal_tokens_t, multimodal_tokens_t
            )
            attn_output = attn_output.transpose(0, 1)  # [batch_size, num_query_tokens, hidden_dim]
            
            # Add & Norm
            query_tokens = query_tokens.transpose(0, 1) + attn_output
            query_tokens = self.ln_layers[i](query_tokens)
            
            # FFN
            ffn_output = self.ffn_layers[i](query_tokens)
            query_tokens = query_tokens + ffn_output
            query_tokens = self.ln_layers[i](query_tokens)
        
        return query_tokens


class RQKmeansTokenizer(nn.Module):
    """
    Residual Quantization K-means tokenizer for converting videos into semantic IDs
    """
    def __init__(self, num_layers: int = 3, codebook_size: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Initialize codebooks for each layer
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, hidden_dim))
            for _ in range(num_layers)
        ])
    
    def forward(self, multimodal_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            multimodal_features: Tensor of shape [batch_size, num_query_tokens, hidden_dim]
        Returns:
            semantic_ids: Tensor of shape [batch_size, num_query_tokens, num_layers]
            reconstructed: Reconstructed features of same shape as input
        """
        batch_size, num_tokens, hidden_dim = multimodal_features.shape
        semantic_ids = torch.zeros(batch_size, num_tokens, self.num_layers, dtype=torch.long, device=multimodal_features.device)
        reconstructed = torch.zeros_like(multimodal_features)
        
        residual = multimodal_features
        
        for layer_idx in range(self.num_layers):
            # Find nearest centroid for each token
            distances = torch.cdist(residual, self.codebooks[layer_idx].unsqueeze(0).expand(batch_size, -1, -1), p=2)
            indices = torch.argmin(distances, dim=-1)  # [batch_size, num_tokens]
            
            # Store semantic IDs
            semantic_ids[:, :, layer_idx] = indices
            
            # Reconstruct using selected centroids
            selected_centroids = F.embedding(indices, self.codebooks[layer_idx])
            reconstructed = reconstructed + selected_centroids
            
            # Update residual
            residual = residual - selected_centroids
        
        return semantic_ids, reconstructed


class OneRecTokenizer(nn.Module):
    """
    Complete tokenizer for OneRec system
    """
    def __init__(self,
                 num_query_tokens: int = None,
                 num_qformer_layers: int = None,
                 num_rq_layers: int = None,
                 codebook_size: int = None,
                 multimodal_hidden_dim: int = None,
                 qformer_hidden_dim: int = None):
        super().__init__()

        # Use config values with fallbacks to maintain backward compatibility
        num_query_tokens = num_query_tokens or config.num_query_tokens
        num_qformer_layers = num_qformer_layers or config.num_qformer_layers
        num_rq_layers = num_rq_layers or config.num_rq_layers
        codebook_size = codebook_size or config.codebook_size
        multimodal_hidden_dim = multimodal_hidden_dim or config.multimodal_hidden_dim
        qformer_hidden_dim = qformer_hidden_dim or config.qformer_hidden_dim

        # QFormer for multimodal token compression
        self.qformer = QFormer(
            num_query_tokens=num_query_tokens,
            num_layers=num_qformer_layers,
            hidden_dim=qformer_hidden_dim
        )

        # RQ-Kmeans for semantic tokenization
        self.rq_kmeans = RQKmeansTokenizer(
            num_layers=num_rq_layers,
            codebook_size=codebook_size,
            hidden_dim=qformer_hidden_dim
        )

        # Projection layer to map multimodal features to QFormer dimension
        self.multimodal_projection = nn.Linear(multimodal_hidden_dim, qformer_hidden_dim)
        
    def forward(self, multimodal_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            multimodal_features: Tensor of shape [batch_size, seq_len, multimodal_hidden_dim]
        Returns:
            semantic_ids: Tensor of shape [batch_size, num_query_tokens, num_rq_layers]
            reconstructed: Reconstructed features of same shape as input after compression
        """
        # Project multimodal features to QFormer dimension
        projected_features = self.multimodal_projection(multimodal_features)
        
        # Compress using QFormer
        compressed_features = self.qformer(projected_features)
        
        # Tokenize using RQ-Kmeans
        semantic_ids, reconstructed = self.rq_kmeans(compressed_features)
        
        return semantic_ids, reconstructed

    def tokenize_video(self, video_features: torch.Tensor) -> torch.Tensor:
        """
        Tokenize a single video into semantic IDs
        Args:
            video_features: Features for a single video [seq_len, multimodal_hidden_dim]
        Returns:
            semantic_ids: Semantic IDs for the video [num_query_tokens, num_rq_layers]
        """
        # Add batch dimension
        video_features = video_features.unsqueeze(0)
        
        # Get semantic IDs
        semantic_ids, _ = self.forward(video_features)
        
        # Remove batch dimension
        return semantic_ids.squeeze(0)


# Example usage and testing
if __name__ == "__main__":
    # Initialize tokenizer using default config values
    tokenizer = OneRecTokenizer()
    
    # Create dummy multimodal features for a batch of videos
    # Shape: [batch_size=2, seq_len=1280, multimodal_hidden_dim=512]
    batch_multimodal_features = torch.randn(2, 1280, 512)
    
    # Tokenize
    semantic_ids, reconstructed = tokenizer(batch_multimodal_features)
    
    print(f"Input shape: {batch_multimodal_features.shape}")
    print(f"Semantic IDs shape: {semantic_ids.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Semantic IDs range: {semantic_ids.min().item()} to {semantic_ids.max().item()}")