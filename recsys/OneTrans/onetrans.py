import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import torchview

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Calculate RMS norm
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale

class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        # Create position encodings matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0)]

class MixedCausalAttention(nn.Module):
    """Mixed Causal Attention with shared parameters for sequential tokens and token-specific for non-sequential"""
    def __init__(self, d_model: int, num_heads: int, seq_token_count: int, ns_token_count: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # For sequential tokens (shared parameters)
        self.q_proj_seq = nn.Linear(d_model, d_model)
        self.k_proj_seq = nn.Linear(d_model, d_model)
        self.v_proj_seq = nn.Linear(d_model, d_model)
        
        # For non-sequential tokens (token-specific parameters)
        self.q_proj_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(ns_token_count)])
        self.k_proj_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(ns_token_count)])
        self.v_proj_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(ns_token_count)])
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_token_count+ns_token_count)
        
        # Causal mask
        self.register_buffer('mask', self._create_causal_mask(seq_token_count+ns_token_count))
        
    def _create_causal_mask(self, seq_len: int):
        """Create causal mask for attention"""
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        return mask
    
    def forward(self, x: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        """
        Forward pass for mixed causal attention
        x: input tensor of shape (batch_size, seq_len, d_model)
        seq_token_count: number of sequential tokens
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Split into sequential and non-sequential tokens
        seq_tokens = x[:, :seq_token_count]
        ns_tokens = x[:, seq_token_count:]
        
        # Apply attention to sequential tokens (shared parameters)
        q_seq = self.q_proj_seq(seq_tokens).view(batch_size, seq_token_count, self.num_heads, self.head_dim)
        k_seq = self.k_proj_seq(seq_tokens).view(batch_size, seq_token_count, self.num_heads, self.head_dim)
        v_seq = self.v_proj_seq(seq_tokens).view(batch_size, seq_token_count, self.num_heads, self.head_dim)
        
        # Apply attention to non-sequential tokens (token-specific parameters)
        q_ns_list = []
        k_ns_list = []
        v_ns_list = []
        
        for i in range(len(self.q_proj_ns)):
            q_ns = self.q_proj_ns[i](ns_tokens[:, i:i+1]).view(batch_size, 1, self.num_heads, self.head_dim)
            k_ns = self.k_proj_ns[i](ns_tokens[:, i:i+1]).view(batch_size, 1, self.num_heads, self.head_dim)
            v_ns = self.v_proj_ns[i](ns_tokens[:, i:i+1]).view(batch_size, 1, self.num_heads, self.head_dim)
            
            q_ns_list.append(q_ns)
            k_ns_list.append(k_ns)
            v_ns_list.append(v_ns)
        
        # Concatenate all queries, keys, and values
        q_all = torch.cat([q_seq] + q_ns_list, dim=1)
        k_all = torch.cat([k_seq] + k_ns_list, dim=1)
        v_all = torch.cat([v_seq] + v_ns_list, dim=1)
        
        # Compute attention scores
        attn_scores = torch.matmul(q_all.transpose(1,2), k_all.transpose(1,2).transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(~self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights
        attn_output = torch.matmul(attn_weights, v_all.transpose(1,2)).transpose(1,2)
        
        # Reshape and apply output projection
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output

class MixedFeedForward(nn.Module):
    """Mixed Feed Forward Network with shared parameters for sequential tokens and token-specific for non-sequential"""
    def __init__(self, d_model: int, d_ff: int, seq_token_count: int, ns_token_count: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Shared FFN for sequential tokens
        self.ffn_seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Token-specific FFNs for non-sequential tokens
        self.ffn_ns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(ns_token_count)
        ])
        
    def forward(self, x: torch.Tensor, seq_token_count: int) -> torch.Tensor:
        """
        Forward pass for mixed feed forward
        x: input tensor of shape (batch_size, seq_len, d_model)
        seq_token_count: number of sequential tokens
        """
        batch_size, seq_len, _ = x.shape
        
        # Split into sequential and non-sequential tokens
        seq_tokens = x[:, :seq_token_count]
        ns_tokens = x[:, seq_token_count:]
        
        # Apply FFN to sequential tokens (shared parameters)
        seq_output = self.ffn_seq(seq_tokens)
        
        # Apply FFN to non-sequential tokens (token-specific parameters)
        ns_outputs = []
        for i in range(len(self.ffn_ns)):
            ns_out = self.ffn_ns[i](ns_tokens[:, i:i+1])
            ns_outputs.append(ns_out)
        
        # Combine outputs
        ns_output = torch.cat(ns_outputs, dim=1)
        
        # Combine sequential and non-sequential outputs
        output = torch.cat([seq_output, ns_output], dim=1)
        
        return output

class OneTransBlock(nn.Module):
    """OneTrans block with mixed parameterization"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seq_token_count: int, ns_token_count: int):
        super().__init__()
        self.seq_token_count = seq_token_count
        self.ns_token_count = ns_token_count
        
        # Mixed causal attention
        self.mha = MixedCausalAttention(d_model, num_heads, seq_token_count , ns_token_count)
        
        # Mixed feed forward
        self.ffn = MixedFeedForward(d_model, d_ff, seq_token_count, ns_token_count)
        
        # RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Dropout
        # self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual connection with attention
        attn_out = self.mha(self.norm1(x), self.seq_token_count)
        # attn_out = self.dropout(attn_out)
        x = x + attn_out
        
        # Second residual connection with feed forward
        ffn_out = self.ffn(self.norm2(x), self.seq_token_count)
        # ffn_out = self.dropout(ffn_out)
        x = x + ffn_out
        
        return x

class OneTrans(nn.Module):
    """Unified Transformer for recommendation with mixed parameterization"""
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_layers: int,
                 seq_token_count: int,
                 ns_token_count: int,
                 seq_pyramid_schedule: Optional[list[int]] = None):
        super().__init__()
        self.d_model = d_model
        self.seq_token_count = seq_token_count
        self.ns_token_count = ns_token_count
        self.seq_pyramid_schedule = seq_pyramid_schedule or [seq_token_count // (2 ** i) for i in range(num_layers)]
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # OneTrans blocks
        self.blocks = nn.ModuleList([
            OneTransBlock(d_model, num_heads, d_ff, seq_pyramid_schedule[layer_idx], ns_token_count)
            for layer_idx in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(d_model)
        
        # Task-specific head
        self.head = nn.Linear(d_model, 1)
        
    def forward(self, 
                seq_tokens: torch.Tensor, 
                ns_tokens: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for OneTrans model
        seq_tokens: (batch_size, seq_token_count)
        ns_tokens: (batch_size, ns_token_count)
        """
        batch_size = seq_tokens.size(0)
        
        # Create token sequence
        seq_emb = self.token_embedding(seq_tokens)
        ns_emb = self.token_embedding(ns_tokens)
        
        # Concatenate into a single sequence
        x = torch.cat([seq_emb, ns_emb], dim=1)
        
        # Pass through OneTrans blocks
        for layer_idx, block in enumerate(self.blocks):
            x = x[:, -(self.seq_pyramid_schedule[layer_idx] + self.ns_token_count):, :]
            x = block(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Take the last non-sequential token (or average) for prediction
        # In a real implementation, you'd use the NS tokens for prediction
        # Here we'll use the last NS token for demonstration
        output = self.head(x[:, -self.ns_token_count:])
        
        return output.squeeze(-1)

def demo_onetrans():
    """Demo function to execute OneTrans model calculation"""
    print("=== OneTrans Demo ===")
    
    # Model parameters
    vocab_size = 10000
    d_model = 256
    num_heads = 4
    d_ff = 512
    num_layers = 6
    seq_token_count = 100  # Number of sequential tokens
    ns_token_count = 20    # Number of non-sequential tokens
    batch_size = 8
    
    # Define pyramid schedule（schedule only for behavior sequence feature， not for non sequence feature）
    seq_pyramid_schedule = [seq_token_count // (2 ** i) for i in range(num_layers)] + [0]
    
    # Create model
    model = OneTrans(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        seq_token_count=seq_token_count,
        ns_token_count=ns_token_count,
        seq_pyramid_schedule=seq_pyramid_schedule
    )
    
    # Create dummy input data (representing token IDs)
    seq_tokens = torch.randint(0, vocab_size, (batch_size, seq_token_count))
    ns_tokens = torch.randint(0, vocab_size, (batch_size, ns_token_count))
    
    print(f"Input shapes:")
    print(f"  Sequential tokens: {seq_tokens.shape}")
    print(f"  Non-sequential tokens: {ns_tokens.shape}")
    print(f"Pyramid schedule: {seq_pyramid_schedule}")
    
    # Forward pass
    with torch.no_grad():
        output = model(seq_tokens, ns_tokens)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output[0]}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Show a brief summary of the model structure
    print("\nModel structure:")
    print(f"  - Input tokens: {seq_token_count} sequential + {ns_token_count} non-sequential")
    print(f"  - Model dimensions: {d_model}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - Pyramid schedule: {seq_pyramid_schedule}")
    
    return model, output

def visualize_model(model):
    """Visualize the model using torchview if available"""
    try:
        # Create a simple input to visualize the model
        # This assumes the model expects sequential and non-sequential tokens
        vocab_size = model.token_embedding.num_embeddings
        seq_token_count = model.seq_token_count
        ns_token_count = model.ns_token_count
        batch_size = 1
        
        seq_tokens = torch.randint(0, vocab_size, (batch_size, seq_token_count))
        ns_tokens = torch.randint(0, vocab_size, (batch_size, ns_token_count))
        
        # Visualize the model
        graph = torchview.draw_graph(model, (seq_tokens, ns_tokens), expand_nested=True, depth=2)
        graph.visual_graph.render("onetrans", format="png")
    except Exception as e:
        print(f"Error during visualization: {e}")
        return None

if __name__ == "__main__":
    model, output = demo_onetrans()
    
    # Try to visualize the model
    print("\n=== Model Visualization ===")
    graph = visualize_model(model)