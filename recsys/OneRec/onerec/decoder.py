import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from .config import config


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for decoder
    """
    def __init__(self, model_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, model_dim]
        Returns:
            Output tensor of same shape with causal attention applied
        """
        batch_size, seq_len, model_dim = x.size()
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        output = self.out_proj(attention_output)
        
        return output


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for attending to encoder output
    """
    def __init__(self, model_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, model_dim]
            key: Key tensor of shape [batch_size, seq_len_k, model_dim]
            value: Value tensor of shape [batch_size, seq_len_k, model_dim]
        Returns:
            Output tensor of shape [batch_size, seq_len_q, model_dim]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.model_dim)
        output = self.out_proj(attention_output)
        
        return output


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer for decoder
    """
    def __init__(self, model_dim: int = 512, ff_dim: int = 2048, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, model_dim)
            ) for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(model_dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, model_dim]
        Returns:
            Output tensor of same shape processed by selected experts
        """
        batch_size, seq_len, model_dim = x.size()
        
        # Flatten for routing
        x_flat = x.view(-1, model_dim)  # [batch_size * seq_len, model_dim]
        
        # Get routing weights
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)  # [batch_size * seq_len, num_experts]
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)  # [batch_size * seq_len, top_k]
        
        # Normalize top-k weights
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Process with selected experts
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            # Get expert indices for this top-k position
            expert_indices = top_k_indices[:, i]  # [batch_size * seq_len]
            weights = top_k_weights[:, i].unsqueeze(-1)  # [batch_size * seq_len, 1]
            
            # Process each expert separately
            for expert_idx in range(self.num_experts):
                # Create mask for this expert
                expert_mask = (expert_indices == expert_idx)
                if expert_mask.any():
                    # Get inputs for this expert
                    expert_input = x_flat[expert_mask]  # [num_inputs_for_expert, model_dim]
                    expert_output = self.experts[expert_idx](expert_input)  # [num_inputs_for_expert, model_dim]
                    
                    # Apply weights and update final output
                    weighted_output = expert_output * weights[expert_mask]
                    final_output[expert_mask] += weighted_output
        
        # Reshape back
        output = final_output.view(batch_size, seq_len, model_dim)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with causal self-attention, cross-attention, and MoE
    """
    def __init__(self, model_dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, 
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.causal_self_attn = CausalSelfAttention(model_dim, num_heads)
        self.cross_attn = CrossAttention(model_dim, num_heads)
        self.moe = MixtureOfExperts(model_dim, ff_dim, num_experts, top_k)
        
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder input [batch_size, seq_len, model_dim]
            encoder_output: Encoder output [batch_size, enc_seq_len, model_dim]
        Returns:
            Output tensor of same shape as input
        """
        # Causal self-attention
        attn_output = self.causal_self_attn(x)
        x = x + attn_output
        x = self.norm1(x)
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output)
        x = x + cross_attn_output
        x = self.norm2(x)
        
        # Mixture of Experts
        moe_output = self.moe(x)
        x = x + moe_output
        x = self.norm3(x)
        
        return x


class OneRecDecoder(nn.Module):
    """
    OneRec decoder for generating semantic IDs
    """
    def __init__(self,
                 vocab_size: int = None,  # Size of codebook
                 num_rq_layers: int = None,  # Number of RQ layers
                 model_dim: int = None,
                 num_decoder_layers: int = None,
                 num_heads: int = None,
                 ff_dim: int = None,
                 num_experts: int = None,
                 top_k: int = None):
        super().__init__()

        # Use config values with fallbacks to maintain backward compatibility
        vocab_size = vocab_size or config.codebook_size
        num_rq_layers = num_rq_layers or config.num_rq_layers
        model_dim = model_dim or config.decoder_model_dim
        num_decoder_layers = num_decoder_layers or config.num_decoder_layers
        num_heads = num_heads or config.decoder_num_heads
        ff_dim = ff_dim or config.decoder_ff_dim
        num_experts = num_experts or config.num_experts
        top_k = top_k or config.top_k

        self.vocab_size = vocab_size
        self.num_rq_layers = num_rq_layers
        self.model_dim = model_dim

        # Embedding for semantic IDs (for each RQ layer)
        self.semantic_id_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(num_rq_layers)
        ])

        # Beginning of sequence token embedding
        self.bos_embedding = nn.Embedding(1, model_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, num_heads, ff_dim, num_experts, top_k)
            for _ in range(num_decoder_layers)
        ])

        # Output projection for each RQ layer
        self.output_projections = nn.ModuleList([
            nn.Linear(model_dim, vocab_size) for _ in range(num_rq_layers)
        ])

        # Final layer norm
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, 
                semantic_ids: torch.Tensor, 
                encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_ids: Semantic IDs for target video [batch_size, num_rq_layers]
            encoder_output: Encoder output [batch_size, enc_seq_len, model_dim]
        Returns:
            Logits for next semantic ID prediction [batch_size, num_rq_layers, vocab_size]
        """
        batch_size = semantic_ids.size(0)
        
        # Create input sequence by concatenating BOS token with semantic IDs
        # Each semantic ID layer gets its own embedding
        embedded_layers = []
        
        # Add BOS token embedding
        bos_emb = self.bos_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=semantic_ids.device))
        embedded_layers.append(bos_emb)
        
        # Add embeddings for each RQ layer
        for layer_idx in range(self.num_rq_layers):
            layer_ids = semantic_ids[:, layer_idx:layer_idx+1]  # [batch_size, 1]
            layer_emb = self.semantic_id_embeddings[layer_idx](layer_ids)  # [batch_size, 1, model_dim]
            embedded_layers.append(layer_emb)
        
        # Concatenate all embeddings
        d_m = torch.cat(embedded_layers, dim=1)  # [ num_rq_layers , batch_size, model_dim]
        
        # Apply decoder layers
        d_m_out = d_m
        for layer in self.decoder_layers:
            d_m_out = layer(d_m_out, encoder_output)
        
        # Apply final layer norm
        d_m_out = self.layer_norm(d_m_out)
        
        # Generate logits for next token prediction
        # We predict the next token for each position
        logits_layers = []
        for layer_idx in range(self.num_rq_layers):
            # Use the representation at the position of the current semantic ID
            # to predict the next one (shifted by 1 due to BOS)
            layer_repr = d_m_out[:, layer_idx:layer_idx+1, :]  # [batch_size, 1, model_dim]
            layer_logits = self.output_projections[layer_idx](layer_repr)  # [batch_size, 1, vocab_size]
            logits_layers.append(layer_logits)
        
        # Concatenate logits for all layers
        logits = torch.cat(logits_layers, dim=1)  # [batch_size, num_rq_layers, vocab_size]
        
        return logits
    
    def generate(self, 
                 encoder_output: torch.Tensor, 
                 max_length: int = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate semantic IDs autoregressively
        Args:
            encoder_output: Encoder output [batch_size, enc_seq_len, model_dim]
            max_length: Maximum generation length (should be num_rq_layers)
            temperature: Temperature for sampling
        Returns:
            Generated semantic IDs [batch_size, num_rq_layers]
        """
        if max_length is None:
            max_length = self.num_rq_layers
            
        batch_size = encoder_output.size(0)
        
        # Start with BOS token
        generated_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=encoder_output.device)
        
        for step in range(max_length):
            # Create input sequence with BOS and previously generated IDs
            if step == 0:
                # Only BOS token initially
                input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_output.device)
            else:
                # Concatenate BOS with generated IDs
                bos_token = torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_output.device)
                input_ids = torch.cat([bos_token, generated_ids], dim=1)
            
            # Get embeddings for input
            if step == 0:
                d_m = self.bos_embedding(input_ids)  # [batch_size, 1, model_dim]
            else:
                embedded_layers = [self.bos_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_output.device))]
                for layer_idx in range(step):
                    layer_emb = self.semantic_id_embeddings[layer_idx](generated_ids[:, layer_idx:layer_idx+1])
                    embedded_layers.append(layer_emb)
                d_m = torch.cat(embedded_layers, dim=1)  # [batch_size, step+1, model_dim]
            
            # Apply decoder layers
            d_m_out = d_m
            for layer in self.decoder_layers:
                d_m_out = layer(d_m_out, encoder_output)
            
            # Apply final layer norm
            d_m_out = self.layer_norm(d_m_out)
            
            # Get logits for next token prediction for current layer
            current_repr = d_m_out[:, step:step+1, :]  # [batch_size, 1, model_dim]
            current_logits = self.output_projections[step](current_repr)  # [batch_size, 1, vocab_size]
            current_logits = current_logits.squeeze(1) / temperature  # [batch_size, vocab_size]
            
            # Sample next token
            next_token = torch.multinomial(F.softmax(current_logits, dim=-1), num_samples=1).squeeze(-1)  # [batch_size]
            next_token = next_token.unsqueeze(1)  # [batch_size, 1]
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids


# Example usage and testing
if __name__ == "__main__":
    # Initialize decoder using default config values
    decoder = OneRecDecoder()
    
    # Create dummy encoder output
    batch_size = 2
    enc_seq_len = 400  # From encoder (1 + 20 + 256 + 128)
    encoder_output = torch.randn(batch_size, enc_seq_len, 512)
    
    # Create dummy semantic IDs (for training)
    semantic_ids = torch.randint(0, 256, (batch_size, 3))  # 3 RQ layers
    
    # Forward pass
    logits = decoder(semantic_ids, encoder_output)
    
    print(f"Input semantic IDs shape: {semantic_ids.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Generate semantic IDs
    generated_ids = decoder.generate(encoder_output, max_length=3)
    print(f"Generated IDs shape: {generated_ids.shape}")
    print(f"Generated IDs: {generated_ids}")