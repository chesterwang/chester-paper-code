"""
Configuration class for OneRec model
This class holds all configuration parameters for the OneRec model
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class OneRecConfig:
    """Configuration class for OneRec model"""
    
    # QFormer and RQ parameters
    num_query_tokens: int = 4
    num_qformer_layers: int = 4
    num_rq_layers: int = 3
    codebook_size: int = 16
    
    # Hidden dimensions
    multimodal_hidden_dim: int = 32
    qformer_hidden_dim: int = 32
    encoder_model_dim: int = 32
    decoder_model_dim: int = 32
    
    # Encoder parameters
    num_encoder_layers: int = 6
    encoder_num_heads: int = 8
    encoder_ff_dim: int = 128
    max_seq_len: int = 2500
    
    # Vocabulary sizes
    uid_vocab_size: int = 1000
    vid_vocab_size: int = 1000
    aid_vocab_size: int = 300
    
    # Feature dimensions
    vid_dim: int = 32
    aid_dim: int = 32
    tag_dim: int = 32
    ts_dim: int = 32
    playtime_dim: int = 32
    dur_dim: int = 32
    label_dim: int = 32
    
    # Decoder parameters
    num_decoder_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ff_dim: int = 128
    num_experts: int = 8
    top_k: int = 2
    
    # Dimension parameters
    user_dim: int = 32
    item_dim: int = 32
    
    # Objective parameters
    num_objectives: int = 5
    num_industrial_objectives: int = 3
    
    # Singleton instance
    _instance: Optional['OneRecConfig'] = None
    
    @classmethod
    def get_instance(cls) -> 'OneRecConfig':
        """Get singleton instance of OneRecConfig"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def update_from_dict(cls, config_dict: dict) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls.get_instance(), key, value)