"""
Final integration test for OneRec model
"""
import os
import torch
from onerec.model import OneRec
from onerec.training import OneRecTrainer, OneRecDataset
from torch.utils.data import DataLoader
from onerec.config import OneRecConfig


# Change working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def test_onerec_integration():
    print("Testing OneRec integration...")

    # Update config with test parameters
    config = OneRecConfig.get_instance()

    # 单独修改某些参数
    # config.num_query_tokens = 4

    # Initialize the complete OneRec model
    model = OneRec()
    
    print(f"Model initialized successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy features for testing
    batch_size = 2
    seq_len_short = 20
    seq_len_pos = 30
    seq_len_life = 50  # Before compression
    
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
    
    # Target semantic IDs for training
    target_semantic_ids = torch.randint(0, config.codebook_size, (batch_size, config.num_rq_layers))  # 3 RQ layers
    
    # Additional data for reward computation
    industrial_metrics = torch.randn(batch_size, config.num_industrial_objectives)
    legal_ids_mask = torch.ones(batch_size, config.num_rq_layers, config.codebook_size).bool()

    print("Testing forward pass in training mode...")
    # Forward pass in training mode
    outputs = model(
        user_features, short_term_features, pos_feedback_features, lifelong_features,
        target_semantic_ids
    )

    print(f"Training mode - NLL Loss: {outputs['nll_loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    print("Testing reward and loss computation...")
    # Compute reward and loss - now using internal representations
    reward_outputs = model.compute_reward_and_loss(
        user_features, short_term_features, pos_feedback_features, lifelong_features,
        target_semantic_ids, industrial_metrics=industrial_metrics, legal_ids_mask=legal_ids_mask
    )
    
    print(f"Combined loss: {reward_outputs['combined_loss'].item():.4f}")
    print(f"Total reward mean: {reward_outputs['total_reward'].mean().item():.4f}")
    print(f"PG loss: {reward_outputs['pg_loss'].item():.4f}")
    
    print("Testing forward pass in inference mode...")
    # Forward pass in inference mode
    outputs_infer = model(
        user_features, short_term_features, pos_feedback_features, lifelong_features
    )
    
    print(f"Inference mode - Generated IDs shape: {outputs_infer['generated_ids'].shape}")
    
    print("Testing trainer initialization...")
    # Initialize trainer
    trainer = OneRecTrainer(
        model=model,
        pretrain_lr=1e-4,
        posttrain_lr=5e-5,
        weight_decay=0.01
    )
    
    print("Testing generation of recommendations...")
    # Test recommendation generation
    recommendations = model.generate_recommendations(
        user_features, short_term_features, pos_feedback_features, lifelong_features
    )
    print(f"Generated recommendations shape: {recommendations.shape}")
    
    print("All tests passed! OneRec implementation is working correctly.")


if __name__ == "__main__":
    test_onerec_integration()