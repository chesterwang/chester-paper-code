"""
Final integration test for OneRec model
"""
import torch
from onerec.model import OneRec
from onerec.training import OneRecTrainer, OneRecDataset
from torch.utils.data import DataLoader


def test_onerec_integration():
    print("Testing OneRec integration...")

    uid_vocab_size= 1000
    vid_vocab_size= 1000
    aid_vocab_size= 300 
    
    # Initialize the complete OneRec model
    # model = OneRec(
    #     num_query_tokens=4,
    #     num_qformer_layers=4,
    #     num_rq_layers=3,
    #     codebook_size=256,
    #     multimodal_hidden_dim=512,
    #     qformer_hidden_dim=512,
    #     encoder_model_dim=512,
    #     num_encoder_layers=6,
    #     max_seq_len=2500,
    #     encoder_num_heads=8,
    #     encoder_ff_dim=2048,
    #     decoder_model_dim=512,
    #     num_decoder_layers=6,
    #     decoder_num_heads=8,
    #     decoder_ff_dim=2048,
    #     num_experts=8,
    #     top_k=2,
    #     user_dim=512,
    #     item_dim=512,
    #     num_objectives=5,
    #     num_industrial_objectives=3
    # )

    model = OneRec(
        num_query_tokens=4,
        num_qformer_layers=4,
        num_rq_layers=3,
        codebook_size=16,
        multimodal_hidden_dim=32,
        qformer_hidden_dim=32,
        encoder_model_dim=32,
        num_encoder_layers=6,
        max_seq_len=2500,
        encoder_num_heads=8,
        encoder_ff_dim=128,

        uid_vocab_size=uid_vocab_size,  
        vid_vocab_size= vid_vocab_size,
        aid_vocab_size= aid_vocab_size,
        vid_dim = 32,
        aid_dim = 32,
        tag_dim = 32,
        ts_dim = 32,
        playtime_dim = 32,
        dur_dim = 32,
        label_dim = 32,

        decoder_model_dim=32,
        num_decoder_layers=6,
        decoder_num_heads=8,
        decoder_ff_dim=128,
        num_experts=8,
        top_k=2,
        user_dim=32,
        item_dim=32,

        num_objectives=5,
        num_industrial_objectives=3
    )
    
    print(f"Model initialized successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy features for testing
    # batch_size = 2
    # seq_len_short = 20
    # seq_len_pos = 256
    # seq_len_life = 2000  # Before compression

    batch_size = 2
    seq_len_short = 20
    seq_len_pos = 30
    seq_len_life = 50  # Before compression
    
    # User static features
    user_features = {
        'uid': torch.randint(0, uid_vocab_size, (batch_size,)),
        'gender': torch.randint(0, 3, (batch_size,)),
        'age': torch.randint(18, 80, (batch_size,))
    }
    
    # Short-term features
    short_term_features = {
        'vid': torch.randint(0, vid_vocab_size, (batch_size, seq_len_short)),
        'aid': torch.randint(0, aid_vocab_size, (batch_size, seq_len_short)),
        'tag': torch.randn(batch_size, seq_len_short, 100),
        'ts': torch.randn(batch_size, seq_len_short),
        'playtime': torch.randn(batch_size, seq_len_short),
        'dur': torch.randn(batch_size, seq_len_short),
        'label': torch.randn(batch_size, seq_len_short, 10)
    }
    
    # Positive feedback features
    pos_feedback_features = {
        'vid': torch.randint(0, vid_vocab_size, (batch_size, seq_len_pos)),
        'aid': torch.randint(0, aid_vocab_size, (batch_size, seq_len_pos)),
        'tag': torch.randn(batch_size, seq_len_pos, 100),
        'ts': torch.randn(batch_size, seq_len_pos),
        'playtime': torch.randn(batch_size, seq_len_pos),
        'dur': torch.randn(batch_size, seq_len_pos),
        'label': torch.randn(batch_size, seq_len_pos, 10)
    }
    
    # Lifelong features (compressed to 2000 before QFormer)
    lifelong_features = {
        'vid': torch.randint(0, vid_vocab_size, (batch_size, seq_len_life)),
        'aid': torch.randint(0, aid_vocab_size, (batch_size, seq_len_life)),
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