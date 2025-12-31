"""
Simple test to verify decoder implementation
"""
import torch
from onerec.decoder import OneRecDecoder


def test_decoder():
    print("Testing OneRec Decoder implementation...")
    
    # Initialize decoder
    decoder = OneRecDecoder()
    
    # Create dummy encoder output
    batch_size = 2
    enc_seq_len = 100  # Smaller for faster testing
    encoder_output = torch.randn(batch_size, enc_seq_len, 512)
    
    # Create dummy semantic IDs (for training)
    semantic_ids = torch.randint(0, 256, (batch_size, 3))  # 3 RQ layers
    
    print(f"Input semantic IDs: {semantic_ids}")
    print(f"Input semantic IDs shape: {semantic_ids.shape}")
    
    # Forward pass
    logits = decoder(semantic_ids, encoder_output)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {3}, {256}]")
    
    # Check if shapes are correct
    assert logits.shape == (batch_size, 3, 256), f"Expected (2, 3, 256), got {logits.shape}"
    
    # Generate semantic IDs
    generated_ids = decoder.generate(encoder_output, max_length=3)
    print(f"Generated IDs shape: {generated_ids.shape}")
    print(f"Generated IDs: {generated_ids}")
    
    assert generated_ids.shape == (batch_size, 3), f"Expected (2, 3), got {generated_ids.shape}"
    
    print("All tests passed! Decoder implementation is correct.")


if __name__ == "__main__":
    test_decoder()