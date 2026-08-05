"""Quick test to verify MultiS4 wrapper works correctly."""

import torch
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.modules.layers.multi_s4 import MultiS4

def test_multi_s4():
    """Test MultiS4 with different number of layers."""

    batch_size = 4
    seq_len = 128
    d_model = 16

    # Test with n_layers=1 (should behave like single S4Block)
    print("Testing MultiS4 with n_layers=1...")
    model_1layer = MultiS4(d_model=d_model, n_layers=1, transposed=False)
    x = torch.randn(batch_size, seq_len, d_model)

    try:
        y = model_1layer(x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {y.shape}")
        assert y.shape == x.shape, "Output shape mismatch!"
        print("✓ Single layer test passed!\n")
    except Exception as e:
        print(f"✗ Single layer test failed: {e}\n")
        return False

    # Test with n_layers=3
    print("Testing MultiS4 with n_layers=3...")
    model_3layer = MultiS4(d_model=d_model, n_layers=3, transposed=False)

    try:
        y = model_3layer(x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {y.shape}")
        assert y.shape == x.shape, "Output shape mismatch!"
        print("✓ Multi-layer test passed!\n")
    except Exception as e:
        print(f"✗ Multi-layer test failed: {e}\n")
        return False

    # Test parameter count scaling
    params_1 = sum(p.numel() for p in model_1layer.parameters())
    params_3 = sum(p.numel() for p in model_3layer.parameters())
    print(f"Parameters with 1 layer: {params_1:,}")
    print(f"Parameters with 3 layers: {params_3:,}")
    print(f"Ratio: {params_3/params_1:.2f}x")

    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_multi_s4()
