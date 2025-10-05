import torch
import numpy as np
from src.losses import AsymmetricLoss


def test_asl_basic():
    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    
    logits = torch.randn(8, 19)
    targets = torch.randint(0, 2, (8, 19)).float()
    
    loss = criterion(logits, targets)
    
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    print("✓ ASL basic test passed")


def test_asl_gradients():
    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    
    logits = torch.randn(8, 19, requires_grad=True)
    targets = torch.randint(0, 2, (8, 19)).float()
    
    loss = criterion(logits, targets)
    loss.backward()
    
    assert logits.grad is not None, "Gradients should be computed"
    assert not torch.isnan(logits.grad).any(), "Gradients should not contain NaN"
    print("✓ ASL gradient test passed")


def test_asl_stability():
    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    
    logits_extreme = torch.tensor([[-100.0, 100.0], [-100.0, 100.0]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    
    loss = criterion(logits_extreme, targets)
    
    assert not torch.isnan(loss), "Loss should be stable with extreme values"
    assert not torch.isinf(loss), "Loss should not be Inf with extreme values"
    print("✓ ASL stability test passed")


if __name__ == '__main__':
    test_asl_basic()
    test_asl_gradients()
    test_asl_stability()
    print("\nAll ASL tests passed!")
