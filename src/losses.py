import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1 - targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            los_pos = los_pos * ((1 - pt_pos) ** self.gamma_pos)
            los_neg = los_neg * (pt_neg ** self.gamma_neg)

        loss = -(los_pos + los_neg)
        return loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
