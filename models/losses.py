"""Shared loss functions."""
import torch
import torch.nn.functional as F


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE - down-weights easy correct predictions to give rare positives
    more gradient.

        L = (1 - p_t)^gamma * BCE,   where p_t = p if y=1 else 1-p
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0.5, p, 1.0 - p)
    focal = (1.0 - p_t).clamp(min=1e-6) ** gamma
    return (focal * bce).mean()
