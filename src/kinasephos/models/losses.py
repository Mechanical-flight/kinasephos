from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossOutput:
    total: torch.Tensor
    binary: torch.Tensor
    family: torch.Tensor
    distill: torch.Tensor
    valid_family_count: int


class MultiTaskDistillationLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, lambda_distill: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_distill = lambda_distill
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.family_loss = nn.CrossEntropyLoss()
        self.distill_loss = nn.MSELoss()

    def forward(
        self,
        binary_logits: torch.Tensor,
        family_logits: torch.Tensor,
        student_projected: torch.Tensor,
        teacher_projected: torch.Tensor,
        binary_targets: torch.Tensor,
        family_targets: torch.Tensor,
    ) -> LossOutput:
        binary = self.binary_loss(binary_logits, binary_targets.float())
        valid = binary_targets.bool() & family_targets.ge(0) & family_targets.lt(4)
        if valid.any():
            family = self.family_loss(family_logits[valid], family_targets[valid])
        else:
            family = family_logits.sum() * 0.0
        distill = self.distill_loss(student_projected, teacher_projected.detach())
        total = self.alpha * binary + self.beta * family + self.lambda_distill * distill
        return LossOutput(total, binary, family, distill, int(valid.sum().item()))
