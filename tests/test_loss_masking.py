import torch

from kinasephos.models.losses import MultiTaskDistillationLoss


def test_family_loss_masks_negative_missing_and_ambiguous():
    loss = MultiTaskDistillationLoss()
    output = loss(
        torch.zeros(4),
        torch.zeros(4, 4, requires_grad=True),
        torch.zeros(4, 8, requires_grad=True),
        torch.ones(4, 8),
        torch.tensor([1.0, 0.0, 1.0, 1.0]),
        torch.tensor([2, 1, -1, 4]),
    )
    assert output.valid_family_count == 1
    output.total.backward()


def test_empty_family_batch_is_differentiable_zero():
    logits = torch.randn(2, 4, requires_grad=True)
    loss = MultiTaskDistillationLoss()(
        torch.zeros(2, requires_grad=True),
        logits,
        torch.zeros(2, 8, requires_grad=True),
        torch.ones(2, 8),
        torch.zeros(2),
        torch.tensor([-1, -1]),
    )
    assert loss.family.item() == 0.0
    loss.total.backward()
    assert logits.grad is not None
