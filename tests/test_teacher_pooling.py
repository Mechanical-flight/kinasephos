import torch

from kinasephos.models.teacher_esm import residue_only_mean_pool, strip_sequence_padding


def test_teacher_pool_excludes_bos_eos_and_pad():
    hidden = torch.tensor([[[100.0], [1.0], [3.0], [200.0], [300.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    pooled = residue_only_mean_pool(hidden, attention, special)
    assert pooled.item() == 2.0


def test_teacher_strips_only_boundary_padding():
    assert strip_sequence_padding("---MST---") == "MST"
    try:
        strip_sequence_padding("MS-T")
    except ValueError as exc:
        assert "boundaries" in str(exc)
    else:
        raise AssertionError("Internal padding must be rejected")
