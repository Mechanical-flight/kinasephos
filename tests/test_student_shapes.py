import torch

from kinasephos.models.kinasephos import KinasePhosModel


def test_student_and_head_shapes():
    model = KinasePhosModel(
        hidden_dim=32, heads=4, ffn_dim=64, layers=1, projection_dim=1280, dropout=0.0
    )
    output = model(torch.ones((2, 61), dtype=torch.long))
    assert output["token_repr"].shape == (2, 61, 32)
    assert output["projected"].shape == (2, 1280)
    assert output["binary_logits"].shape == (2,)
    assert output["family_logits"].shape == (2, 4)
