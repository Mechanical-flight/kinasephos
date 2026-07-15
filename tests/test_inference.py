import csv

import torch

from kinasephos.inference.predictor import predict_csv
from kinasephos.models.kinasephos import KinasePhosModel


def test_inference_schema(tmp_path):
    model_config = {
        "hidden_dim": 32,
        "heads": 4,
        "ffn_dim": 64,
        "layers": 1,
        "projection_dim": 1280,
        "dropout": 0.0,
    }
    model = KinasePhosModel(**model_config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"model": model_config},
            "vocabulary": {},
            "family_mapping": {},
        },
        checkpoint,
    )
    source = tmp_path / "input.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "window_61"])
        writer.writeheader()
        writer.writerow({"sample_id": "x", "window_61": "A" * 30 + "S" + "A" * 30})
    rows = predict_csv(str(source), str(tmp_path / "out.csv"), str(checkpoint))
    assert "phospho_probability" in rows[0]
    assert "prob_CMGC" in rows[0]
