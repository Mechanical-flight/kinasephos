import csv

import yaml

from kinasephos.training.trainer import train_from_config


def test_one_epoch_smoke(tmp_path):
    data = tmp_path / "demo.csv"
    fields = ["sample_id", "window_61", "binary_label", "family_label"]
    with data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "sample_id": str(index),
                    "window_61": "A" * 30 + "STY"[index % 3] + "A" * 30,
                    "binary_label": int(index > 0),
                    "family_label": index if index > 0 else -1,
                }
            )
    config = {
        "seed": 1,
        "synthetic_smoke": True,
        "data_path": str(data),
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "batch_size": 2,
        "epochs": 1,
        "model": {
            "hidden_dim": 32,
            "heads": 4,
            "ffn_dim": 64,
            "layers": 1,
            "projection_dim": 1280,
            "dropout": 0.0,
        },
        "loss": {"alpha": 1.0, "beta": 1.0, "lambda_distill": 1.0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = train_from_config(str(config_path))
    assert summary["status"] == "success"
    assert summary["synthetic_smoke_only"] is True
    assert (tmp_path / "out/best.pt").exists()


def test_formal_training_requires_teacher_cache(tmp_path):
    data = tmp_path / "formal.csv"
    data.write_text(
        f"sample_id,window_61,binary_label,family_label\ns1,{'A' * 30}S{'A' * 30},1,0\n",
        encoding="utf-8",
    )
    config = {
        "data_path": str(data),
        "output_dir": str(tmp_path / "out"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        train_from_config(str(config_path))
    except ValueError as exc:
        assert "teacher_cache" in str(exc)
    else:
        raise AssertionError("Formal training without a teacher cache must fail")
