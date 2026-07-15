import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kinasephos.data.collate import collate_sites
from kinasephos.data.dataset import SiteDataset
from kinasephos.training.checkpointing import load_model_checkpoint
from kinasephos.training.metrics import binary_metrics, family_metrics
from kinasephos.utils.config import load_config
from kinasephos.utils.io import ensure_dir, write_json


@torch.inference_mode()
def evaluate_from_config(config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    model, _ = load_model_checkpoint(config["checkpoint"])
    dataset = SiteDataset.from_csv(config["data_path"])
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_sites)
    binary_targets: list[int] = []
    binary_probabilities: list[float] = []
    family_targets: list[int] = []
    family_probabilities: list[list[float]] = []
    errors: list[dict[str, object]] = []
    for batch in loader:
        output = model(batch["tokens"])
        binary_probability = torch.sigmoid(output["binary_logits"])
        family_probability = torch.softmax(output["family_logits"], dim=-1)
        binary_targets.extend(batch["binary_label"].int().tolist())
        binary_probabilities.extend(binary_probability.tolist())
        family_targets.extend(batch["family_label"].tolist())
        family_probabilities.extend(family_probability.tolist())
        for sample_id, target, probability in zip(
            batch["sample_id"], batch["binary_label"], binary_probability, strict=True
        ):
            prediction = int(float(probability) >= float(config.get("threshold", 0.5)))
            if prediction != int(target):
                errors.append(
                    {
                        "sample_id": sample_id,
                        "binary_target": int(target),
                        "probability": float(probability),
                        "error_type": "false_positive" if prediction else "false_negative",
                    }
                )
    results = {
        "scope": "synthetic smoke test only" if "demo" in config["data_path"] else "evaluation",
        "binary": binary_metrics(
            binary_targets, binary_probabilities, float(config.get("threshold", 0.5))
        ),
        "family": family_metrics(family_targets, family_probabilities),
    }
    output_dir = ensure_dir(config["output_dir"])
    write_json(Path(output_dir) / "metrics.json", results)
    if errors:
        with (Path(output_dir) / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(errors[0]))
            writer.writeheader()
            writer.writerows(errors)
    return results
