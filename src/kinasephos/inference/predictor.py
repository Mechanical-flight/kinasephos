import csv
from pathlib import Path

import torch

from kinasephos.constants import ID_TO_FAMILY
from kinasephos.data.dataset import encode_window
from kinasephos.training.checkpointing import load_model_checkpoint
from kinasephos.utils.io import ensure_dir


@torch.inference_mode()
def predict_csv(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> list[dict[str, object]]:
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tokens = torch.stack([encode_window(row["window_61"]) for row in rows]).to(device)
    output = model(tokens)
    binary_probabilities = torch.sigmoid(output["binary_logits"]).cpu()
    family_probabilities = torch.softmax(output["family_logits"], dim=-1).cpu()
    predictions: list[dict[str, object]] = []
    for row, binary_probability, family_probability in zip(
        rows, binary_probabilities, family_probabilities, strict=True
    ):
        top = torch.argsort(family_probability, descending=True)
        positive = float(binary_probability) >= threshold
        predictions.append(
            {
                **row,
                "phospho_logit": float(torch.logit(binary_probability.clamp(1e-7, 1 - 1e-7))),
                "phospho_probability": float(binary_probability),
                "binary_prediction": int(positive),
                "binary_threshold": threshold,
                "family_top1": ID_TO_FAMILY[int(top[0])] if positive else "unavailable",
                "family_top1_probability": float(family_probability[top[0]]),
                "family_top3": ";".join(ID_TO_FAMILY[int(index)] for index in top[:3]),
                **{
                    f"prob_{ID_TO_FAMILY[index]}": float(family_probability[index])
                    for index in range(4)
                },
                "model_version": checkpoint.get("config", {}).get("version", "0.1.0-smoke"),
            }
        )
    destination = Path(output_path)
    ensure_dir(destination.parent)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return predictions
