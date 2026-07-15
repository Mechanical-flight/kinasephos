from pathlib import Path

import torch

from kinasephos.constants import FAMILY_TO_ID, VOCAB
from kinasephos.utils.io import ensure_dir


def save_checkpoint(
    path: str | Path, model, optimizer, epoch: int, config: dict, metric: float
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
            "config": config,
            "vocabulary": VOCAB,
            "family_mapping": FAMILY_TO_ID,
            "threshold": 0.5,
            "data_fingerprint": "synthetic-smoke" if config.get("synthetic_smoke") else "TBD",
        },
        target,
    )


def load_model_checkpoint(path: str | Path, device: str = "cpu"):
    from kinasephos.models.kinasephos import KinasePhosModel

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = KinasePhosModel(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
