import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kinasephos.data.collate import collate_sites
from kinasephos.data.dataset import SiteDataset
from kinasephos.models.kinasephos import KinasePhosModel
from kinasephos.models.losses import MultiTaskDistillationLoss
from kinasephos.training.checkpointing import save_checkpoint
from kinasephos.training.logging_utils import environment_info
from kinasephos.utils.config import load_config
from kinasephos.utils.io import ensure_dir, write_json
from kinasephos.utils.seed import seed_everything


def train_from_config(config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    requested = config.get("device", "auto")
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    data_path = Path(config["data_path"])
    dataset = SiteDataset.from_csv(data_path, synthetic_teacher=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        collate_fn=collate_sites,
        generator=torch.Generator().manual_seed(seed),
    )
    model = KinasePhosModel(**config["model"]).to(device)
    loss_config = config.get("loss", {})
    criterion = MultiTaskDistillationLoss(**loss_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
    )
    output_dir = ensure_dir(config["output_dir"])
    history: list[dict[str, object]] = []
    start = time.perf_counter()
    for epoch in range(int(config.get("epochs", 20))):
        model.train()
        totals = {"total": 0.0, "binary": 0.0, "family": 0.0, "distill": 0.0}
        batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            tokens = batch["tokens"].to(device)
            output = model(tokens)
            losses = criterion(
                output["binary_logits"],
                output["family_logits"],
                output["projected"],
                batch["teacher_embedding"].to(device),
                batch["binary_label"].to(device),
                batch["family_label"].to(device),
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("gradient_clip", 1.0))
            )
            optimizer.step()
            for name in totals:
                totals[name] += float(getattr(losses, name).detach())
            batches += 1
        row = {"epoch": epoch + 1, **{name: value / batches for name, value in totals.items()}}
        history.append(row)
    config["synthetic_smoke"] = "demo" in data_path.name
    save_checkpoint(
        output_dir / "last.pt", model, optimizer, len(history), config, -history[-1]["total"]
    )
    save_checkpoint(
        output_dir / "best.pt", model, optimizer, len(history), config, -history[-1]["total"]
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "status": "success",
        "synthetic_smoke_only": config["synthetic_smoke"],
        "epochs": len(history),
        "samples": len(dataset),
        "duration_seconds": round(time.perf_counter() - start, 3),
        "environment": environment_info(),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
