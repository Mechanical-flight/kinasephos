#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import torch

from kinasephos.models.teacher_esm import FrozenESMTeacher
from kinasephos.utils.config import load_config
from kinasephos.utils.hashing import stable_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with Path(config["data_path"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    teacher = FrozenESMTeacher(config["teacher"]["model_name"])
    cache = Path(config["teacher_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), 8):
        shard = rows[start : start + 8]
        embeddings = teacher([row["window_61"] for row in shard]).cpu()
        torch.save(
            {"sample_ids": [row["sample_id"] for row in shard], "embeddings": embeddings},
            cache / f"shard_{start // 8:05d}.pt",
        )
    manifest = {
        "model": config["teacher"]["model_name"],
        "hidden_dim": 1280,
        "pooling": "residue_only_mean",
        "samples": len(rows),
        "config_hash": stable_hash(config),
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
