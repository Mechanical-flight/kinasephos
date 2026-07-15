#!/usr/bin/env python
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from kinasephos.data.teacher_cache import window_hash
from kinasephos.models.teacher_esm import FrozenESMTeacher
from kinasephos.utils.config import load_config
from kinasephos.utils.hashing import sha256_file, stable_hash


def read_rows(path: Path, split: str | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None and rows and "split" in rows[0]:
        rows = [row for row in rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"No samples found in {path} for split={split!r}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Teacher cache requires unique sample_id values")
    return rows


def valid_shard(
    path: Path,
    rows: list[dict[str, str]],
    hidden_dim: int,
) -> bool:
    if not path.exists():
        return False
    try:
        shard = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError):
        return False
    expected_ids = [row["sample_id"] for row in rows]
    expected_hashes = [window_hash(row["window_61"]) for row in rows]
    embeddings = shard.get("embeddings")
    return (
        shard.get("sample_ids") == expected_ids
        and shard.get("window_hashes") == expected_hashes
        and isinstance(embeddings, torch.Tensor)
        and tuple(embeddings.shape) == (len(rows), hidden_dim)
    )


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = Path(config["data_path"])
    split = config.get("train_split")
    rows = read_rows(data_path, split)

    teacher_config = config["teacher"]
    model_name = teacher_config["model_name"]
    revision = teacher_config.get("revision", "main")
    hidden_dim = int(teacher_config.get("hidden_dim", 1280))
    batch_size = int(teacher_config.get("batch_size", 8))
    shard_size = int(teacher_config.get("shard_size", 256))
    if batch_size < 1 or shard_size < 1:
        raise ValueError("Teacher batch_size and shard_size must be positive")

    cache = Path(config["teacher_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    identity = {
        "format_version": 1,
        "model": model_name,
        "revision": revision,
        "hidden_dim": hidden_dim,
        "pooling": teacher_config.get("pooling", "residue_only_mean"),
        "data_sha256": sha256_file(data_path),
        "split": split,
        "samples": len(rows),
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        differences = {
            key: (previous.get(key), value)
            for key, value in identity.items()
            if previous.get(key) != value
        }
        if differences:
            raise ValueError(
                "Existing teacher cache has a different identity; use a new cache directory: "
                f"{differences}"
            )

    groups: list[tuple[str, list[dict[str, str]]]] = []
    for start in range(0, len(rows), shard_size):
        groups.append((f"shard_{start // shard_size:05d}.pt", rows[start : start + shard_size]))
    pending = [group for group in groups if not valid_shard(cache / group[0], group[1], hidden_dim)]

    if pending:
        requested = config.get("teacher_device", "auto")
        device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if device == "auto":
            device = "cpu"
        teacher = FrozenESMTeacher(model_name, revision=revision).to(device)
        for shard_name, shard_rows in pending:
            batches = []
            for start in range(0, len(shard_rows), batch_size):
                batch = shard_rows[start : start + batch_size]
                batches.append(teacher([row["window_61"] for row in batch]).cpu())
            embeddings = torch.cat(batches, dim=0).float()
            if tuple(embeddings.shape) != (len(shard_rows), hidden_dim):
                raise ValueError(
                    f"Teacher returned {tuple(embeddings.shape)}, expected "
                    f"({len(shard_rows)}, {hidden_dim})"
                )
            atomic_torch_save(
                {
                    "sample_ids": [row["sample_id"] for row in shard_rows],
                    "site_keys": [row.get("site_key", "") for row in shard_rows],
                    "window_hashes": [window_hash(row["window_61"]) for row in shard_rows],
                    "embeddings": embeddings,
                },
                cache / shard_name,
            )

    index: dict[str, dict[str, object]] = {}
    for shard_name, shard_rows in groups:
        if not valid_shard(cache / shard_name, shard_rows, hidden_dim):
            raise RuntimeError(f"Teacher cache shard failed validation: {shard_name}")
        for offset, row in enumerate(shard_rows):
            index[row["sample_id"]] = {
                "shard": shard_name,
                "offset": offset,
                "site_key": row.get("site_key", ""),
                "window_hash": window_hash(row["window_61"]),
            }

    manifest = {
        **identity,
        "tokenizer": model_name,
        "config_hash": stable_hash(config),
        "seed": int(config.get("seed", 42)),
        "teacher_batch_size": batch_size,
        "shard_size": shard_size,
        "shards": len(groups),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache / "index.json").write_text(
        json.dumps({"format_version": 1, "samples": index}, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "success", "cache": str(cache), **manifest}, indent=2))


if __name__ == "__main__":
    main()
