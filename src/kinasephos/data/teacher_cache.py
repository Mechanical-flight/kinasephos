import json
from collections import defaultdict
from pathlib import Path

import torch

from kinasephos.utils.hashing import stable_hash


def window_hash(window: str) -> str:
    return stable_hash({"window_61": window})


def load_teacher_cache(
    cache_dir: str | Path,
    rows: list[dict[str, str]],
    expected_hidden_dim: int = 1280,
) -> dict[str, torch.Tensor]:
    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    index_path = root / "index.json"
    if not manifest_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Teacher cache is incomplete at {root}; run cache_teacher_embeddings.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("hidden_dim", -1)) != expected_hidden_dim:
        raise ValueError("Teacher cache hidden dimension does not match the model contract")
    index = json.loads(index_path.read_text(encoding="utf-8"))["samples"]
    requested: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        sample_id = row["sample_id"]
        entry = index.get(sample_id)
        if entry is None:
            raise KeyError(f"Teacher cache does not contain sample {sample_id}")
        if entry["window_hash"] != window_hash(row["window_61"]):
            raise ValueError(f"Stale teacher cache for sample {sample_id}: window hash changed")
        requested[entry["shard"]].append((sample_id, int(entry["offset"])))

    embeddings: dict[str, torch.Tensor] = {}
    for shard_name, members in requested.items():
        shard_path = root / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing teacher cache shard: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        tensor = shard["embeddings"]
        if tensor.ndim != 2 or tensor.shape[1] != expected_hidden_dim:
            raise ValueError(f"Invalid embedding shape in {shard_path}: {tuple(tensor.shape)}")
        for sample_id, offset in members:
            if offset < 0 or offset >= len(tensor):
                raise ValueError(f"Invalid offset for {sample_id} in {shard_path}")
            if shard["sample_ids"][offset] != sample_id:
                raise ValueError(f"Teacher cache index mismatch for {sample_id}")
            embeddings[sample_id] = tensor[offset].float()
    return embeddings
