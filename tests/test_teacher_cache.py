import json

import pytest
import torch

from kinasephos.data.teacher_cache import load_teacher_cache, window_hash


def write_cache(root, row):
    root.mkdir()
    torch.save(
        {
            "sample_ids": [row["sample_id"]],
            "window_hashes": [window_hash(row["window_61"])],
            "embeddings": torch.ones(1, 1280),
        },
        root / "shard_00000.pt",
    )
    (root / "manifest.json").write_text(json.dumps({"hidden_dim": 1280}), encoding="utf-8")
    (root / "index.json").write_text(
        json.dumps(
            {
                "samples": {
                    row["sample_id"]: {
                        "shard": "shard_00000.pt",
                        "offset": 0,
                        "window_hash": window_hash(row["window_61"]),
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_teacher_cache_loads_and_checks_window_identity(tmp_path):
    row = {
        "sample_id": "s1",
        "window_61": "A" * 30 + "S" + "A" * 30,
    }
    cache = tmp_path / "cache"
    write_cache(cache, row)
    loaded = load_teacher_cache(cache, [row])
    assert loaded["s1"].shape == (1280,)

    stale = {**row, "window_61": "A" * 30 + "T" + "A" * 30}
    with pytest.raises(ValueError, match="Stale teacher cache"):
        load_teacher_cache(cache, [stale])
