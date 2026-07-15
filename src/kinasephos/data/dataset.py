import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kinasephos.constants import UNK_TOKEN, VOCAB
from kinasephos.utils.validation import validate_window


def encode_window(window: str) -> torch.Tensor:
    validate_window(window)
    unknown = VOCAB[UNK_TOKEN]
    return torch.tensor([VOCAB.get(residue, unknown) for residue in window], dtype=torch.long)


def synthetic_teacher_embedding(window: str, dimension: int = 1280) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(window.encode()).digest()[:8], "big") % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dimension, generator=generator) * 0.1


class SiteDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], synthetic_teacher: bool = False):
        self.rows = rows
        self.synthetic_teacher = synthetic_teacher

    @classmethod
    def from_csv(cls, path: str | Path, synthetic_teacher: bool = False) -> "SiteDataset":
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return cls(list(csv.DictReader(handle)), synthetic_teacher=synthetic_teacher)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        window = row["window_61"]
        item: dict[str, object] = {
            "sample_id": row["sample_id"],
            "window": window,
            "tokens": encode_window(window),
            "binary_label": torch.tensor(float(row["binary_label"]), dtype=torch.float32),
            "family_label": torch.tensor(int(row.get("family_label", -1)), dtype=torch.long),
        }
        if self.synthetic_teacher:
            item["teacher_embedding"] = synthetic_teacher_embedding(window)
        return item
