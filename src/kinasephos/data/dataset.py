import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from kinasephos.constants import PAD_TOKEN, SEQUENCE_PAD_CHAR, UNK_TOKEN, VOCAB
from kinasephos.utils.validation import validate_window


def encode_window(window: str) -> torch.Tensor:
    validate_window(window)
    unknown = VOCAB[UNK_TOKEN]
    padding = VOCAB[PAD_TOKEN]
    return torch.tensor(
        [
            padding if residue == SEQUENCE_PAD_CHAR else VOCAB.get(residue, unknown)
            for residue in window
        ],
        dtype=torch.long,
    )


def synthetic_teacher_embedding(window: str, dimension: int = 1280) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(window.encode()).digest()[:8], "big") % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dimension, generator=generator) * 0.1


class SiteDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        synthetic_teacher: bool = False,
        teacher_embeddings: dict[str, torch.Tensor] | None = None,
    ):
        if synthetic_teacher and teacher_embeddings is not None:
            raise ValueError("Choose synthetic or cached teacher embeddings, not both")
        self.rows = rows
        self.synthetic_teacher = synthetic_teacher
        self.teacher_embeddings = teacher_embeddings

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        synthetic_teacher: bool = False,
        teacher_embeddings: dict[str, torch.Tensor] | None = None,
        split: str | None = None,
    ) -> "SiteDataset":
        with Path(path).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if split is not None and rows and "split" in rows[0]:
            rows = [row for row in rows if row.get("split") == split]
        return cls(
            rows,
            synthetic_teacher=synthetic_teacher,
            teacher_embeddings=teacher_embeddings,
        )

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
        elif self.teacher_embeddings is not None:
            try:
                item["teacher_embedding"] = self.teacher_embeddings[row["sample_id"]]
            except KeyError as exc:
                raise KeyError(f"Missing cached teacher embedding for {row['sample_id']}") from exc
        return item
