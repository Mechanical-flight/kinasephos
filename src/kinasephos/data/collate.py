import torch


def collate_sites(batch: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "sample_id": [item["sample_id"] for item in batch],
        "window": [item["window"] for item in batch],
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "binary_label": torch.stack([item["binary_label"] for item in batch]),
        "family_label": torch.stack([item["family_label"] for item in batch]),
    }
    if "teacher_embedding" in batch[0]:
        result["teacher_embedding"] = torch.stack([item["teacher_embedding"] for item in batch])
    return result
