from pathlib import Path

import torch

from kinasephos.training.checkpointing import load_model_checkpoint
from kinasephos.utils.io import ensure_dir, write_json


def export_student(checkpoint_path: str, output_dir: str) -> None:
    model, checkpoint = load_model_checkpoint(checkpoint_path)
    destination = ensure_dir(output_dir)
    torch.save(model.state_dict(), Path(destination) / "student_model.pt")
    write_json(Path(destination) / "config.json", checkpoint["config"])
    write_json(Path(destination) / "vocabulary.json", checkpoint["vocabulary"])
    write_json(Path(destination) / "family_mapping.json", checkpoint["family_mapping"])
