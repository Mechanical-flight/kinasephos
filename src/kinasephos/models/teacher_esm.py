import torch
from torch import nn

from kinasephos.constants import SEQUENCE_PAD_CHAR


def strip_sequence_padding(sequence: str) -> str:
    stripped = sequence.strip(SEQUENCE_PAD_CHAR)
    if SEQUENCE_PAD_CHAR in stripped:
        raise ValueError("Sequence padding is only allowed at window boundaries")
    if not stripped:
        raise ValueError("A teacher sequence cannot contain padding only")
    return stripped


def residue_only_mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("Hidden states must be [batch, tokens, hidden]")
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("Each sequence needs at least one real residue token")
    expanded = mask.unsqueeze(-1)
    return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1)


class FrozenESMTeacher(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        revision: str = "main",
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install kinasephos[teacher] to use the ESM teacher") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, sequences: list[str]) -> torch.Tensor:
        sequences = [strip_sequence_padding(sequence) for sequence in sequences]
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        special = encoded.pop("special_tokens_mask")
        hidden = self.model(**encoded).last_hidden_state
        return residue_only_mean_pool(hidden, encoded["attention_mask"], special)
