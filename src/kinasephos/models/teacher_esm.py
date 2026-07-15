import torch
from torch import nn


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
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install kinasephos[teacher] to use the ESM teacher") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, sequences: list[str]) -> torch.Tensor:
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
