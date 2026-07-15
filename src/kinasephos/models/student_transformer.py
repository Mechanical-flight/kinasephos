import torch
from torch import nn

from kinasephos.constants import VOCAB


class StudentTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        heads: int = 8,
        ffn_dim: int = 1024,
        layers: int = 4,
        projection_dim: int = 1280,
        max_length: int = 61,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(len(VOCAB), hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, projection_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != 61:
            raise ValueError(f"Expected tokens shaped [batch, 61], got {tuple(tokens.shape)}")
        padding_mask = tokens.eq(0)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        residue_mask = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1)
        return encoded, self.projection(pooled)
