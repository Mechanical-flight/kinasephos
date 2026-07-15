from torch import nn

from kinasephos.models.heads import PredictionHead
from kinasephos.models.student_transformer import StudentTransformer


class KinasePhosModel(nn.Module):
    def __init__(self, **student_kwargs):
        super().__init__()
        projection_dim = int(student_kwargs.get("projection_dim", 1280))
        dropout = float(student_kwargs.get("dropout", 0.2))
        self.student = StudentTransformer(**student_kwargs)
        self.binary_head = PredictionHead(projection_dim, 1, dropout)
        self.family_head = PredictionHead(projection_dim, 4, dropout)

    def forward(self, tokens):
        token_repr, projected = self.student(tokens)
        return {
            "token_repr": token_repr,
            "projected": projected,
            "binary_logits": self.binary_head(projected).squeeze(-1),
            "family_logits": self.family_head(projected),
        }
