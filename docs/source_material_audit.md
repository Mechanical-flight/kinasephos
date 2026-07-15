# Architecture source-material audit

Audit date: 2026-07-14. The two supplied files remain outside Git; only their fingerprints and the
resulting implementation decisions are recorded here.

| Source | SHA-256 | Role |
|---|---|---|
| `KinasePhos_架构确切数字版_v2(2).pdf` | `8130b60ac6879a74dca9265d9271ea4877f7c472766dcf5d1b854934f3eff9b5` | Exact architecture contract; highest precedence |
| `KinasePhos_AI蛋白设计算法岗项目展示(2).pptx` | `e42f616fcf656f443aa77ee729a16c099d57b7b4eba77521b504dbb982c05f65` | Project narrative, historical claims, and roadmap |

## Implemented contract

| Component | Exact value |
|---|---|
| Input | 61 residues; phosphosite at 1-indexed position 31 |
| Teacher | `ESM2-t33-650M-UR50D`; 33 layers; hidden 1280; 20 heads; frozen |
| Teacher representation | residue-only mean pool, `B x 1280` |
| Student | 4 Transformer layers; hidden 256; 8 heads; FFN 1024 |
| Student representation | mask-aware pool `B x 256`, then `Linear(256, 1280)` |
| Binary head | `1280 -> 256 -> 1`, ReLU, dropout 0.2 |
| Family head | `1280 -> 256 -> 4`, ReLU, dropout 0.2 |
| Families | CMGC, AGC, TK, CAMK |
| Loss | BCE-with-logits + cross-entropy + pooled-representation MSE; weights 1/1/1 |
| Optimization | AdamW, learning rate `1e-3`, weight decay `1e-2`, batch 128, 20 epochs |

The PDF depicts sigmoid/softmax at the task outputs. The implementation correctly supplies raw logits
to numerically stable training losses and applies sigmoid/softmax only for metrics and inference.

## Conflict resolution

- Presentation slide 6 describes a legacy division in which a large model performs site detection
  and a smaller model performs kinase classification. It is superseded by the exact specification:
  the Student owns both deployable heads and learns from the frozen ESM representation.
- The presentation's references to intermediate-layer or attention distillation are treated as
  experimental framing. The exact contract defines MSE on the pooled 1280-dimensional representation.
- Historical figures (including site accuracy above 89% and kinase Top-1 above 65%) are unverified
  claims. They are not published as reproduced Results without a data fingerprint, split manifest,
  checkpoint, and evaluation artifact.
- ProteinMPNN, RFdiffusion, and AlphaFold are roadmap context, not repository capabilities.

These decisions are protected by shape, pooling, padding, cache-integrity, smoke-training, inference,
and export tests in CI.
