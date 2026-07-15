# Changelog

## 0.1.0 - 2026-07-14

- Initial clean-room implementation.
- Added asset auditing, leakage-aware splitting, Student Transformer, frozen-teacher pooling,
  masked multi-task/distillation loss, training, evaluation, inference, tests, and CI.
- Reconciled the exact architecture PDF and presentation, documenting conflicts and provenance.
- Separated true sequence padding from unknown residues and excluded padding from Student attention.
- Made formal training require hash-validated ESM teacher caches with resumable sharding.
- Added family per-class precision/recall/F1/support and end-to-end CI verification.
