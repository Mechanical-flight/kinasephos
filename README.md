# KinasePhos 4.0

[![CI](https://github.com/Mechanical-flight/kinasephos/actions/workflows/ci.yml/badge.svg)](https://github.com/Mechanical-flight/kinasephos/actions/workflows/ci.yml)

KinasePhos 4.0 is a clean-room PyTorch implementation for predicting phosphorylation sites and four
kinase groups from a 61-residue local sequence. It combines a frozen ESM-2 representation teacher
with a deployable Student Transformer. This repository started with no source code, checkpoint, or
reproducible training log; historical presentation figures are therefore **not yet reproduced**.

> Status: the software and synthetic CPU smoke path are implemented. The supplied historical assets
> were audited conservatively and are not sufficient for a provenance-verified formal training run.
> No raw database export, teacher weight, embedding cache, or checkpoint is committed.

## Architecture

```text
61-mer
  |-- frozen facebook/esm2_t33_650M_UR50D
  |      `-- residue-only mean pool (B x 1280)
  `-- Student Transformer (4 layers, 256 hidden, 8 heads, 1024 FFN)
         `-- mask-aware pool -> Linear(256, 1280)
                |-- binary head: 1280 -> 256 -> 1
                |-- family head: 1280 -> 256 -> 4
                `-- MSE distillation to teacher representation
```

The family classes are `CMGC`, `AGC`, `TK`, and `CAMK`. Family cross-entropy is computed only for
positive, unambiguous samples with a valid target family. The model emits raw logits; sigmoid and
softmax are applied only for metrics and inference.

## Install and quick start

```bash
python -m pip install -e ".[dev]"
python scripts/generate_demo_data.py --output data/processed/demo.csv
python scripts/train.py --config configs/train_smoke.yaml
python scripts/evaluate.py --config configs/evaluate.yaml
python scripts/predict.py \
  --input examples/demo_input.csv \
  --output outputs/demo_predictions.csv
```

These commands use synthetic fixtures and do not establish biological performance.

## Data workflow

```bash
kinasephos audit-assets --input /path/to/assets --output audit/private
kinasephos prepare-data --config configs/data.yaml
python scripts/run_cdhit.py --config configs/data.yaml
python scripts/cache_teacher_embeddings.py --config configs/train_full.yaml
kinasephos train --config configs/train_full.yaml
kinasephos evaluate --config configs/evaluate.yaml
```

Positive sites are accession- and coordinate-validated against one reference proteome. Candidate
negatives are explicitly described as **unannotated candidate negatives**, never absolute biological
negatives. Splits are grouped by CD-HIT cluster (or protein as a minimum fallback), and the pipeline
checks protein, cluster, and site leakage.

Obtain UniProt and any phosphorylation database data from their official providers under current
terms. Do not place restricted exports in this repository. See [DATA_CARD.md](DATA_CARD.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [docs/data_asset_audit.md](docs/data_asset_audit.md).

## Teacher cache and deployment

Full training expects offline ESM-2 embeddings. Cache manifests record model name, representation
dimension, revision, pooling rule, input hash, sample/window identity, and configuration hash. Formal
training fails closed when the cache is missing or stale; synthetic teacher vectors require the
explicit smoke configuration. The teacher remains frozen and CI never downloads its 650M
parameters. `scripts/export_student.py` exports a Student-only state dict, vocabulary, configuration,
family mapping, and threshold.

## Evaluation

Binary reporting includes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, sensitivity,
specificity, and a confusion matrix. Valid family samples receive Top-1, Top-3, macro-F1,
weighted-F1, per-class analysis, and a confusion matrix. Error outputs are machine-readable.

No formal Results table is published yet. Historical figures such as 89% site accuracy or 65% kinase
Top-1 are presentation-only claims until a data fingerprint, leakage-free split, checkpoint, exact
configuration, and reproducible evaluation output exist.

## Limitations

- Missing annotation does not prove a negative site.
- Only four broad kinase groups are modeled.
- Multi-family sites are excluded from family softmax loss by default.
- A 61-mer cannot capture all long-range structure or cellular context.
- Database labels, accession mapping, and isoform coordinates can introduce bias or loss.
- Predictions are not experimental validation or medical advice.
- Protein generation tools are roadmap items, not implemented capabilities.

## Repository map

- `src/kinasephos/data`: audit, parsing, windows, negatives, clustering, splitting, datasets
- `src/kinasephos/models`: frozen teacher wrapper, Student, heads, loss
- `src/kinasephos/training`: trainer, metrics, checkpoints, evaluation
- `src/kinasephos/inference`: Student-only prediction and export
- `configs`, `scripts`, `tests`, `docs`, `examples`: reproducible interfaces and documentation

## Citation and license

Original code is MIT licensed. Data, model weights, and third-party tools retain their own terms. See
`CITATION.cff` and `THIRD_PARTY_NOTICES.md`.
