# Training

`configs/train_full.yaml` records the formal defaults: AdamW, learning rate `1e-3`, weight decay
`1e-2`, cached-teacher batch size 128, and 20 epochs. Online teacher use should reduce its micro-batch
and may use gradient accumulation. Teacher weights and embedding caches are never committed.

Checkpoints contain Student/head weights, optimizer, epoch, best metric, config, vocabulary, family
mapping, threshold, and a data fingerprint. The synthetic smoke path is explicitly labeled.

