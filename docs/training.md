# Training

`configs/train_full.yaml` records the formal defaults: AdamW, learning rate `1e-3`, weight decay
`1e-2`, Student batch size 128, and 20 epochs. Teacher caching defaults to micro-batches of 8 and
shards of 256 samples. Teacher weights and embedding caches are never committed.

Formal training refuses to start without a complete cached-teacher manifest, index, and validated
shards. Synthetic teacher vectors are available only when `synthetic_smoke: true` is explicitly set;
the filename is never used to infer that mode. Cache entries are joined by `sample_id` and protected
by per-window hashes, so edited or reordered data cannot silently reuse stale embeddings.

Checkpoints contain Student/head weights, optimizer, epoch, best metric, config, vocabulary, family
mapping, threshold, and a data fingerprint. The synthetic smoke path is explicitly labeled.
