# Architecture

The exact implementation contract is encoded in model constructors and tests. ESM tokenization may
add BOS, EOS, and PAD; `residue_only_mean_pool` explicitly removes all special tokens. The teacher is
frozen with `requires_grad_(False)`, remains in evaluation mode, and runs under inference mode.

The Student produces `B x 61 x 256`, mask-pools to `B x 256`, and projects to `B x 1280`. Both heads
consume that projected representation. An all-missing family batch yields a differentiable zero
family loss instead of NaN.

Sequence-edge padding uses the reserved `-` character and maps to vocabulary ID 0. A biological `X`
maps to the unknown-residue ID instead, so the Student attention mask excludes only true padding.
The teacher strips boundary padding before ESM tokenization and rejects internal padding.

The exact two-page architecture specification takes precedence over the presentation narrative. In
particular, both deployed heads consume the projected Student representation and the distillation
target is the teacher's residue-only pooled 1280-vector. See
[source_material_audit.md](source_material_audit.md).
