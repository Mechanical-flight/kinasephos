# Architecture

The exact implementation contract is encoded in model constructors and tests. ESM tokenization may
add BOS, EOS, and PAD; `residue_only_mean_pool` explicitly removes all special tokens. The teacher is
frozen with `requires_grad_(False)`, remains in evaluation mode, and runs under inference mode.

The Student produces `B x 61 x 256`, mask-pools to `B x 256`, and projects to `B x 1280`. Both heads
consume that projected representation. An all-missing family batch yields a differentiable zero
family loss instead of NaN.

