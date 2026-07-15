import random

from kinasephos.constants import PHOSPHO_RESIDUES


def sample_unannotated_sites(
    sequences: dict[str, str],
    positives: set[tuple[str, int]],
    count: int,
    seed: int = 42,
) -> list[tuple[str, int, str]]:
    candidates = [
        (protein, index + 1, residue)
        for protein, sequence in sequences.items()
        for index, residue in enumerate(sequence)
        if residue in PHOSPHO_RESIDUES and (protein, index + 1) not in positives
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]
