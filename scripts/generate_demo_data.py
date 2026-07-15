#!/usr/bin/env python
import argparse
import csv
import random
from pathlib import Path


def generate(output: str, samples: int = 24, seed: int = 42) -> None:
    rng = random.Random(seed)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    families = ["CMGC", "AGC", "TK", "CAMK"]
    rows = []
    for index in range(samples):
        center = "STY"[index % 3]
        window = "".join(rng.choice(amino_acids) for _ in range(30))
        window += center
        window += "".join(rng.choice(amino_acids) for _ in range(30))
        positive = int(index % 3 != 0)
        family = index % 4 if positive else -1
        rows.append(
            {
                "sample_id": f"synthetic_{index:04d}",
                "protein_id": f"SYNTHETIC_PROTEIN_{index // 2:03d}",
                "protein_accession": f"SYN{index // 2:05d}",
                "gene_name": "SYNTHETIC",
                "position_1based": 31,
                "center_residue": center,
                "window_61": window,
                "binary_label": positive,
                "family_label": family,
                "family_label_status": families[family] if positive else "not_applicable",
                "kinase_list": "synthetic",
                "kinase_family_list": families[family] if positive else "",
                "source_list": "synthetic_ci_fixture",
                "source_count": 1,
                "evidence_list": "none_synthetic",
                "cluster_id": f"synthetic_cluster_{index // 2:03d}",
                "split": "train",
                "sequence_validation_status": "synthetic_valid",
            }
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    generate(args.output, args.samples)
