import csv
from pathlib import Path

from kinasephos.data.parsers import parse_fasta, parse_site_table
from kinasephos.data.windowing import build_window
from kinasephos.utils.io import ensure_dir


def prepare_sites(
    fasta_path: str,
    site_path: str,
    output_path: str,
    source_dataset: str = "user_supplied_requires_verification",
) -> dict[str, int]:
    sequences = {accession: sequence for accession, sequence, _ in parse_fasta(fasta_path)}
    counts = {"valid": 0, "unmapped": 0, "invalid": 0}
    rows: list[dict[str, object]] = []
    for site in parse_site_table(
        site_path,
        source_dataset=source_dataset,
        accession_column="UniProt",
        position_column="Position",
    ):
        sequence = sequences.get(site.protein_accession_raw)
        if sequence is None:
            counts["unmapped"] += 1
            continue
        result = build_window(sequence, site.position_1based, site.residue)
        if result.window is None:
            counts["invalid"] += 1
            continue
        counts["valid"] += 1
        rows.append(
            {
                "sample_id": site.site_key,
                "protein_id": site.protein_accession_raw,
                "protein_accession": site.protein_accession_raw,
                "position_1based": site.position_1based,
                "center_residue": site.residue,
                "window_61": result.window,
                "binary_label": 1,
                "family_label": -1,
                "family_label_status": "mapping_not_audited",
                "source_list": source_dataset,
                "sequence_validation_status": result.status,
            }
        )
    destination = Path(output_path)
    ensure_dir(destination.parent)
    if rows:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return counts
