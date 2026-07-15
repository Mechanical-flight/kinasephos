import csv
from collections.abc import Iterator
from pathlib import Path

from kinasephos.data.schema import SiteRecord


def parse_fasta(path: str | Path) -> Iterator[tuple[str, str, str]]:
    header: str | None = None
    sequence: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace", newline=None) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    token = header.split()[0]
                    parts = token.split("|")
                    accession = parts[1] if len(parts) >= 2 else token
                    yield accession, "".join(sequence).upper().rstrip("*"), header
                header = line[1:]
                sequence = []
            elif header is None:
                raise ValueError("FASTA sequence encountered before a header")
            else:
                sequence.append(line)
    if header is not None:
        token = header.split()[0]
        parts = token.split("|")
        accession = parts[1] if len(parts) >= 2 else token
        yield accession, "".join(sequence).upper().rstrip("*"), header


def parse_site_table(
    path: str | Path,
    *,
    source_dataset: str,
    accession_column: str,
    position_column: str,
    delimiter: str = "\t",
) -> Iterator[SiteRecord]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=delimiter):
            position_raw = row[position_column].strip().upper()
            residue = position_raw[0]
            position = int(position_raw[1:])
            yield SiteRecord(
                source_dataset=source_dataset,
                source_file=str(path),
                protein_accession_raw=row[accession_column].strip(),
                residue=residue,
                position_1based=position,
                evidence_type=row.get("Source"),
                reference_id=row.get("EPSD ID"),
            )
