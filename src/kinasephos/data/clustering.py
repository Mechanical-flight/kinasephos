import subprocess
from pathlib import Path


def parse_cdhit_clusters(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    cluster = ""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">Cluster "):
                cluster = line.strip().replace(">Cluster ", "cluster_")
            elif ">" in line and "..." in line:
                accession = line.split(">", 1)[1].split("...", 1)[0]
                mapping[accession] = cluster
    return mapping


def run_cdhit(input_fasta: str, output_fasta: str, identity: float = 0.4) -> None:
    command = ["cd-hit", "-i", input_fasta, "-o", output_fasta, "-c", str(identity), "-n", "2"]
    subprocess.run(command, check=True, text=True, capture_output=True)
