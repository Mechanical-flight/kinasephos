from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SiteRecord:
    source_dataset: str
    source_file: str
    protein_accession_raw: str
    residue: str
    position_1based: int
    protein_accession_canonical: str | None = None
    gene_name: str | None = None
    evidence_type: str | None = None
    reference_id: str | None = None

    @property
    def site_key(self) -> str:
        accession = self.protein_accession_canonical or self.protein_accession_raw
        return f"{accession}:{self.residue}{self.position_1based}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleRecord:
    sample_id: str
    protein_id: str
    protein_accession: str
    gene_name: str
    position_1based: int
    center_residue: str
    window_61: str
    binary_label: int
    family_label: int = -1
    family_label_status: str = "missing"
    kinase_list: list[str] = field(default_factory=list)
    kinase_family_list: list[str] = field(default_factory=list)
    source_list: list[str] = field(default_factory=list)
    source_count: int = 0
    evidence_list: list[str] = field(default_factory=list)
    cluster_id: str = ""
    split: str = ""
    sequence_validation_status: str = "valid"

    def to_dict(self) -> dict:
        return asdict(self)
