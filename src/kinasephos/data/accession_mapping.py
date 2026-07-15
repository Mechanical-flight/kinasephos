from dataclasses import replace

from kinasephos.data.schema import SiteRecord


def canonicalize_uniprot(accession: str, explicit_mapping: dict[str, str] | None = None) -> str:
    value = accession.strip()
    if explicit_mapping and value in explicit_mapping:
        return explicit_mapping[value]
    return value


def map_site(record: SiteRecord, mapping: dict[str, str] | None = None) -> SiteRecord:
    return replace(
        record,
        protein_accession_canonical=canonicalize_uniprot(record.protein_accession_raw, mapping),
    )
