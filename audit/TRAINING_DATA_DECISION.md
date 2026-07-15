# Training-data decision

## Decision

Formal biological training is **not approved** from the audited assets. Synthetic smoke training is
approved for engineering validation only.

## Evidence

- The FASTA contains 75,248 records with `EPSD-like-ID|UniProt|gene` headers, no embedded species or
  release declaration, 12,466 duplicate sequences, and 37 terminal stop markers.
- The archive contains 12,740 relationship rows, 9,391 unique accession-position pairs, 3,397 unique
  substrate accessions, and 389 unique kinase strings. Its internal EPSD schema conflicts with the
  archive filename, so it is not called an official PhosphoSitePlus export.
- Against the supplied FASTA, 10,000 relationship rows produced residue-validated 61-mers, 2,630
  accessions were unmapped, and 110 rows failed coordinate/residue validation. These counts measure
  internal compatibility only, not provenance or human specificity.
- The workbook has 3,487 data rows and seven metadata columns, spans multiple organisms, and provides
  no phosphosite position or kinase-group mapping.

## Task decisions

- Reference proteome: `requires verification`; supplied FASTA is not promoted to the canonical human
  reference.
- Binary positives: `requires provenance verification`; internally compatible windows remain private.
- Kinase-substrate relations: structurally present, but source/license/species status is unresolved.
- Four-family labels: unavailable from audited data; no fuzzy mapping is fabricated.
- Candidate negatives: not generated until a canonical human reference and final positive union exist.
- Formal CD-HIT split, teacher cache, 20-epoch training, and biological metrics: blocked by the above.

