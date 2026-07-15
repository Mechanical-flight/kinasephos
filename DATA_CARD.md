# Data card

## Repository data

Only synthetic fixtures are committed. They test schemas, tensor shapes, loss masking, training, and
inference; they cannot support biological claims.

## Audited private assets

The supplied FASTA, compressed site table, and UniProt-named workbook were inspected outside Git.
Filename labels were not accepted as provenance. The FASTA uses EPSD-like identifiers, the compressed
site table uses an EPSD schema despite its archive name, and the workbook contains multi-organism
kinase entries. Formal training is blocked until provenance, species, reference version, licensing,
and kinase-family mapping are verified.

## Required real-data schema

Site records require source, canonical accession, gene, residue, 1-based position, evidence, reference,
mapping status, and sequence validation. Kinase relations additionally require kinase gene/accession
and an auditable group mapping. Ambiguous multi-family sites stay available for binary training but
are excluded from family loss.

## Splitting

CD-HIT clusters are assigned wholly to one split. Protein and site intersections across train,
validation, and test must also be zero. A non-compliant version must not be trained.

