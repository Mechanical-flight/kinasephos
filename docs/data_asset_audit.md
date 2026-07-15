# Uploaded data asset audit

Audit date: 2026-07-14. Raw files remain outside Git and were not modified.

| Supplied asset | SHA-256 | Observation | Decision |
|---|---|---|---|
| `Homo sapiens.fasta` | `16bc120a...a68f7` | 75,248 records; EPSD-like headers; 12,466 duplicate sequences; 37 terminal `*` characters | Private, provenance/species verification required; not accepted as the formal reference yet |
| `PhosphoSitePlus(2).zip` | `edce6587...4256` | Safe single member, but its table schema starts `EPSD ID, UniProt, Position, Kinase, Source` | Private/restricted and source-name conflict; not described as official PSP data |
| UniProt-named `.xlsx.gz` | `17934977...3ee0` | One visible sheet; 3,487 data rows, 7 columns; includes multiple organisms and kinase-like proteins | Private audit input; not a human phosphosite label table |
| Work Mode prompt | `0d5bee03...590a` | Architecture and governance contract, not training data | Used as implementation specification; not copied as a raw asset |

No byte-identical files, unreadable files, obvious secret filenames, model weights, or checkpoints were
found among the four supplied top-level assets. The misleading archive/workbook names require
conservative lineage handling. Formal binary and family training remain `TBD`.

