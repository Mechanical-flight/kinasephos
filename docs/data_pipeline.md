# Data pipeline

1. Inventory every input by relative path, type, size, time, SHA-256, readability, archive status,
   duplication, and sensitivity indicators.
2. Inspect archive members before extraction and reject traversal or extreme compression ratios.
3. Select one provenance-verified human reference proteome.
4. Normalize accessions through explicit maps; preserve raw identifiers and isoform status.
5. Validate residue and 1-based coordinate, then construct a padded 61-mer.
6. Merge evidence by canonical `accession:residue+position`, retaining source sets.
7. Sample reproducible, stratified unannotated S/T/Y candidates after removing all known positives.
8. Cluster proteins and split whole clusters; assert zero protein, cluster, and site overlap.

