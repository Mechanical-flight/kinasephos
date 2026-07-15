# Model card

## Intended use

Research and engineering evaluation of local-sequence phosphorylation prediction and four broad
kinase groups. It is not an experimentally validated assay, a clinical device, or a protein design
closed loop.

## Architecture

- Input: 61 residues; center index 30 (zero-based) must be S/T/Y.
- Teacher: frozen `facebook/esm2_t33_650M_UR50D`, 33 layers, hidden size 1280, 20 heads.
- Student: four encoder layers, hidden 256, eight heads, FFN 1024, mask-aware mean pool.
- Projection: 256 to 1280.
- Heads: binary `1280 -> 256 -> 1`; family `1280 -> 256 -> 4`.
- Loss: BCE-with-logits + masked cross-entropy + representation MSE.

## Training and evaluation status

Only synthetic smoke execution is represented in this repository. There is no released biological
checkpoint and no formally reproduced metric. Historical presentation figures remain unverified.

## Risks

Annotation incompleteness, source bias, homolog leakage, isoform coordinate mismatch, broad family
labels, and missing structural/cellular context can all cause misleading predictions.

