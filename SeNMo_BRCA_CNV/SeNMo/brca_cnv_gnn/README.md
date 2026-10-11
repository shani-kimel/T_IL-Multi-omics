# BRCA CNV-as-graph experiments

Snapshot of the scripts behind an investigation into adding CNV data to SeNMo's
BRCA survival prediction via graph neural networks, after naive concatenation
and PCA-reduced CNV both underperformed the no-CNV baseline.

Data files, pretrained checkpoints, and extracted embeddings are **not**
included (`data/embedding_experiment/*.pt`, `*.pkl`) — only the source. Paths
in these scripts are absolute (`/Users/yariv/Desktop/shani project/...`) and
will need adapting to wherever the TCGA PanCancer Atlas data and this repo
actually live.

## Pipeline order

1. `preprocessing/build_cna_graph.py` — genomic-adjacency graph over the raw
   10kb CNA bins (fixed grid, shared across patients).
2. `preprocessing/call_cna_segments.py` + `cna_segment_gnn_model.py` — per-patient
   CNV-segment graph (variable topology).
3. `preprocessing/fetch_gene_annotations.py` (Ensembl GRCh37) and
   `fetch_string_ppi.py` (STRING PPI, medium confidence) + `build_gene_level_cna.py`
   / `build_gene_cna_matrix.py` — real genes-as-nodes graph with genomic +
   protein-protein-interaction edges.
4. `preprocessing/pretrain_cna_gnn.py` / `pretrain_cna_segment_gnn.py` /
   `pretrain_gene_gnn.py` (+ `_coxmtl` variants) — self-supervised (masked-node
   reconstruction) pretraining of each graph's encoder on the non-BRCA
   pan-cancer cohort only.
5. `preprocessing/extract_*_embeddings.py` — frozen encoder → pooled embedding
   for the BRCA cohort.
6. `finetune_cna_gnn_brca.py` — end-to-end fine-tune of the grid-GNN's last
   layer directly on BRCA.
7. `train_embedding_experiment.py` — the actual comparison: no-CNV baseline,
   raw/PCA CNV, every graph variant × {linear, MLP} fusion head, and the two
   ensembling strategies (feature concatenation vs. prediction averaging).

## Result

Best result prior to the cross-modal extension below: averaging the risk
scores of the grid-GNN and gene+PPI-GNN (each frozen + MLP fusion head) —
mean test C-index 0.790 vs. 0.774 no-CNV baseline, 5-fold CV. See
`train_embedding_experiment.py`'s summary table for the full comparison.

## Cross-modal extension (CNV&harr;mRNA / methylation / mutation)

Follow-up investigation: instead of only a genomic-adjacency prior on CNV,
align a CNV encoder to a second omics modality so it keeps only
copy-number variation that modality reflects. Same non-BRCA-only
pretraining discipline as the GNN track above.

1. `preprocessing/cnv_mrna_vae_model.py` — `CrossModalVAE`: paired
   MLP encoders (CNV + a second modality) sharing one latent, trained with
   self-reconstruction + cross-reconstruction + a &mu;-alignment term + KL,
   plus a Cox multi-task term on the CNV latent. Generalized via
   `modality_b_dist` (Gaussian for continuous mRNA/methylation, Bernoulli/BCE
   for binary mutation).
2. `preprocessing/pretrain_cnv_{mrna,methylation,mutation}_vae.py` —
   pretrain one instance per second modality on the non-BRCA cohort.
3. `preprocessing/extract_cnv_{mrna,methylation,mutation}_vae_embeddings.py`
   — frozen CNV-encoder inference over the BRCA cohort (only the CNV branch
   runs at inference; the second modality is never needed for BRCA).
4. `preprocessing/cna_gnn_model.py`'s `CNAGraphCrossModalAutoencoder` +
   `preprocessing/pretrain_cna_gnn_crossmodal{,_mutation}.py` +
   `preprocessing/extract_cna_gnn_crossmodal{,_mutation}_embeddings.py` —
   fusion variant: the grid-GNN encoder trained jointly with a cross-modal
   prediction head, instead of two separate models. Tried for mRNA (at
   hidden_dim 16 and 32) and mutation.
5. `minds/pull_brca_cnv_expression.py` — collision-safe multi-pass GDC
   downloader that expanded the BRCA CNV+mRNA cohort from 758 to 1,089
   patients (the pan-cancer file used above is capped at 758 BRCA patients
   because it requires all 4 omics simultaneously); not yet wired into the
   pretraining above, which still runs on the smaller pan-cancer file for
   consistency with the GNN track.

### Findings

- **Mutation is a clearly better cross-modal partner for CNV than mRNA or
  methylation** — plausibly because both are DNA-level alterations sharing
  driver genes/pathways, vs. CNV&rarr;expression's noisier, more indirect
  path. Standalone CNV&harr;Mutation VAE: 0.794 mean test C-index, the best
  single embedding found and the lowest fold-to-fold variance of any config.
- **Joint fusion (one encoder, two objectives) never beat training two
  specialized encoders separately and ensembling their predictions** — true
  for both mRNA and mutation, at multiple widths. The reconstruction and
  cross-modal objectives compete for one bottleneck rather than
  reinforcing each other.
- **Best overall: grid-GNN + standalone CNV&harr;Mutation VAE, ensembled**
  (prediction-averaged, `+CNV-GNN+MutationVAE-PredEnsemble` in
  `train_embedding_experiment.py`) — mean test C-index **0.799** vs. 0.774
  no-CNV baseline, 5-fold CV.
