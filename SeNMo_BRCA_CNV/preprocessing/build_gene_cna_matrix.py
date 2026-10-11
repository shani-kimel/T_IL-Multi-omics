"""
Combines gene_cna_graph.pt (genomic-neighbor edges + bin->gene pooling
matrix) with gene_ppi_edges.csv (STRING PPI edges) into one gene-level
graph, and materializes the per-patient gene-level CNA matrix (8238 x 344)
for the full pan-cancer cohort by applying the bin->gene mean-pooling
matrix to the raw bin-level CNA table -- computed once here so pretraining
and extraction don't need to repeat the sparse matmul per script.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/gene_cna_full.pt
  -- dict: gene_names, patient_ids, gene_cna (n_patients x n_genes float32),
     edge_index (2xE, genomic-neighbor + PPI, deduplicated)
"""
import os

import numpy as np
import pandas as pd
import torch

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
GENE_GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_cna_graph.pt")
PPI_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_ppi_edges.csv")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_cna_full.pt")


def main():
    gene_graph = torch.load(GENE_GRAPH_PATH, weights_only=False)
    gene_names = gene_graph["gene_names"]
    n_genes = gene_graph["n_genes"]
    bin_to_gene = gene_graph["bin_to_gene"]
    bin_columns = gene_graph["bin_columns"]
    name_to_idx = {name: i for i, name in enumerate(gene_names)}

    ppi = pd.read_csv(PPI_PATH)
    ppi_edges = []
    n_skipped_dup_name = 0
    for _, row in ppi.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        if a not in name_to_idx or b not in name_to_idx:
            n_skipped_dup_name += 1
            continue
        ppi_edges.append((name_to_idx[a], name_to_idx[b]))

    genomic_edges = gene_graph["genomic_edge_index"].t().tolist()
    all_edges = genomic_edges + ppi_edges
    edge_index = torch.tensor(all_edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    print(f"combined graph: {n_genes} genes, {len(genomic_edges)//2} genomic-neighbor edges + "
          f"{len(ppi_edges)//2} PPI edges -> {edge_index.shape[1]} directed edges total")

    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == bin_columns
    patient_ids = list(cna.index)
    X_bins = torch.tensor(cna.values.astype(np.float32))  # (n_patients, n_bins)

    # gene_cna[p, g] = mean over bins overlapping gene g of X_bins[p, bin]
    # bin_to_gene is (n_genes x n_bins) row-normalized -> gene_cna = X_bins @ bin_to_gene.T
    gene_cna = torch.sparse.mm(bin_to_gene, X_bins.t()).t()  # (n_patients, n_genes)
    gene_cna = gene_cna.numpy().astype(np.float32)
    print(f"gene_cna matrix: {gene_cna.shape}")
    assert not np.isnan(gene_cna).any()

    torch.save(
        {
            "gene_names": gene_names,
            "patient_ids": patient_ids,
            "gene_cna": gene_cna,
            "edge_index": edge_index,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
