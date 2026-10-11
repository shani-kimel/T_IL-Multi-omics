"""
Builds the gene-level CNA representation: maps each 10kb bin's CNV value
onto the protein-coding genes it overlaps (from gene_annotations.csv,
fetched from Ensembl GRCh37), producing one CNV value per gene per patient
(mean of the bins that gene spans), plus a gene-adjacency graph.

This is the "genes as nodes" graph the user asked for, as a middle ground
that's actually buildable here: a real gene identity and real genomic
coordinates per node (unlike the arbitrary-bin grid), with a
genomic-neighbor edge structure computed directly from those coordinates.
A true PPI/pathway edge set (STRING, Reactome, etc.) is attempted
separately in fetch_string_ppi.py since it depends on external service
reachability; this script only depends on Ensembl, which we've confirmed
we can reach.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/gene_cna_graph.pt
  -- dict: gene_names (list[str]), gene_ids (list[str]), n_genes,
     bin_to_gene (sparse row-normalized (n_genes x n_bins) mean-pooling
     matrix), genomic_edge_index (2xE, chain + cross-run hub edges)
"""
import os

import numpy as np
import pandas as pd
import torch

ROOT = "/Users/yariv/Desktop/shani project"
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
GENES_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_annotations.csv")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_cna_graph.pt")

BIN_SIZE = 10_000


def main():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    bin_columns = graph["bin_columns"]
    run_id = graph["run_id"].numpy()
    n_bins = len(bin_columns)

    import re
    bin_re = re.compile(r"\d+_chr([0-9XY]+)_(\d+)to(\d+)_10k")
    bin_chrom, bin_start, bin_end = [], [], []
    for c in bin_columns:
        m = bin_re.match(c)
        bin_chrom.append(m.group(1))
        bin_start.append(int(m.group(2)) * BIN_SIZE)
        bin_end.append(int(m.group(3)) * BIN_SIZE)
    bin_chrom = np.array(bin_chrom)
    bin_start = np.array(bin_start)
    bin_end = np.array(bin_end)

    genes = pd.read_csv(GENES_PATH)
    genes["chrom"] = genes["chrom"].astype(str)
    genes = genes.sort_values(["chrom", "start"]).reset_index(drop=True)

    gene_bin_lists = []
    gene_run = []
    for _, g in genes.iterrows():
        overlap = np.where(
            (bin_chrom == g["chrom"]) & (bin_start < g["end"]) & (bin_end > g["start"])
        )[0]
        gene_bin_lists.append(overlap)
        gene_run.append(run_id[overlap[0]] if len(overlap) else -1)
    genes["n_bins_overlap"] = [len(x) for x in gene_bin_lists]
    genes["run_id"] = gene_run

    n_dropped = (genes["n_bins_overlap"] == 0).sum()
    print(f"{n_dropped} of {len(genes)} genes have zero overlapping bins (in coverage gaps) -- dropping them")
    keep = genes["n_bins_overlap"] > 0
    genes = genes[keep].reset_index(drop=True)
    gene_bin_lists = [b for b, k in zip(gene_bin_lists, keep) if k]

    n_genes = len(genes)
    print(f"{n_genes} genes retained, with 1-{max(len(b) for b in gene_bin_lists)} overlapping bins each")

    # Order genes by (run, genomic position) so genomic-neighbor edges below
    # are just consecutive-index chains; re-derive each gene's bin overlap
    # in this new order (cheap: only ~a few hundred genes).
    order = sorted(range(n_genes), key=lambda i: (genes.loc[i, "run_id"], genes.loc[i, "start"]))
    genes = genes.iloc[order].reset_index(drop=True)
    gene_bin_lists = [gene_bin_lists[i] for i in order]

    # Row-normalized (mean-pooling) bin -> gene matrix, sparse.
    rows, cols, vals = [], [], []
    for gi, bins in enumerate(gene_bin_lists):
        w = 1.0 / len(bins)
        for b in bins:
            rows.append(gi)
            cols.append(b)
            vals.append(w)
    bin_to_gene = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long), torch.tensor(vals, dtype=torch.float32), (n_genes, n_bins)
    ).coalesce()

    run_ids_sorted = genes["run_id"].values
    edges = []
    for i in range(1, n_genes):
        if run_ids_sorted[i] == run_ids_sorted[i - 1]:
            edges.append((i - 1, i))
    run_start_idx = {}
    for i, r in enumerate(run_ids_sorted):
        if r not in run_start_idx:
            run_start_idx[r] = i
    starts = [run_start_idx[r] for r in sorted(run_start_idx)]
    for a in range(len(starts) - 1):
        edges.append((starts[a], starts[a + 1]))

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    print(f"genomic-neighbor graph: {n_genes} genes, {edge_index.shape[1]} directed edges")

    torch.save(
        {
            "gene_names": genes["external_name"].tolist(),
            "gene_ids": genes["gene_id"].tolist(),
            "n_genes": n_genes,
            "bin_to_gene": bin_to_gene,
            "genomic_edge_index": edge_index,
            "bin_columns": bin_columns,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
