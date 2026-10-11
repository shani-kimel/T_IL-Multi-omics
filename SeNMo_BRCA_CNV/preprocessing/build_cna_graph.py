"""
Builds a shared genomic-adjacency graph over the 7460 CNA bins.

The CNA columns are not genes -- they are 10kb genomic bins
(e.g. "128843_chr7_5473to5474_10k") covering three amplicon-relevant
regions: chr7 (~EGFR, 81 bins), chr8 (~MYC/8q24, 7366 bins), chr11
(~CCND1, 13 bins). There is no gene-gene interaction network available
locally (and no gene symbols to map to one), but CNVs are contiguous
chromosomal segments, so genomically-adjacent bins are the natural graph
edges: a bin's copy-number state is highly correlated with its neighbors'.

This script parses the column names, groups bins into contiguous runs
(a run breaks whenever the next bin's start position isn't the previous
bin's end), and builds a chain graph within each run (bin i <-> bin i+1).
It also links the last/first bin of each of the 3 chromosome blocks
together (few extra edges) so a 2-layer GNN can, in principle, propagate
information across the EGFR / MYC / CCND1 regions -- co-amplification of
these loci is a documented phenomenon in breast cancer -- without
depending on that structure being learned in the same run graph.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_graph.pt
  -- a dict with keys: bin_columns (list[str], in the same order as the
     CNA csv columns), edge_index (2xE long tensor, undirected, no self
     loops -- self loops are added at normalization time), n_runs, chrom_of_bin
"""
import os
import re

import pandas as pd
import torch

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")

BIN_RE = re.compile(r"\d+_chr([0-9XY]+)_(\d+)to(\d+)_10k")


def parse_bins(columns):
    chroms, starts, ends = [], [], []
    for c in columns:
        m = BIN_RE.match(c)
        if not m:
            raise ValueError(f"Column name doesn't match expected bin pattern: {c}")
        chroms.append(m.group(1))
        starts.append(int(m.group(2)))
        ends.append(int(m.group(3)))
    return chroms, starts, ends


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    columns = pd.read_csv(CNA_PATH, nrows=0).columns.tolist()[1:]  # drop "ID"
    n_bins = len(columns)
    chroms, starts, ends = parse_bins(columns)

    # Group into contiguous runs: a run continues while chrom matches and
    # this bin's start == previous bin's end.
    run_id = [0] * n_bins
    run_boundaries = [0]  # bin indices where a new run starts
    for i in range(1, n_bins):
        same_run = (chroms[i] == chroms[i - 1]) and (starts[i] == ends[i - 1])
        run_id[i] = run_id[i - 1] if same_run else run_id[i - 1] + 1
        if not same_run:
            run_boundaries.append(i)
    n_runs = run_id[-1] + 1
    print(f"{n_bins} bins parsed into {n_runs} contiguous run(s); run start indices: {run_boundaries}")
    print("chrom distribution:", pd.Series(chroms).value_counts().to_dict())

    # Chain edges within each contiguous run.
    edges = []
    for i in range(1, n_bins):
        if run_id[i] == run_id[i - 1]:
            edges.append((i - 1, i))

    # A handful of cross-region edges linking each run's boundary bins to the
    # next run's boundary bins, so a 2-layer GNN can reach across the
    # EGFR / MYC / CCND1 regions without them being otherwise disconnected.
    for r in range(n_runs - 1):
        start_next = run_boundaries[r + 1]
        end_prev = run_boundaries[r + 1] - 1
        edges.append((end_prev, start_next))  # already added above if adjacent; harmless duplicate is fine
        # Also connect the very first bin of run r to the very first bin of every other run,
        # giving each region a shared low-degree "hub" path.
        start_r = run_boundaries[r]
        edges.append((start_r, start_next))

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # 2 x E, directed one-way
    # Make undirected.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    print(f"edge_index: {edge_index.shape[1]} directed edges (undirected pairs included)")

    out_path = os.path.join(OUT_DIR, "cna_graph.pt")
    torch.save(
        {
            "bin_columns": columns,
            "edge_index": edge_index,
            "n_bins": n_bins,
            "n_runs": n_runs,
            "run_id": torch.tensor(run_id, dtype=torch.long),
            "chrom_of_bin": chroms,
        },
        out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
