"""
Calls per-patient CNV segments from the 10kb-bin CNA table, instead of
treating every fixed bin as its own node (build_cna_graph.py's approach).

Inspecting the raw values shows the 10kb bins are already piecewise-constant
within each patient (a "staircase": long runs of bins sharing the exact
same value) -- i.e. this table is a pre-segmented/smoothed CNA profile that
was broadcast onto a uniform bin grid, not raw noisy probe-level data. So
"calling a segment" is just: within each of the 7 contiguous genomic runs
(from cna_graph.pt's run_id), merge consecutive bins with an (almost)
identical value into one segment. No changepoint/CBS algorithm needed.

This is the "CNV segments as nodes" graph: unlike the fixed 7460-bin grid,
every patient gets THEIR OWN graph (different breakpoints -> different
number of segments -- median ~11, mean ~15, max ~150 in a 300-patient
sample). Each segment node's features are (value, log-length), and edges
chain consecutive segments within a run, with a few boundary edges linking
across the 7 runs (mirroring build_cna_graph.py's cross-region hubs).

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_segments.pkl
  -- dict: patient_ids (list[str]), segments (list of length len(patient_ids);
     each entry is a float32 array of shape (n_segments_i, 3) with columns
     [value, log1p(length_in_bins), run_id])
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segments.pkl")

VALUE_TOL = 1e-6


def call_segments_for_row(row, run_boundary_mask):
    """row: (n_bins,) float array for one patient, sorted in genomic (run) order.
    run_boundary_mask: (n_bins,) bool, True at the first bin of each run (a
    breakpoint is forced there regardless of value, so segments never span
    a run boundary). Returns (n_segments, 3) array of
    [value, log1p(length_in_bins), run_id_placeholder=-1 (filled by caller)]."""
    n_bins = len(row)
    value_break = np.empty(n_bins, dtype=bool)
    value_break[0] = True
    np.greater(np.abs(np.diff(row)), VALUE_TOL, out=value_break[1:])
    breaks = value_break | run_boundary_mask
    starts = np.flatnonzero(breaks)
    ends = np.append(starts[1:], n_bins)
    lengths = ends - starts
    values = row[starts]
    return starts, values.astype(np.float32), np.log1p(lengths).astype(np.float32)


def main():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    run_id = graph["run_id"].numpy()
    n_runs = graph["n_runs"]
    n_bins = len(run_id)

    run_boundary_mask = np.empty(n_bins, dtype=bool)
    run_boundary_mask[0] = True
    run_boundary_mask[1:] = run_id[1:] != run_id[:-1]

    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == graph["bin_columns"]

    patient_ids = list(cna.index)
    X = cna.values.astype(np.float32)

    segments = []
    seg_counts = []
    for i in range(X.shape[0]):
        starts, values, log_lengths = call_segments_for_row(X[i], run_boundary_mask)
        seg_run_id = run_id[starts].astype(np.float32)
        seg = np.stack([values, log_lengths, seg_run_id], axis=1)
        segments.append(seg)
        seg_counts.append(len(seg))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{X.shape[0]} patients segmented")

    seg_counts = np.array(seg_counts)
    print(f"Segments per patient: min={seg_counts.min()} median={np.median(seg_counts):.0f} "
          f"mean={seg_counts.mean():.1f} max={seg_counts.max()}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"patient_ids": patient_ids, "segments": segments, "n_runs": n_runs}, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
