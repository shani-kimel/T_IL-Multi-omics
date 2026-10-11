"""
Self-supervised pretraining of the segment-graph GNN encoder, on the same
non-BRCA pan-cancer pool as pretrain_cna_gnn.py (no BRCA patient touches
pretraining). Objective: masked-node reconstruction -- zero out a random
subset of segment nodes' [value, log_length] features, encode with 2-layer
GraphSAGE, decode back to [value, log_length], MSE loss on the masked nodes
only.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_segment_gnn_encoder.pt
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from cna_segment_gnn_model import SegmentGraphAutoencoder, segments_to_graph

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
SEGMENTS_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segments.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segment_gnn_encoder.pt")

HIDDEN_DIM = 16
MASK_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 60
LR = 5e-3
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.1
SEED = 66


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(SEGMENTS_PATH, "rb") as f:
        seg_data = pickle.load(f)
    patient_ids = seg_data["patient_ids"]
    segments = seg_data["segments"]
    n_runs = seg_data["n_runs"]

    clinical = pd.read_csv(CLINICAL_PATH).set_index("ID", drop=False)
    id_to_cancer = clinical["Cancer_Type"].to_dict()
    non_brca_idx = [i for i, pid in enumerate(patient_ids) if id_to_cancer.get(pid) != "BRCA"]
    print(f"Pretraining pool (non-BRCA): {len(non_brca_idx)} patients")

    all_values = np.concatenate([segments[i][:, 0] for i in non_brca_idx])
    all_lengths = np.concatenate([segments[i][:, 1] for i in non_brca_idx])
    mean = np.array([all_values.mean(), all_lengths.mean()], dtype=np.float32)
    std = np.array([all_values.std(), all_lengths.std()], dtype=np.float32)
    std[std == 0] = 1.0
    print(f"feature mean={mean} std={std}")

    def make_graph(i):
        seg = segments[i].copy()
        seg[:, 0] = (seg[:, 0] - mean[0]) / std[0]
        seg[:, 1] = (seg[:, 1] - mean[1]) / std[1]
        return segments_to_graph(seg, n_runs)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_idx))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_local, train_local = perm[:n_val], perm[n_val:]
    train_graphs = [make_graph(non_brca_idx[i]) for i in train_local]
    val_graphs = [make_graph(non_brca_idx[i]) for i in val_local]
    print(f"train={len(train_graphs)} val={len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = SegmentGraphAutoencoder(hidden_dim=HIDDEN_DIM, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_batch(batch, train):
        mask = torch.rand(batch.x.shape[0]) < MASK_RATIO
        x_in = batch.x.clone()
        x_in[mask] = 0.0

        model.train(train)
        recon, _ = model(x_in, batch.edge_index)
        loss = nn.functional.mse_loss(recon[mask], batch.x[mask])
        return loss

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = run_batch(batch, train=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        with torch.no_grad():
            val_losses = [run_batch(batch, train=False).item() for batch in val_loader]
            val_loss = float(np.mean(val_losses))

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train MSE {train_loss:.4f}  val MSE {val_loss:.4f}")

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "hidden_dim": HIDDEN_DIM,
            "n_runs": n_runs,
            "scaler_mean": mean,
            "scaler_std": std,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
