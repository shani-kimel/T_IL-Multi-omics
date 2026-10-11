"""
Self-supervised pretraining of the CNA-GNN encoder on the pan-cancer cohort,
EXCLUDING all BRCA patients -- so no patient that could later appear in a
BRCA train or test fold ever touches GNN pretraining. This keeps the
downstream comparison against no-CNV / +CNV-raw / +CNV-PCA (which use the
same held-out BRCA folds) leakage-free.

Objective: masked-bin reconstruction. For each patient, a random subset of
bins have their (standardized) CNA value zeroed out; the GCN encoder/decoder
must reconstruct the true value from graph context (genomically-adjacent
bins) alone. This is the graph analogue of a denoising autoencoder, and
gives the encoder something to learn from ~7480 patients rather than the
~600-700 in BRCA alone.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_gnn_encoder.pt
  -- dict with encoder state_dict, scaler mean/scale, hidden_dim, and the
     bin_columns / edge_index it was trained on (for a consistency check
     at extraction time).
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cna_gnn_model import CNAGraphAutoencoder, build_normalized_adjacency

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_gnn_encoder.pt")

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

    graph = torch.load(GRAPH_PATH, weights_only=False)
    bin_columns = graph["bin_columns"]
    n_bins = graph["n_bins"]
    adj = build_normalized_adjacency(graph["edge_index"], n_bins)

    clinical = pd.read_csv(CLINICAL_PATH).set_index("ID", drop=False)
    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == bin_columns, "CNA csv columns must match the graph's bin order"

    non_brca_ids = sorted(set(clinical[clinical["Cancer_Type"] != "BRCA"]["ID"]) & set(cna.index))
    print(f"Pretraining pool (non-BRCA, has CNA): {len(non_brca_ids)} patients")

    X = cna.loc[non_brca_ids].values.astype(np.float32)  # (n_patients, n_bins)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train = torch.tensor(X[train_idx])  # (n_train, n_bins)
    X_val = torch.tensor(X[val_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} bins={n_bins}")

    model = CNAGraphAutoencoder(hidden_dim=HIDDEN_DIM, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_batch(batch_x, train):
        # batch_x: (batch_size, n_bins)
        batch_size = batch_x.shape[0]
        mask = (torch.rand(batch_size, n_bins) < MASK_RATIO)  # True = held out, must reconstruct
        x_in = batch_x.clone()
        x_in[mask] = 0.0
        x_in = x_in.t().unsqueeze(-1)  # (n_bins, batch_size, 1)

        if train:
            model.train()
        else:
            model.eval()
        recon, _ = model(x_in, adj)  # (n_bins, batch_size, 1)
        recon = recon.squeeze(-1).t()  # (batch_size, n_bins)

        target = batch_x[mask]
        pred = recon[mask]
        loss = nn.functional.mse_loss(pred, target)
        return loss

    for epoch in range(1, EPOCHS + 1):
        perm_e = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm_e[start:start + BATCH_SIZE]
            batch_x = X_train[idx]
            optimizer.zero_grad()
            loss = run_batch(batch_x, train=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        with torch.no_grad():
            val_losses = []
            for start in range(0, X_val.shape[0], BATCH_SIZE):
                batch_x = X_val[start:start + BATCH_SIZE]
                val_losses.append(run_batch(batch_x, train=False).item())
            val_loss = float(np.mean(val_losses))

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train MSE {train_loss:.4f}  val MSE {val_loss:.4f}")

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "hidden_dim": HIDDEN_DIM,
            "bin_columns": bin_columns,
            "edge_index": graph["edge_index"],
            "scaler_mean": mean,
            "scaler_std": std,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
