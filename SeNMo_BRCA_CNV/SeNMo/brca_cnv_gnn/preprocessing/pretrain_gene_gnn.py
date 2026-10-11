"""
Self-supervised pretraining of the gene-level CNA-GNN: same masked-node
reconstruction recipe as pretrain_cna_gnn.py (the grid-bin version), but
over the 344-gene graph (gene_cna_full.pt) with real genomic-neighbor +
STRING PPI edges instead of 7460 arbitrary bins. Same non-BRCA-only,
leakage-free pretraining pool.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/gene_gnn_encoder.pt
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cna_gnn_model import CNAGraphAutoencoder, build_normalized_adjacency

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
GENE_CNA_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_cna_full.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_gnn_encoder.pt")

HIDDEN_DIM = 16
MASK_RATIO = 0.15
BATCH_SIZE = 128
EPOCHS = 60
LR = 5e-3
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.1
SEED = 66


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    gene_data = torch.load(GENE_CNA_PATH, weights_only=False)
    gene_names = gene_data["gene_names"]
    patient_ids = gene_data["patient_ids"]
    n_genes = len(gene_names)
    adj = build_normalized_adjacency(gene_data["edge_index"], n_genes)

    clinical = pd.read_csv(CLINICAL_PATH).set_index("ID", drop=False)
    id_to_cancer = clinical["Cancer_Type"].to_dict()
    non_brca_idx = [i for i, pid in enumerate(patient_ids) if id_to_cancer.get(pid) != "BRCA"]
    print(f"Pretraining pool (non-BRCA): {len(non_brca_idx)} patients, {n_genes} genes")

    X = gene_data["gene_cna"][non_brca_idx]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_idx))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train = torch.tensor(X[train_idx])
    X_val = torch.tensor(X[val_idx])
    print(f"train={len(train_idx)} val={len(val_idx)}")

    model = CNAGraphAutoencoder(hidden_dim=HIDDEN_DIM, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_batch(batch_x, train):
        batch_size = batch_x.shape[0]
        mask = torch.rand(batch_size, n_genes) < MASK_RATIO
        x_in = batch_x.clone()
        x_in[mask] = 0.0
        x_in = x_in.t().unsqueeze(-1)

        model.train(train)
        recon, _ = model(x_in, adj)
        recon = recon.squeeze(-1).t()
        return nn.functional.mse_loss(recon[mask], batch_x[mask])

    for epoch in range(1, EPOCHS + 1):
        perm_e = torch.randperm(X_train.shape[0])
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm_e[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = run_batch(X_train[idx], train=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        with torch.no_grad():
            val_losses = []
            for start in range(0, X_val.shape[0], BATCH_SIZE):
                val_losses.append(run_batch(X_val[start:start + BATCH_SIZE], train=False).item())
            val_loss = float(np.mean(val_losses))

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train MSE {train_loss:.4f}  val MSE {val_loss:.4f}")

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "hidden_dim": HIDDEN_DIM,
            "gene_names": gene_names,
            "edge_index": gene_data["edge_index"],
            "scaler_mean": mean,
            "scaler_std": std,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
