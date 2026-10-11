"""
Fused GNN + cross-modal pretraining: masked-bin reconstruction (as in
pretrain_cna_gnn.py) plus a head predicting mRNA expression from the same
pooled graph embedding (as in pretrain_cnv_mrna_vae.py's cross-modal idea,
but built on top of the genomic-adjacency GCN encoder instead of a plain
MLP). One model, one embedding, shaped by both priors jointly -- rather
than training the GNN and the cross-modal VAE separately and ensembling
their outputs after the fact.

Same non-BRCA-only pretraining pool and per-fold-safe usage as every other
encoder in this project: BRCA is never touched here, so the frozen encoder
can be applied to all BRCA patients (any fold) with no leakage.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_gnn_crossmodal_encoder.pt
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cna_gnn_model import CNAGraphCrossModalAutoencoder, build_normalized_adjacency

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
EXPR_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_Expression_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_gnn_crossmodal_encoder_hd32.pt")

# Widened from the hidden_dim=16 used everywhere else in the GNN track: the
# hd16 fused run showed the cross-modal head's val loss getting *worse*
# while self-reconstruction kept improving (0.953->1.004 over 60 epochs),
# consistent with the two objectives competing for a bottleneck sized only
# for the reconstruction task alone. Doubling it gives both heads more room
# without changing anything else about the setup.
HIDDEN_DIM = 32
MRNA_HIDDEN = (128,)
MASK_RATIO = 0.15
CROSS_WEIGHT = 0.5
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
    expr = pd.read_csv(EXPR_PATH).set_index("ID")
    expr_cols = list(expr.columns)
    assert list(cna.columns) == bin_columns, "CNA csv columns must match the graph's bin order"

    non_brca_ids = sorted(set(clinical[clinical["Cancer_Type"] != "BRCA"]["ID"]) & set(cna.index) & set(expr.index))
    print(f"Pretraining pool (non-BRCA, has CNA + expression): {len(non_brca_ids)} patients")

    X = cna.loc[non_brca_ids].values.astype(np.float32)
    Y = expr.loc[non_brca_ids, expr_cols].values.astype(np.float32)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    cna_mean, cna_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    cna_std[cna_std == 0] = 1.0
    X = (X - cna_mean) / cna_std

    expr_mean, expr_std = Y[train_idx].mean(axis=0), Y[train_idx].std(axis=0)
    expr_std[expr_std == 0] = 1.0
    Y = (Y - expr_mean) / expr_std

    X_train, Y_train = torch.tensor(X[train_idx]), torch.tensor(Y[train_idx])
    X_val, Y_val = torch.tensor(X[val_idx]), torch.tensor(Y[val_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} bins={n_bins} mrna_dim={Y.shape[1]}")

    model = CNAGraphCrossModalAutoencoder(hidden_dim=HIDDEN_DIM, mrna_dim=Y.shape[1],
                                           mrna_hidden=MRNA_HIDDEN, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_batch(batch_x, batch_y, train):
        batch_size = batch_x.shape[0]
        mask = torch.rand(batch_size, n_bins) < MASK_RATIO
        x_in = batch_x.clone()
        x_in[mask] = 0.0
        x_in = x_in.t().unsqueeze(-1)  # (n_bins, batch_size, 1)

        model.train(train)
        self_recon, pooled, mrna_pred = model(x_in, adj)
        self_recon = self_recon.squeeze(-1).t()  # (batch_size, n_bins)

        self_loss = nn.functional.mse_loss(self_recon[mask], batch_x[mask])
        cross_loss = nn.functional.mse_loss(mrna_pred, batch_y)
        loss = self_loss + CROSS_WEIGHT * cross_loss
        return loss, self_loss, cross_loss

    for epoch in range(1, EPOCHS + 1):
        perm_e = torch.randperm(X_train.shape[0])
        sums = {"loss": 0.0, "self": 0.0, "cross": 0.0}
        n_batches = 0
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm_e[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss, self_loss, cross_loss = run_batch(X_train[idx], Y_train[idx], train=True)
            loss.backward()
            optimizer.step()
            sums["loss"] += loss.item()
            sums["self"] += self_loss.item()
            sums["cross"] += cross_loss.item()
            n_batches += 1
        for k in sums:
            sums[k] /= n_batches

        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                val_loss, val_self, val_cross = run_batch(X_val, Y_val, train=False)
            print(f"epoch {epoch:3d}  train[loss {sums['loss']:.4f} self {sums['self']:.4f} cross {sums['cross']:.4f}]"
                  f"  val[loss {val_loss.item():.4f} self {val_self.item():.4f} cross {val_cross.item():.4f}]")

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "hidden_dim": HIDDEN_DIM,
            "bin_columns": bin_columns,
            "edge_index": graph["edge_index"],
            "scaler_mean": cna_mean,
            "scaler_std": cna_std,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
