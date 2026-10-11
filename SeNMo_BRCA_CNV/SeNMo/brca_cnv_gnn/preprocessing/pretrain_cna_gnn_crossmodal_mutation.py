"""
Fused GNN + cross-modal pretraining, mutation variant: masked-bin
reconstruction (as in pretrain_cna_gnn.py) plus a head predicting mutation
status from the same pooled graph embedding, instead of mRNA
(pretrain_cna_gnn_crossmodal.py). Mutation is binary per-gene
presence/absence and very sparse (~1.9% positive rate pan-cancer), so the
cross-modal head uses BCEWithLogitsLoss and mutation is left as raw {0,1}
-- same convention as pretrain_cnv_mutation_vae.py.

Motivation: the standalone (non-fused) CNV<->mutation VAE was the best
single embedding found so far (+CNV-MutationVAE-MLPhead: 0.7938, the
lowest-variance config across all folds), clearly ahead of the mRNA and
methylation variants -- plausibly because CNV and mutation are both
DNA-level alterations. Worth checking whether that same DNA-level signal
survives being fused into the GNN's graph-structured encoder, given the
mRNA version of this fusion underperformed plain GNN twice (hidden_dim 16
and 32).

hidden_dim=16 here (not the failed 32 from the mRNA hd32 run) -- matching
plain GNN's own working width rather than repeating the widening that
didn't help.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_gnn_crossmodal_mutation_encoder.pt
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
MUT_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_Mutation_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_gnn_crossmodal_mutation_encoder.pt")

HIDDEN_DIM = 16
MUT_HIDDEN = (128,)
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
    mut = pd.read_csv(MUT_PATH).set_index("ID")
    mut_cols = list(mut.columns)
    assert list(cna.columns) == bin_columns, "CNA csv columns must match the graph's bin order"

    non_brca_ids = sorted(set(clinical[clinical["Cancer_Type"] != "BRCA"]["ID"]) & set(cna.index) & set(mut.index))
    print(f"Pretraining pool (non-BRCA, has CNA + mutation): {len(non_brca_ids)} patients")

    X = cna.loc[non_brca_ids].values.astype(np.float32)
    Y = mut.loc[non_brca_ids, mut_cols].values.astype(np.float32)
    print(f"mutation positive rate: {Y.mean():.4%}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    cna_mean, cna_std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    cna_std[cna_std == 0] = 1.0
    X = (X - cna_mean) / cna_std
    # Mutation stays raw 0/1 -- BCEWithLogitsLoss needs {0,1} targets.

    X_train, Y_train = torch.tensor(X[train_idx]), torch.tensor(Y[train_idx])
    X_val, Y_val = torch.tensor(X[val_idx]), torch.tensor(Y[val_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} bins={n_bins} mut_dim={Y.shape[1]}")

    model = CNAGraphCrossModalAutoencoder(hidden_dim=HIDDEN_DIM, mrna_dim=Y.shape[1],
                                           mrna_hidden=MUT_HIDDEN, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_batch(batch_x, batch_y, train):
        batch_size = batch_x.shape[0]
        mask = torch.rand(batch_size, n_bins) < MASK_RATIO
        x_in = batch_x.clone()
        x_in[mask] = 0.0
        x_in = x_in.t().unsqueeze(-1)  # (n_bins, batch_size, 1)

        model.train(train)
        self_recon, pooled, mut_pred = model(x_in, adj)
        self_recon = self_recon.squeeze(-1).t()  # (batch_size, n_bins)

        self_loss = nn.functional.mse_loss(self_recon[mask], batch_x[mask])
        cross_loss = nn.functional.binary_cross_entropy_with_logits(mut_pred, batch_y)
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
