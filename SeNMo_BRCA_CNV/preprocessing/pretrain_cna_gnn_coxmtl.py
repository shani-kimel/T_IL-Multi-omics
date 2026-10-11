"""
More advanced pretraining for the fixed-bin-grid CNA-GNN encoder: multi-task
objective = masked-bin reconstruction (as before) + a Cox survival loss on
the pooled graph embedding, using the non-BRCA pan-cancer cohort's OWN real
survival outcomes (OS/OS.time -- 7417 of 7480 non-BRCA patients have both).

Rationale: pure self-supervised reconstruction doesn't know anything about
what actually predicts survival; it worked reasonably for the fixed-bin
grid (lots of redundancy to reconstruct from), but there's no reason to
expect a reconstruction-only embedding to be optimally *risk-relevant*.
Adding a real Cox task on ~7400 labeled pan-cancer patients (12x more
labels than the ~600 BRCA-only patients available for the downstream task)
should push the embedding toward encoding risk-relevant CNV structure
directly, while the reconstruction term keeps it regularized/general rather
than overfit to survival alone.

The Cox loss here pools all cancer types together in one partial-likelihood
risk set (unstratified) -- the same convention SeNMo's own pancancer_combined
training already uses in this codebase (train_senmo_brca.py's CoxLoss), so
this isn't a new modeling choice, just applied to a new component. Batch
size is increased vs. the reconstruction-only run (64 -> 256) so each Cox
risk set is large enough to be a meaningful comparison.

Still strictly BRCA-leakage-free: only non-BRCA patients, in both loss terms.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_gnn_encoder_coxmtl.pt
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cna_gnn_model import CNAGraphAutoencoder, build_normalized_adjacency, pool_embedding
from cox_utils import cox_ptl_loss

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_gnn_encoder_coxmtl.pt")

HIDDEN_DIM = 16
MASK_RATIO = 0.15
BATCH_SIZE = 256
EPOCHS = 60
LR = 5e-3
WEIGHT_DECAY = 1e-5
LAMBDA_COX = 1.0
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
    assert list(cna.columns) == bin_columns

    non_brca = clinical[(clinical["Cancer_Type"] != "BRCA") & clinical["OS"].notna() & clinical["OS.time"].notna()]
    non_brca_ids = sorted(set(non_brca["ID"]) & set(cna.index))
    print(f"Pretraining pool (non-BRCA, has CNA + survival): {len(non_brca_ids)} patients")

    X = cna.loc[non_brca_ids].values.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    e = clinical.loc[non_brca_ids, "OS"].astype(np.float32).values
    t = clinical.loc[non_brca_ids, "OS.time"].astype(np.float32).values

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, e_train, t_train = torch.tensor(X[train_idx]), torch.tensor(e[train_idx]), torch.tensor(t[train_idx])
    X_val, e_val, t_val = torch.tensor(X[val_idx]), torch.tensor(e[val_idx]), torch.tensor(t[val_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} bins={n_bins}")

    model = CNAGraphAutoencoder(hidden_dim=HIDDEN_DIM, dropout=0.1)
    cox_head = nn.Linear(2 * HIDDEN_DIM, 1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cox_head.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )

    def run_batch(batch_x, batch_e, batch_t, train):
        batch_size = batch_x.shape[0]
        mask = torch.rand(batch_size, n_bins) < MASK_RATIO
        x_in = batch_x.clone()
        x_in[mask] = 0.0
        x_in = x_in.t().unsqueeze(-1)  # (n_bins, batch_size, 1)

        model.train(train)
        cox_head.train(train)
        recon, node_emb = model(x_in, adj)
        recon = recon.squeeze(-1).t()
        recon_loss = nn.functional.mse_loss(recon[mask], batch_x[mask])

        pooled = pool_embedding(node_emb)  # (batch_size, 2*hidden_dim)
        risk_pred = cox_head(pooled)
        cox_loss = cox_ptl_loss(risk_pred, batch_t, batch_e)

        loss = recon_loss + LAMBDA_COX * cox_loss
        return loss, recon_loss, cox_loss

    for epoch in range(1, EPOCHS + 1):
        perm_e = torch.randperm(X_train.shape[0])
        sums = {"loss": 0.0, "recon": 0.0, "cox": 0.0}
        n_batches = 0
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm_e[start:start + BATCH_SIZE]
            if len(idx) < 8:  # too small a risk set for a stable Cox loss
                continue
            optimizer.zero_grad()
            loss, recon_loss, cox_loss = run_batch(X_train[idx], e_train[idx], t_train[idx], train=True)
            loss.backward()
            optimizer.step()
            sums["loss"] += loss.item()
            sums["recon"] += recon_loss.item()
            sums["cox"] += cox_loss.item()
            n_batches += 1
        for k in sums:
            sums[k] /= n_batches

        with torch.no_grad():
            val_loss, val_recon, val_cox = run_batch(X_val, e_val, t_val, train=False)

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train[loss {sums['loss']:.4f} recon {sums['recon']:.4f} cox {sums['cox']:.4f}]"
                  f"  val[loss {val_loss.item():.4f} recon {val_recon.item():.4f} cox {val_cox.item():.4f}]")

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
