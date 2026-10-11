"""
Same idea as pretrain_cna_gnn_coxmtl.py (multi-task: masked-node
reconstruction + a Cox survival loss using the non-BRCA pan-cancer cohort's
own real OS/OS.time), applied to the per-patient segment-graph encoder.

This one matters more than the grid version's upgrade: the segment graphs'
pure self-supervised pretraining was weak (median ~10 nodes per graph gives
little redundancy for masked-node reconstruction to exploit; val MSE
plateaued around 0.5, barely above chance). A directly survival-supervised
signal from ~7400 pan-cancer patients doesn't depend on reconstruction
redundancy at all, so it may recover useful signal that reconstruction
alone couldn't find in such small graphs.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cna_segment_gnn_encoder_coxmtl.pt
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from cna_segment_gnn_model import SegmentGraphAutoencoder, pool_embedding, segments_to_graph
from cox_utils import cox_ptl_loss

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
SEGMENTS_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segments.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segment_gnn_encoder_coxmtl.pt")

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

    import pickle
    with open(SEGMENTS_PATH, "rb") as f:
        seg_data = pickle.load(f)
    patient_ids = seg_data["patient_ids"]
    segments = seg_data["segments"]
    n_runs = seg_data["n_runs"]

    clinical = pd.read_csv(CLINICAL_PATH).set_index("ID", drop=False)
    valid = clinical[(clinical["Cancer_Type"] != "BRCA") & clinical["OS"].notna() & clinical["OS.time"].notna()]
    valid_ids = set(valid["ID"])
    non_brca_idx = [i for i, pid in enumerate(patient_ids) if pid in valid_ids]
    print(f"Pretraining pool (non-BRCA, has segments + survival): {len(non_brca_idx)} patients")

    all_values = np.concatenate([segments[i][:, 0] for i in non_brca_idx])
    all_lengths = np.concatenate([segments[i][:, 1] for i in non_brca_idx])
    mean = np.array([all_values.mean(), all_lengths.mean()], dtype=np.float32)
    std = np.array([all_values.std(), all_lengths.std()], dtype=np.float32)
    std[std == 0] = 1.0

    e_map = clinical["OS"].astype(np.float32).to_dict()
    t_map = clinical["OS.time"].astype(np.float32).to_dict()

    def make_graph(i):
        seg = segments[i].copy()
        seg[:, 0] = (seg[:, 0] - mean[0]) / std[0]
        seg[:, 1] = (seg[:, 1] - mean[1]) / std[1]
        data = segments_to_graph(seg, n_runs)
        pid = patient_ids[i]
        data.e = torch.tensor([e_map[pid]], dtype=torch.float32)
        data.t = torch.tensor([t_map[pid]], dtype=torch.float32)
        return data

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_idx))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_local, train_local = perm[:n_val], perm[n_val:]
    train_graphs = [make_graph(non_brca_idx[i]) for i in train_local]
    val_graphs = [make_graph(non_brca_idx[i]) for i in val_local]
    print(f"train={len(train_graphs)} val={len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)

    model = SegmentGraphAutoencoder(hidden_dim=HIDDEN_DIM, dropout=0.1)
    cox_head = nn.Linear(2 * HIDDEN_DIM, 1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cox_head.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )

    def run_batch(batch, train):
        mask = torch.rand(batch.x.shape[0]) < MASK_RATIO
        x_in = batch.x.clone()
        x_in[mask] = 0.0

        model.train(train)
        cox_head.train(train)
        recon, node_emb = model(x_in, batch.edge_index)
        recon_loss = nn.functional.mse_loss(recon[mask], batch.x[mask])

        pooled = pool_embedding(node_emb, batch.batch)
        risk_pred = cox_head(pooled)
        cox_loss = cox_ptl_loss(risk_pred, batch.t, batch.e)

        loss = recon_loss + LAMBDA_COX * cox_loss
        return loss, recon_loss, cox_loss

    for epoch in range(1, EPOCHS + 1):
        sums = {"loss": 0.0, "recon": 0.0, "cox": 0.0}
        n_batches = 0
        for batch in train_loader:
            if batch.num_graphs < 8:
                continue
            optimizer.zero_grad()
            loss, recon_loss, cox_loss = run_batch(batch, train=True)
            loss.backward()
            optimizer.step()
            sums["loss"] += loss.item()
            sums["recon"] += recon_loss.item()
            sums["cox"] += cox_loss.item()
            n_batches += 1
        for k in sums:
            sums[k] /= n_batches

        with torch.no_grad():
            for val_batch in val_loader:  # single batch (batch_size = len(val_graphs))
                val_loss, val_recon, val_cox = run_batch(val_batch, train=False)

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train[loss {sums['loss']:.4f} recon {sums['recon']:.4f} cox {sums['cox']:.4f}]"
                  f"  val[loss {val_loss.item():.4f} recon {val_recon.item():.4f} cox {val_cox.item():.4f}]")

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
