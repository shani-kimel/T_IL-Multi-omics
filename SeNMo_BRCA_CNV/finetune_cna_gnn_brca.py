"""
End-to-end fine-tuning of the grid CNA-GNN, instead of treating it as a
fully frozen feature extractor (which is what every earlier +CNV-GNN*
config in train_embedding_experiment.py did).

Uses the SAME idiom this project's own train_senmo_brca.py already applies
to SeNMo itself (SeNMo_FT.freeze_layers): freeze the early, generically-
pretrained layers and only fine-tune the later ones + task head directly on
the small target cohort. Here: freeze the first GCN layer (gc1) at its
pan-cancer-pretrained weights, and fine-tune the second GCN layer (gc2) +
a small MLP Cox fusion head on [48-dim SeNMo embedding, pooled CNA-GNN
embedding], end-to-end, on each BRCA train fold -- rather than extracting a
once-and-for-all frozen embedding and only ever training a downstream
probe on top of it.

Starts from the RECONSTRUCTION-ONLY grid-GNN checkpoint (cna_gnn_encoder.pt),
not the Cox-multi-task one -- the multi-task pretraining measurably hurt
downstream results (see train_embedding_experiment.py's +CNV-GNN-CoxMTL:
0.7738 vs 0.7813 for the plain reconstruction-pretrained version), so
building on the weaker checkpoint isn't a well-motivated next step.

Same 5-fold split, same standardization scaler (from pretraining, non-BRCA
patients only -- never refit on BRCA), same seed=66 convention as
everywhere else in this project.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "preprocessing"))
from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency, pool_embedding  # noqa: E402
from cox_utils import cox_ptl_loss  # noqa: E402

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
ENCODER_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_gnn_encoder.pt")
DATA_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")

NUM_FOLDS = 5
EPOCHS = 200
LR = 5e-4
WEIGHT_DECAY = 1e-3
SEED = 66


class FusionModel(nn.Module):
    def __init__(self, hidden_dim, senmo_dim=48):
        super().__init__()
        self.encoder = CNAGraphEncoder(hidden_dim=hidden_dim, dropout=0.1)
        fusion_dim = senmo_dim + 2 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, senmo_embedding, cna_x, adj):
        node_emb = self.encoder(cna_x, adj)
        pooled = pool_embedding(node_emb)
        fused = torch.cat([senmo_embedding, pooled], dim=-1)
        return self.head(fused)


def load_pretrained_and_freeze_gc1(model, ckpt):
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in model.encoder.gc1.parameters():
        p.requires_grad = False


def main():
    ckpt = torch.load(ENCODER_PATH, weights_only=False)
    bin_columns = ckpt["bin_columns"]
    hidden_dim = ckpt["hidden_dim"]
    adj = build_normalized_adjacency(ckpt["edge_index"], len(bin_columns))

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    patient_ids = data["patient_ids"]
    senmo_embedding = data["embedding"]
    e_all, t_all, fold_all = data["e"], data["t"], data["fold"]

    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == bin_columns
    X = cna.loc[patient_ids, bin_columns].values.astype(np.float32)
    X = (X - ckpt["scaler_mean"]) / ckpt["scaler_std"]  # pretraining (non-BRCA) scaler, never refit here

    train_cs, test_cs = [], []
    for fold in range(1, NUM_FOLDS + 1):
        torch.manual_seed(SEED)
        test_mask = fold_all == fold
        train_mask = ~test_mask

        model = FusionModel(hidden_dim=hidden_dim)
        load_pretrained_and_freeze_gc1(model, ckpt)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
        )

        senmo_train = torch.tensor(senmo_embedding[train_mask], dtype=torch.float32)
        senmo_test = torch.tensor(senmo_embedding[test_mask], dtype=torch.float32)
        x_train = torch.tensor(X[train_mask]).t().unsqueeze(-1)  # (n_bins, n_train, 1)
        x_test = torch.tensor(X[test_mask]).t().unsqueeze(-1)
        e_train_t = torch.tensor(e_all[train_mask], dtype=torch.float32)
        t_train_t = torch.tensor(t_all[train_mask], dtype=torch.float32)

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            risk_pred = model(senmo_train, x_train, adj)
            loss = cox_ptl_loss(risk_pred, t_train_t, e_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_risk = model(senmo_train, x_train, adj).numpy().reshape(-1)
            test_risk = model(senmo_test, x_test, adj).numpy().reshape(-1)

        train_c = concordance_index(t_all[train_mask], -train_risk, e_all[train_mask])
        test_c = concordance_index(t_all[test_mask], -test_risk, e_all[test_mask])
        train_cs.append(train_c)
        test_cs.append(test_c)
        print(f"[finetune-GNN] fold {fold}: train C-index={train_c:.4f}  test C-index={test_c:.4f} "
              f"(n_train={train_mask.sum()} n_test={test_mask.sum()})")

    print(f"[finetune-GNN] mean train C-index = {np.mean(train_cs):.4f} +/- {np.std(train_cs):.4f}")
    print(f"[finetune-GNN] mean test  C-index = {np.mean(test_cs):.4f} +/- {np.std(test_cs):.4f}")


if __name__ == "__main__":
    main()
