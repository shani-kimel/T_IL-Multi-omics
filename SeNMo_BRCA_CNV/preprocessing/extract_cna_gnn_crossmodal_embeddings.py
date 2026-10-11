"""
Runs the frozen, non-BRCA-pretrained GNN+cross-modal encoder
(pretrain_cna_gnn_crossmodal.py) over the BRCA cohort and writes a pkl with
the same schema as brca_embedding_cna.pkl, plus a new "cna_gnn_crossmodal"
key (Nx32: mean+max pooled 16-dim node embeddings), so it drops straight
into train_embedding_experiment.py.

Same convention as extract_cna_gnn_embeddings.py: only the encoder + its
train-only (non-BRCA) CNA scaler are needed at inference, since the
cross-modal decoder's job was only to shape the encoder during pretraining.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch

from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency, pool_embedding

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
ENCODER_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_gnn_crossmodal_encoder_hd32.pt")
EXISTING_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna_gnn_crossmodal_hd32.pkl")
EMBEDDING_KEY = "cna_gnn_crossmodal_hd32"


def main():
    ckpt = torch.load(ENCODER_PATH, weights_only=False)
    bin_columns = ckpt["bin_columns"]
    adj = build_normalized_adjacency(ckpt["edge_index"], len(bin_columns))

    encoder = CNAGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    patient_ids = existing["patient_ids"]

    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == bin_columns, "CNA csv columns must match the encoder's bin order"
    X = cna.loc[patient_ids, bin_columns].values.astype(np.float32)
    X = (X - ckpt["scaler_mean"]) / ckpt["scaler_std"]

    x_in = torch.tensor(X).t().unsqueeze(-1)  # (n_bins, n_patients, 1)
    with torch.no_grad():
        node_embeddings = encoder(x_in, adj)
        cna_gnn_crossmodal = pool_embedding(node_embeddings).numpy()

    print(f"{EMBEDDING_KEY} embedding shape: {cna_gnn_crossmodal.shape}")
    assert not np.isnan(cna_gnn_crossmodal).any()

    out = dict(existing)
    out[EMBEDDING_KEY] = cna_gnn_crossmodal
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
