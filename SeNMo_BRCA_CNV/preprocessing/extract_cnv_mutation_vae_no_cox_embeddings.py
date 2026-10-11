"""
Extracts the frozen CNV encoder from the no-Cox ablation
(pretrain_cnv_mutation_vae_no_cox.py) over the BRCA cohort.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch

from cnv_mrna_vae_model import Encoder

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
ENCODER_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cnv_mutation_vae_no_cox_encoder.pt")
EXISTING_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cnv_mutation_vae_no_cox.pkl")
EMBEDDING_KEY = "cnv_mutation_vae_no_cox"


def main():
    ckpt = torch.load(ENCODER_PATH, weights_only=False)

    encoder = Encoder(ckpt["cna_dim"], ckpt["hidden_dims"], ckpt["latent_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_cnv_state_dict"])
    encoder.eval()

    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    patient_ids = existing["patient_ids"]

    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == ckpt["cna_columns"]
    X = cna.loc[patient_ids, ckpt["cna_columns"]].values.astype(np.float32)
    X = (X - ckpt["cna_scaler_mean"]) / ckpt["cna_scaler_std"]

    with torch.no_grad():
        mu_cnv, _ = encoder(torch.tensor(X))
        embedding = mu_cnv.numpy()

    print(f"{EMBEDDING_KEY} embedding shape: {embedding.shape}")
    assert not np.isnan(embedding).any()

    out = dict(existing)
    out[EMBEDDING_KEY] = embedding
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
