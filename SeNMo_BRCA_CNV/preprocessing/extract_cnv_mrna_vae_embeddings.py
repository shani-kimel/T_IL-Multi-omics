"""
Runs the frozen, non-BRCA-pretrained CNV<->mRNA VAE's CNV encoder over the
BRCA cohort and writes a pkl with the same schema as brca_embedding_cna.pkl,
plus a new "cnv_mrna_vae" key (Nx32: mu_cnv), so it drops straight into
train_embedding_experiment.py as a new extra_configs entry.

Only the CNV encoder is used (mu_cnv, not sampled z) -- mRNA isn't an input
here since the whole point is a CNV-only feature at inference time, shaped
by cross-modal training to encode CNV variation that predicts expression.
Uses the SAME (train-only, non-BRCA) scaler mean/std saved by
pretrain_cnv_mrna_vae.py, so nothing about the BRCA folds leaks into the
encoder or its normalization -- same convention as extract_cna_gnn_embeddings.py.
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
ENCODER_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cnv_mrna_vae_encoder.pt")
EXISTING_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cnv_mrna_vae.pkl")


def main():
    ckpt = torch.load(ENCODER_PATH, weights_only=False)

    encoder = Encoder(ckpt["cna_dim"], ckpt["hidden_dims"], ckpt["latent_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_cnv_state_dict"])
    encoder.eval()

    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    patient_ids = existing["patient_ids"]

    cna = pd.read_csv(CNA_PATH).set_index("ID")
    assert list(cna.columns) == ckpt["cna_columns"], "CNA csv columns must match the encoder's training column order"
    X = cna.loc[patient_ids, ckpt["cna_columns"]].values.astype(np.float32)
    X = (X - ckpt["cna_scaler_mean"]) / ckpt["cna_scaler_std"]

    with torch.no_grad():
        mu_cnv, _ = encoder(torch.tensor(X))
        cnv_mrna_vae = mu_cnv.numpy()

    print(f"cnv_mrna_vae embedding shape: {cnv_mrna_vae.shape}")
    assert not np.isnan(cnv_mrna_vae).any()

    out = dict(existing)
    out["cnv_mrna_vae"] = cnv_mrna_vae
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
