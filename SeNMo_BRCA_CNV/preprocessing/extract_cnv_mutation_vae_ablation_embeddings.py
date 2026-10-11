"""
Extracts the frozen CNV encoder from each of the three cross-direction
ablation variants (pretrain_cnv_mutation_vae_ablation.py) over the BRCA
cohort. Same convention as extract_cnv_mutation_vae_embeddings.py, run once
per variant.
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
EXISTING_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")

VARIANTS = ["a", "b", "c"]


def main():
    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    patient_ids = existing["patient_ids"]
    cna = pd.read_csv(CNA_PATH).set_index("ID")

    for name in VARIANTS:
        ckpt_path = os.path.join(OUT_DIR, f"cnv_mutation_vae_ablation_{name}_encoder.pt")
        ckpt = torch.load(ckpt_path, weights_only=False)

        encoder = Encoder(ckpt["cna_dim"], ckpt["hidden_dims"], ckpt["latent_dim"], dropout=0.0)
        encoder.load_state_dict(ckpt["encoder_cnv_state_dict"])
        encoder.eval()

        assert list(cna.columns) == ckpt["cna_columns"]
        X = cna.loc[patient_ids, ckpt["cna_columns"]].values.astype(np.float32)
        X = (X - ckpt["cna_scaler_mean"]) / ckpt["cna_scaler_std"]

        with torch.no_grad():
            mu_cnv, _ = encoder(torch.tensor(X))
            embedding = mu_cnv.numpy()

        key = f"cnv_mutation_vae_ablation_{name}"
        print(f"{key} (cross_direction={ckpt['cross_direction']}) embedding shape: {embedding.shape}")
        assert not np.isnan(embedding).any()

        out = dict(existing)
        out[key] = embedding
        out_path = os.path.join(OUT_DIR, f"brca_embedding_{key}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=4)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
