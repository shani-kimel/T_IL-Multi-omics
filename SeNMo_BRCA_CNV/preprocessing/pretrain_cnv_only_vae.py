"""
Control for the CNV<->Mutation cross-modal VAE: the exact same encoder/decoder
architecture (Encoder/Decoder from cnv_mrna_vae_model.py, latent_dim=32,
hidden_dims=(256,64)) and the exact same auxiliary Cox multi-task term on the
latent, trained on the same non-BRCA cohort with the same hyperparameters --
but with NO mutation, no second encoder, no cross-reconstruction, no
alignment. Just CNV self-reconstruction + KL + Cox.

This isolates: how much of the CNV<->Mutation VAE's downstream gain comes
from actual cross-modal information (mutation shaping what CNV variation is
kept) versus simply having a good nonlinear CNV compressor pretrained
jointly with a survival objective on 7,417 pan-cancer labels? If this
control matches or beats +CNV-MutationVAE-MLPhead (0.7938), the cross-modal
framing was doing little of the real work.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cnv_only_vae_encoder.pt
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnv_mrna_vae_model import Encoder, Decoder, reparameterize, kl_divergence
from cox_utils import cox_ptl_loss

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cnv_only_vae_encoder.pt")

LATENT_DIM = 32
HIDDEN_DIMS = (256, 64)
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
KL_WEIGHT = 0.01
LAMBDA_COX = 1.0
VAL_FRACTION = 0.1
SEED = 66


def standardize(X_train, X):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    clinical = pd.read_csv(CLINICAL_PATH).set_index("ID", drop=False)
    cna = pd.read_csv(CNA_PATH).set_index("ID")
    cna_cols = list(cna.columns)

    # Same pool definition as pretrain_cnv_mutation_vae.py, minus the mutation
    # file requirement (irrelevant here) -- kept identical to the non-BRCA
    # CNA+mutation intersection anyway so the comparison is apples-to-apples
    # on the exact same patients.
    mut = pd.read_csv(os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_Mutation_230109_modified.csv")).set_index("ID")
    non_brca_ids = sorted(set(clinical[clinical["Cancer_Type"] != "BRCA"]["ID"]) & set(cna.index) & set(mut.index))
    print(f"Pretraining pool (non-BRCA, same patient set as the mutation VAE): {len(non_brca_ids)} patients")
    has_survival = clinical.loc[non_brca_ids, "OS"].notna() & clinical.loc[non_brca_ids, "OS.time"].notna()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ids = [non_brca_ids[i] for i in train_idx]

    cna_all = cna.loc[non_brca_ids, cna_cols].values.astype(np.float32)
    cna_train_raw = cna.loc[train_ids, cna_cols].values.astype(np.float32)
    cna_std, cna_mean_, cna_std_ = standardize(cna_train_raw, cna_all)

    e_all = clinical.loc[non_brca_ids, "OS"].fillna(0).astype(np.float32).values
    t_all = clinical.loc[non_brca_ids, "OS.time"].fillna(0).astype(np.float32).values
    has_survival_all = has_survival.values

    cna_t = torch.tensor(cna_std)
    e_t = torch.tensor(e_all)
    t_t = torch.tensor(t_all)
    surv_mask = torch.tensor(has_survival_all)

    train_mask = np.zeros(len(non_brca_ids), dtype=bool)
    train_mask[train_idx] = True
    val_mask = ~train_mask
    print(f"train={train_mask.sum()} val={val_mask.sum()}  cna_dim={cna_std.shape[1]}")

    encoder = Encoder(cna_std.shape[1], HIDDEN_DIMS, LATENT_DIM, dropout=DROPOUT)
    decoder = Decoder(LATENT_DIM, cna_std.shape[1], tuple(reversed(HIDDEN_DIMS)), dropout=DROPOUT)
    cox_head = nn.Linear(LATENT_DIM, 1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(cox_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    def run_batch(idx, train):
        encoder.train(train)
        decoder.train(train)
        cox_head.train(train)
        cnv_b = cna_t[idx]
        mu, logvar = encoder(cnv_b)
        z = reparameterize(mu, logvar)
        recon = decoder(z)
        recon_loss = F.mse_loss(recon, cnv_b)
        kl = kl_divergence(mu, logvar)

        mask_b = surv_mask[idx]
        if mask_b.sum() >= 8:
            risk_pred = cox_head(mu[mask_b])
            cox_loss = cox_ptl_loss(risk_pred, t_t[idx][mask_b], e_t[idx][mask_b])
        else:
            cox_loss = torch.tensor(0.0)

        loss = recon_loss + KL_WEIGHT * kl + LAMBDA_COX * cox_loss
        return loss, {"recon": recon_loss.item(), "kl": kl.item(),
                       "cox": cox_loss.item() if torch.is_tensor(cox_loss) else cox_loss}

    train_idx_all = np.where(train_mask)[0]
    val_idx_all = np.where(val_mask)[0]

    for epoch in range(1, EPOCHS + 1):
        perm_e = np.random.permutation(train_idx_all)
        sums = {}
        n_batches = 0
        for start in range(0, len(perm_e), BATCH_SIZE):
            idx = perm_e[start:start + BATCH_SIZE]
            if len(idx) < 8:
                continue
            optimizer.zero_grad()
            loss, parts = run_batch(idx, train=True)
            loss.backward()
            optimizer.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            n_batches += 1
        for k in sums:
            sums[k] /= n_batches

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                val_loss, val_parts = run_batch(val_idx_all, train=False)
            print(f"epoch {epoch:3d}  train[loss {sums['loss']:.4f} recon {sums['recon']:.4f} cox {sums['cox']:.4f}]  "
                  f"val[loss {val_loss.item():.4f} recon {val_parts['recon']:.4f} cox {val_parts['cox']:.4f}]")

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "cna_dim": cna_std.shape[1], "latent_dim": LATENT_DIM,
            "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
            "cna_columns": cna_cols,
            "cna_scaler_mean": cna_mean_, "cna_scaler_std": cna_std_,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
