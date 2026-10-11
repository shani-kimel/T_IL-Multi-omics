"""
Follow-up to the cross-direction ablation: direction B (Mutation -> CNV
only) was the best of the three align-off variants, and the alignment term
turned out to matter more than direction choice. This combines both --
B direction only, alignment back on (align_weight=0.5, matching the
original recipe) -- to see whether dropping direction A specifically (which
the ablation suggested was net-negative for the extracted CNV encoder) beats
the original "both directions + align" recipe (0.7938 MLPhead).

Everything else identical to pretrain_cnv_mutation_vae_ablation.py's config B
and to the original pretrain_cnv_mutation_vae.py: same non-BRCA pool, same
architecture/hyperparameters, same Cox term.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cnv_mutation_vae_b_align_encoder.pt
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cnv_mrna_vae_model import CrossModalVAE
from cox_utils import cox_ptl_loss

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
MUT_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_Mutation_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cnv_mutation_vae_b_align_encoder.pt")

CROSS_DIRECTION = "b_to_a"
LATENT_DIM = 32
HIDDEN_DIMS = (256, 64)
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
CROSS_WEIGHT = 0.5
ALIGN_WEIGHT = 0.5  # back on, matching the original recipe
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
    mut = pd.read_csv(MUT_PATH).set_index("ID")
    cna_cols = list(cna.columns)
    mut_cols = list(mut.columns)

    non_brca_ids = sorted(set(clinical[clinical["Cancer_Type"] != "BRCA"]["ID"]) & set(cna.index) & set(mut.index))
    print(f"Pretraining pool (non-BRCA, has CNA + mutation): {len(non_brca_ids)} patients")
    has_survival = clinical.loc[non_brca_ids, "OS"].notna() & clinical.loc[non_brca_ids, "OS.time"].notna()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ids = [non_brca_ids[i] for i in train_idx]

    cna_all = cna.loc[non_brca_ids, cna_cols].values.astype(np.float32)
    mut_all = mut.loc[non_brca_ids, mut_cols].values.astype(np.float32)
    cna_train_raw = cna.loc[train_ids, cna_cols].values.astype(np.float32)
    cna_std, cna_mean_, cna_std_ = standardize(cna_train_raw, cna_all)

    e_all = clinical.loc[non_brca_ids, "OS"].fillna(0).astype(np.float32).values
    t_all = clinical.loc[non_brca_ids, "OS.time"].fillna(0).astype(np.float32).values
    has_survival_all = has_survival.values

    cna_t = torch.tensor(cna_std)
    mut_t = torch.tensor(mut_all)
    e_t = torch.tensor(e_all)
    t_t = torch.tensor(t_all)
    surv_mask = torch.tensor(has_survival_all)

    train_mask = np.zeros(len(non_brca_ids), dtype=bool)
    train_mask[train_idx] = True
    val_mask = ~train_mask
    print(f"train={train_mask.sum()} val={val_mask.sum()}  cna_dim={cna_std.shape[1]} mut_dim={mut_all.shape[1]}")

    model = CrossModalVAE(cnv_dim=cna_std.shape[1], mrna_dim=mut_all.shape[1],
                           latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    cox_head = nn.Linear(LATENT_DIM, 1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cox_head.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )

    def run_batch(idx, train):
        model.train(train)
        cox_head.train(train)
        cnv_b, mut_b = cna_t[idx], mut_t[idx]
        vae_loss, parts = model.compute_loss(cnv_b, mut_b, CROSS_WEIGHT, ALIGN_WEIGHT, KL_WEIGHT,
                                              modality_b_dist="bernoulli", cross_direction=CROSS_DIRECTION)
        mask_b = surv_mask[idx]
        if mask_b.sum() >= 8:
            with torch.set_grad_enabled(train):
                mu_cnv, _ = model.encoder_cnv(cnv_b)
            risk_pred = cox_head(mu_cnv[mask_b])
            cox_loss = cox_ptl_loss(risk_pred, t_t[idx][mask_b], e_t[idx][mask_b])
        else:
            cox_loss = torch.tensor(0.0)
        loss = vae_loss + LAMBDA_COX * cox_loss
        parts["cox"] = cox_loss.item() if torch.is_tensor(cox_loss) else cox_loss
        return loss, parts

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
            print(f"epoch {epoch:3d}  train[loss {sums['loss']:.4f} recon_self {sums['recon_self']:.4f} "
                  f"recon_cross {sums['recon_cross']:.4f} align {sums['align']:.4f} cox {sums['cox']:.4f}]  "
                  f"val[loss {val_loss.item():.4f} recon_self {val_parts['recon_self']:.4f} cox {val_parts['cox']:.4f}]")

    torch.save(
        {
            "encoder_cnv_state_dict": model.encoder_cnv.state_dict(),
            "cna_dim": cna_std.shape[1], "expr_dim": mut_all.shape[1],
            "latent_dim": LATENT_DIM, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT,
            "cna_columns": cna_cols,
            "cna_scaler_mean": cna_mean_, "cna_scaler_std": cna_std_,
        },
        OUT_PATH,
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
