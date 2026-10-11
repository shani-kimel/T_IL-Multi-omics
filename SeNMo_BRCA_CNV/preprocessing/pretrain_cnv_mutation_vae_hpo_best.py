"""
Final full-length (100 epoch) training run using the best hyperparameters
found by hpo_cnv_mutation_vae.py's 25-trial Bayesian (TPE) search, which
used only 50 epochs/trial for search-time budget:

  lr=3.721e-4, weight_decay=1.278e-4, dropout=0.317, latent_dim=32,
  cox_weight=1.566, recon_weight=1.569, align_weight=0.885, kl_weight=0.00142

Same loss-combination scheme as the HPO search: recon_weight scales
(recon_self + recon_cross) combined; align_weight applies to the alignment
term normalized by ALIGN_REF_SCALE (0.05), since its raw magnitude is
~100x smaller than the other terms (see hpo_cnv_mutation_vae.py's docstring
for the magnitude inspection this is based on) -- NOT the same weighting
convention as the original pretrain_cnv_mutation_vae.py, so this is not a
drop-in replacement for that script, it's the HPO-tuned recipe in its own
right. Same non-BRCA pool, both cross-reconstruction directions, Cox
multi-task term, as every other variant in this project.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/cnv_mutation_vae_hpo_best_full_encoder.pt
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
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment",
                         "cnv_mutation_vae_hpo_best_full_encoder.pt")

HIDDEN_DIMS = (256, 64)
MODALITY_B_DIST = "bernoulli"
CROSS_DIRECTION = "both"
BATCH_SIZE = 256
EPOCHS = 100
VAL_FRACTION = 0.1
SEED = 66
ALIGN_REF_SCALE = 0.05

# Best trial (21) from hpo_cnv_mutation_vae.py's 25-trial study.
LR = 0.00037206072730579194
WEIGHT_DECAY = 0.00012783821371084388
DROPOUT = 0.3167339733159142
LATENT_DIM = 32
COX_WEIGHT = 1.5656646193455437
RECON_WEIGHT = 1.5686777190102341
ALIGN_WEIGHT = 0.8850865150199444
KL_WEIGHT = 0.0014160625190000272


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
    optimizer = torch.optim.Adam(list(model.parameters()) + list(cox_head.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    def run_batch(idx, train):
        model.train(train)
        cox_head.train(train)
        cnv_b, mut_b = cna_t[idx], mut_t[idx]
        parts = model.forward_losses(cnv_b, mut_b, modality_b_dist=MODALITY_B_DIST, cross_direction=CROSS_DIRECTION)
        recon_total = parts["recon_self"] + parts["recon_cross"]
        align_norm = parts["align"] / ALIGN_REF_SCALE

        mask_b = surv_mask[idx]
        if mask_b.sum() >= 8:
            with torch.set_grad_enabled(train):
                mu_cnv, _ = model.encoder_cnv(cnv_b)
            risk_pred = cox_head(mu_cnv[mask_b])
            cox_loss = cox_ptl_loss(risk_pred, t_t[idx][mask_b], e_t[idx][mask_b])
        else:
            cox_loss = torch.tensor(0.0)

        loss = RECON_WEIGHT * recon_total + ALIGN_WEIGHT * align_norm + KL_WEIGHT * parts["kl"] + COX_WEIGHT * cox_loss
        return loss, {
            "recon_self": parts["recon_self"].item(), "recon_cross": parts["recon_cross"].item(),
            "align": parts["align"].item(), "kl": parts["kl"].item(),
            "cox": cox_loss.item() if torch.is_tensor(cox_loss) else cox_loss,
        }

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
