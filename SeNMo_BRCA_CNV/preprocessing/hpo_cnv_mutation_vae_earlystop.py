"""
Stage A Bayesian HPO, v2: same search space and objective as
hpo_cnv_mutation_vae.py, but fixes the horizon-mismatch bug that run
uncovered -- comparing all trials at a fixed 50 epochs picked hyperparameters
that looked good at that specific stopping point but overfit badly when
retrained to the deployment length of 100 epochs (align collapsed to
~0, validation loss rose from epoch 20 onward for the "winning" trial).

Fix: each trial now trains up to MAX_EPOCHS with early stopping on its OWN
validation loss (the same weighted combination as its training loss,
computed on the held-out non-BRCA validation split), and the BEST
checkpoint by validation loss -- not the final epoch -- is what gets
extracted and scored on BRCA. This makes "how long to train" a property of
each trial's own hyperparameters rather than a fixed, externally-imposed
number that happens to match one particular recipe. A trial that wants to
train for 300 epochs can; a trial that overfits by epoch 25 stops there
rather than continuing to a mismatched fixed horizon.

Same fixed architecture (hidden_dims=(256,64), both cross-reconstruction
directions) and same loss-magnitude-informed normalization (align divided
by ALIGN_REF_SCALE before its weight is applied) as the first run -- see
hpo_cnv_mutation_vae.py's docstring for that analysis, unchanged here.

New, separate study (own sqlite db) since the training protocol differs
from the first run -- mixing fixed-epoch and early-stopped trials in one
TPE search would conflate two different experiments.

Outputs:
  SeNMo_BRCA_CNV/data/embedding_experiment/hpo_cnv_mutation_vae_earlystop_study.db
  SeNMo_BRCA_CNV/data/embedding_experiment/cnv_mutation_vae_hpo_earlystop_best_encoder.pt
  SeNMo_BRCA_CNV/data/embedding_experiment/hpo_cnv_mutation_vae_earlystop_trials.csv
"""
import os
import pickle
import sys

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

from cnv_mrna_vae_model import CrossModalVAE
from cox_utils import cox_ptl_loss

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
CNA_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")
MUT_PATH = os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_Mutation_230109_modified.csv")
CLINICAL_PATH = os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
BRCA_PKL = os.path.join(EMBED_DIR, "brca_embedding_cna.pkl")
STUDY_DB = os.path.join(EMBED_DIR, "hpo_cnv_mutation_vae_earlystop_study.db")
BEST_ENCODER_PATH = os.path.join(EMBED_DIR, "cnv_mutation_vae_hpo_earlystop_best_encoder.pt")
TRIALS_CSV = os.path.join(EMBED_DIR, "hpo_cnv_mutation_vae_earlystop_trials.csv")

HIDDEN_DIMS = (256, 64)
CROSS_DIRECTION = "both"
MODALITY_B_DIST = "bernoulli"
BATCH_SIZE = 256
MAX_EPOCHS = 100  # matches the project's actual deployment training length everywhere
                  # else -- no trial can ever train past what it would be deployed at,
                  # which is what the fixed-epoch run's horizon-mismatch bug needed
EVAL_EVERY = 5     # epochs between validation-loss checks
PATIENCE = 4       # consecutive non-improving checks (= 20 epochs) before stopping
VAL_FRACTION = 0.1
SEED = 66
ALIGN_REF_SCALE = 0.05

NUM_FOLDS = 5
MLP_EPOCHS = 300
MLP_LR = 1e-3
MLP_WD = 3e-3

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def load_pretrain_data():
    clinical = pd.read_csv(CLINICAL_PATH).set_index("ID", drop=False)
    cna = pd.read_csv(CNA_PATH).set_index("ID")
    mut = pd.read_csv(MUT_PATH).set_index("ID")
    cna_cols = list(cna.columns)
    mut_cols = list(mut.columns)

    non_brca_ids = sorted(set(clinical[clinical["Cancer_Type"] != "BRCA"]["ID"]) & set(cna.index) & set(mut.index))
    has_survival = clinical.loc[non_brca_ids, "OS"].notna() & clinical.loc[non_brca_ids, "OS.time"].notna()
    print(f"Pretraining pool: {len(non_brca_ids)} non-BRCA patients, {has_survival.sum()} with survival labels")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(non_brca_ids))
    n_val = max(1, int(len(perm) * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ids = [non_brca_ids[i] for i in train_idx]

    cna_all = cna.loc[non_brca_ids, cna_cols].values.astype(np.float32)
    mut_all = mut.loc[non_brca_ids, mut_cols].values.astype(np.float32)
    cna_train_raw = cna.loc[train_ids, cna_cols].values.astype(np.float32)
    mean = cna_train_raw.mean(axis=0)
    std = cna_train_raw.std(axis=0)
    std[std == 0] = 1.0
    cna_std = (cna_all - mean) / std

    e_all = clinical.loc[non_brca_ids, "OS"].fillna(0).astype(np.float32).values
    t_all = clinical.loc[non_brca_ids, "OS.time"].fillna(0).astype(np.float32).values

    train_mask = np.zeros(len(non_brca_ids), dtype=bool)
    train_mask[train_idx] = True
    val_mask = ~train_mask

    return {
        "cna_cols": cna_cols, "cna_std": cna_std, "mut_all": mut_all,
        "e_all": e_all, "t_all": t_all, "has_survival_all": has_survival.values,
        "train_mask": train_mask, "val_mask": val_mask, "cna_mean": mean, "cna_std_": std,
    }


def cox_mlp_loss(survtime, censor, hazard_pred):
    R_mat = (survtime.reshape(1, -1) >= survtime.reshape(-1, 1)).float()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)


def make_mlp(d):
    return nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))


def eval_embedding_on_brca(embedding, brca):
    senmo_emb = brca["embedding"]
    e_all, t_all, fold_all = brca["e"], brca["t"], brca["fold"]
    test_cs = []
    for fold in range(1, NUM_FOLDS + 1):
        test_mask = fold_all == fold
        train_mask = ~test_mask
        scaler = StandardScaler()
        f_tr = scaler.fit_transform(embedding[train_mask])
        f_te = scaler.transform(embedding[test_mask])
        Xtr = np.concatenate([senmo_emb[train_mask], f_tr], axis=1)
        Xte = np.concatenate([senmo_emb[test_mask], f_te], axis=1)

        torch.manual_seed(SEED)
        model = make_mlp(Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        etr_t = torch.tensor(e_all[train_mask], dtype=torch.float32)
        ttr_t = torch.tensor(t_all[train_mask], dtype=torch.float32)
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        for _ in range(MLP_EPOCHS):
            model.train()
            opt.zero_grad()
            loss = cox_mlp_loss(ttr_t, etr_t, model(Xtr_t))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            te_risk = model(Xte_t).numpy().reshape(-1)
        test_cs.append(concordance_index(t_all[test_mask], -te_risk, e_all[test_mask]))
    return float(np.mean(test_cs)), float(np.std(test_cs))


def extract_cnv_embedding(encoder, cna_full, patient_ids, cna_cols, cna_mean, cna_std_):
    X = cna_full.loc[patient_ids, cna_cols].values.astype(np.float32)
    X = (X - cna_mean) / cna_std_
    encoder.eval()
    with torch.no_grad():
        mu, _ = encoder(torch.tensor(X))
    return mu.numpy()


def main():
    data = load_pretrain_data()
    with open(BRCA_PKL, "rb") as f:
        brca = pickle.load(f)
    cna_full = pd.read_csv(CNA_PATH).set_index("ID")

    cna_t = torch.tensor(data["cna_std"])
    mut_t = torch.tensor(data["mut_all"])
    e_t = torch.tensor(data["e_all"])
    t_t = torch.tensor(data["t_all"])
    surv_mask = torch.tensor(data["has_survival_all"])
    train_idx_all = np.where(data["train_mask"])[0]
    val_idx_all = np.where(data["val_mask"])[0]

    best_value_so_far = {"value": -1.0}

    def weighted_loss(parts, cox_loss, recon_weight, align_weight, kl_weight, cox_weight):
        recon_total = parts["recon_self"] + parts["recon_cross"]
        align_norm = parts["align"] / ALIGN_REF_SCALE
        return (recon_weight * recon_total + align_weight * align_norm
                + kl_weight * parts["kl"] + cox_weight * cox_loss)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
        cox_weight = trial.suggest_float("cox_weight", 0.1, 2.0)
        recon_weight = trial.suggest_float("recon_weight", 0.1, 2.0)
        align_weight = trial.suggest_float("align_weight", 0.1, 2.0)
        kl_weight = trial.suggest_float("kl_weight", 0.001, 0.1, log=True)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = CrossModalVAE(cnv_dim=data["cna_std"].shape[1], mrna_dim=data["mut_all"].shape[1],
                               latent_dim=latent_dim, hidden_dims=HIDDEN_DIMS, dropout=dropout)
        cox_head = nn.Linear(latent_dim, 1)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(cox_head.parameters()),
                                      lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_encoder_state = None
        best_epoch = 0
        bad_checks = 0
        stopped_epoch = MAX_EPOCHS

        for epoch in range(1, MAX_EPOCHS + 1):
            perm_e = np.random.permutation(train_idx_all)
            for start in range(0, len(perm_e), BATCH_SIZE):
                idx = perm_e[start:start + BATCH_SIZE]
                if len(idx) < 8:
                    continue
                cnv_b, mut_b = cna_t[idx], mut_t[idx]
                parts = model.forward_losses(cnv_b, mut_b, modality_b_dist=MODALITY_B_DIST,
                                              cross_direction=CROSS_DIRECTION)
                mask_b = surv_mask[idx]
                if mask_b.sum() >= 8:
                    mu_cnv, _ = model.encoder_cnv(cnv_b)
                    risk_pred = cox_head(mu_cnv[mask_b])
                    cox_loss = cox_ptl_loss(risk_pred, t_t[idx][mask_b], e_t[idx][mask_b])
                else:
                    cox_loss = torch.tensor(0.0)

                loss = weighted_loss(parts, cox_loss, recon_weight, align_weight, kl_weight, cox_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % EVAL_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    val_parts = model.forward_losses(cna_t[val_idx_all], mut_t[val_idx_all],
                                                       modality_b_dist=MODALITY_B_DIST,
                                                       cross_direction=CROSS_DIRECTION)
                    val_mask_surv = surv_mask[val_idx_all]
                    if val_mask_surv.sum() >= 8:
                        mu_cnv_val, _ = model.encoder_cnv(cna_t[val_idx_all])
                        val_risk = cox_head(mu_cnv_val[val_mask_surv])
                        val_cox = cox_ptl_loss(val_risk, t_t[val_idx_all][val_mask_surv],
                                                e_t[val_idx_all][val_mask_surv])
                    else:
                        val_cox = torch.tensor(0.0)
                    val_loss = weighted_loss(val_parts, val_cox, recon_weight, align_weight,
                                              kl_weight, cox_weight).item()

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_encoder_state = {k: v.clone() for k, v in model.encoder_cnv.state_dict().items()}
                    best_epoch = epoch
                    bad_checks = 0
                else:
                    bad_checks += 1
                    if bad_checks >= PATIENCE:
                        stopped_epoch = epoch
                        break

        model.encoder_cnv.load_state_dict(best_encoder_state)
        embedding = extract_cnv_embedding(model.encoder_cnv, cna_full, brca["patient_ids"],
                                           data["cna_cols"], data["cna_mean"], data["cna_std_"])
        mean_c, std_c = eval_embedding_on_brca(embedding, brca)
        trial.set_user_attr("std_c", std_c)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("stopped_epoch", stopped_epoch)

        if mean_c > best_value_so_far["value"]:
            best_value_so_far["value"] = mean_c
            torch.save({
                "encoder_cnv_state_dict": best_encoder_state,
                "cna_dim": data["cna_std"].shape[1], "latent_dim": latent_dim,
                "hidden_dims": HIDDEN_DIMS, "dropout": dropout,
                "cna_columns": data["cna_cols"],
                "cna_scaler_mean": data["cna_mean"], "cna_scaler_std": data["cna_std_"],
                "trial_params": trial.params, "best_epoch": best_epoch,
            }, BEST_ENCODER_PATH)
            print(f"[trial {trial.number}] NEW BEST mean_c={mean_c:.4f} std={std_c:.4f} "
                  f"best_epoch={best_epoch} stopped_epoch={stopped_epoch} params={trial.params}", flush=True)
        else:
            print(f"[trial {trial.number}] mean_c={mean_c:.4f} std={std_c:.4f} "
                  f"best_epoch={best_epoch} stopped_epoch={stopped_epoch} params={trial.params}", flush=True)

        return mean_c

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{STUDY_DB}",
        study_name="cnv_mutation_vae_hpo_earlystop",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== BEST TRIAL ===")
    print(f"value={study.best_value:.4f}")
    print(f"params={study.best_params}")
    print(f"best_epoch={study.best_trial.user_attrs.get('best_epoch')}")

    study.trials_dataframe().to_csv(TRIALS_CSV, index=False)
    print(f"Wrote {TRIALS_CSV}")
    print(f"Best encoder checkpoint: {BEST_ENCODER_PATH}")


if __name__ == "__main__":
    main()
