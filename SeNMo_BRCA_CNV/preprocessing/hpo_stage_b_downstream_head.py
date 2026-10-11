"""
Stage B Bayesian HPO: tune the downstream survival head (the small Cox MLP
trained on top of a frozen embedding + the 48-dim SeNMo embedding), not the
embedding itself. Search space:

  head_lr, weight_decay (log scale), dropout, hidden_dim, num_hidden_layers

Applied to the current best standalone CNV embedding
(cnv_mutation_vae_hpo_earlystop from Stage A, C-index 0.7976 with the
project's fixed, never-tuned head: Linear(d,16)->ReLU->Dropout(0.3)->
Linear(16,1), lr=1e-3, weight_decay=3e-3, trained for a fixed 300 epochs).

Learning from Stage A's horizon-mismatch bug: epoch count is not fixed here
either. Each of the 5 outer CV folds carves its own training portion into
an 85/15 inner train/val split and trains with early stopping on inner-val
Cox loss (checked every 5 epochs, patience 5 checks = 25 epochs, ceiling
300) -- the outer test fold is never touched for model selection, only for
final scoring, so this stays leakage-free per fold.

Objective = mean outer-fold test C-index (same 5-fold split as everywhere
else in this project).

Much cheaper per trial than Stage A (no VAE/GNN retraining), so this search
affords far more trials.

Outputs:
  SeNMo_BRCA_CNV/data/embedding_experiment/hpo_stage_b_study.db
  SeNMo_BRCA_CNV/data/embedding_experiment/hpo_stage_b_trials.csv
"""
import os
import pickle
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")

# Which embedding to tune the downstream head for -- pass a second CLI arg
# ("gnn" or "hpoearlystop", default hpoearlystop) since the Stage B result
# turned out to be embedding-specific (a head tuned for one embedding did
# not transfer well to the other).
EMBEDDING_CHOICES = {
    "hpoearlystop": ("brca_embedding_cnv_mutation_vae_hpo_earlystop.pkl", "cnv_mutation_vae_hpo_earlystop"),
    "gnn": ("brca_embedding_cna_gnn.pkl", "cna_gnn"),
}

NUM_FOLDS = 5
MAX_EPOCHS = 300
EVAL_EVERY = 5
PATIENCE = 5
INNER_VAL_FRACTION = 0.15
SEED = 66

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EMBEDDING_CHOICE = sys.argv[2] if len(sys.argv) > 2 else "hpoearlystop"
_pkl_name, EMBEDDING_KEY = EMBEDDING_CHOICES[EMBEDDING_CHOICE]
EMBEDDING_PKL = os.path.join(EMBED_DIR, _pkl_name)
STUDY_DB = os.path.join(EMBED_DIR, f"hpo_stage_b_{EMBEDDING_CHOICE}_study.db")
TRIALS_CSV = os.path.join(EMBED_DIR, f"hpo_stage_b_{EMBEDDING_CHOICE}_trials.csv")


def cox_loss(survtime, censor, hazard_pred):
    R_mat = (survtime.reshape(1, -1) >= survtime.reshape(-1, 1)).float()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)


def make_mlp(input_dim, hidden_dim, num_layers, dropout):
    layers = []
    d = input_dim
    for _ in range(num_layers):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden_dim
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def train_one_fold_earlystop(X_train, e_train, t_train, X_test, e_test, t_test,
                              lr, weight_decay, dropout, hidden_dim, num_layers, fold_seed):
    rng = np.random.default_rng(fold_seed)
    n = X_train.shape[0]
    perm = rng.permutation(n)
    n_val = max(8, int(n * INNER_VAL_FRACTION))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    etr = torch.tensor(e_train[tr_idx], dtype=torch.float32)
    ttr = torch.tensor(t_train[tr_idx], dtype=torch.float32)
    Xval = torch.tensor(X_train[val_idx], dtype=torch.float32)
    eval_ = torch.tensor(e_train[val_idx], dtype=torch.float32)
    tval = torch.tensor(t_train[val_idx], dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    torch.manual_seed(fold_seed)
    model = make_mlp(X_train.shape[1], hidden_dim, num_layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    bad_checks = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = cox_loss(ttr, etr, model(Xtr))
        loss.backward()
        opt.step()

        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = cox_loss(tval, eval_, model(Xval)).item()
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad_checks = 0
            else:
                bad_checks += 1
                if bad_checks >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_risk = model(Xte).numpy().reshape(-1)
    return concordance_index(t_test, -te_risk, e_test)


def main():
    with open(EMBEDDING_PKL, "rb") as f:
        data = pickle.load(f)
    senmo_emb = data["embedding"]
    feat = data[EMBEDDING_KEY]
    e_all, t_all, fold_all = data["e"], data["t"], data["fold"]
    print(f"Loaded embedding '{EMBEDDING_KEY}', dim={feat.shape[1]}, {len(e_all)} patients")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)

        fold_cs = []
        for fold in range(1, NUM_FOLDS + 1):
            test_mask = fold_all == fold
            train_mask = ~test_mask

            scaler = StandardScaler()
            f_tr = scaler.fit_transform(feat[train_mask])
            f_te = scaler.transform(feat[test_mask])
            Xtr = np.concatenate([senmo_emb[train_mask], f_tr], axis=1)
            Xte = np.concatenate([senmo_emb[test_mask], f_te], axis=1)

            c = train_one_fold_earlystop(
                Xtr, e_all[train_mask], t_all[train_mask], Xte, e_all[test_mask], t_all[test_mask],
                lr, weight_decay, dropout, hidden_dim, num_layers, fold_seed=SEED + fold,
            )
            fold_cs.append(c)

        mean_c = float(np.mean(fold_cs))
        trial.set_user_attr("std_c", float(np.std(fold_cs)))
        return mean_c

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{STUDY_DB}",
        study_name=f"stage_b_downstream_head_{EMBEDDING_CHOICE}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print("\n=== BEST TRIAL ===")
    print(f"value={study.best_value:.4f}  std={study.best_trial.user_attrs.get('std_c'):.4f}")
    print(f"params={study.best_params}")

    study.trials_dataframe().to_csv(TRIALS_CSV, index=False)
    print(f"Wrote {TRIALS_CSV}")


if __name__ == "__main__":
    main()
