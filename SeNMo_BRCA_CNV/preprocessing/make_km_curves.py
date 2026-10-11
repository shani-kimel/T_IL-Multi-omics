"""
Kaplan-Meier risk-stratification plot for the frozen final model: the GNN +
CNV<->Mutation-VAE(HPOEarlyStop) 50/50 ensemble, exactly the recipe that
produced this project's headline result (0.7996 original split, 0.7990 on
an independent fresh split -- see the fresh-split verification).

Risk scores are OUT-OF-FOLD: this project's "model" is a 5-fold retrain
recipe (fit per fold, frozen GNN/VAE embeddings + a small Cox head refit per
fold), not a single persisted checkpoint -- there is no single "the trained
head" to run everyone through. So, as throughout this project, each
patient's risk score comes from whichever fold's head treated them as held-
out test data, pooled into one de-biased risk score per patient. This is
the standard way to get a KM-plottable risk score out of a cross-validated
pipeline without a separate train/deploy split.

Produces two risk groups by median split (High vs Low predicted risk),
log-rank test p-value, and an at-risk table, matching standard survival-
analysis publication format.

Output: SeNMo_BRCA_CNV/results/km_curve_frozen_ensemble.png
        SeNMo_BRCA_CNV/results/km_curve_frozen_ensemble_risk_scores.csv
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
RESULTS_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results")
GNN_PKL = os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl")
ES_PKL = os.path.join(EMBED_DIR, "brca_embedding_cnv_mutation_vae_hpo_earlystop.pkl")
OUT_PNG = os.path.join(RESULTS_DIR, "km_curve_frozen_ensemble.png")
OUT_CSV = os.path.join(RESULTS_DIR, "km_curve_frozen_ensemble_risk_scores.csv")

SEED = 66


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cox_loss(survtime, censor, hazard_pred):
    R_mat = (survtime.reshape(1, -1) >= survtime.reshape(-1, 1)).float()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)


def make_mlp(d):
    return nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))


def train_fold_frozen(X_train, e_train, t_train, X_test, seed=SEED):
    """Exact frozen recipe: fixed head, lr=1e-3, wd=3e-3, 300 fixed epochs."""
    torch.manual_seed(seed)
    model = make_mlp(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-3)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    etr = torch.tensor(e_train, dtype=torch.float32)
    ttr = torch.tensor(t_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    for _ in range(300):
        model.train()
        opt.zero_grad()
        loss = cox_loss(ttr, etr, model(Xtr))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        tr_risk = model(Xtr).numpy().reshape(-1)
        te_risk = model(Xte).numpy().reshape(-1)
    return tr_risk, te_risk


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gnn_data = load(GNN_PKL)
    es_data = load(ES_PKL)
    assert list(gnn_data["patient_ids"]) == list(es_data["patient_ids"])

    patient_ids = gnn_data["patient_ids"]
    senmo_emb = gnn_data["embedding"]
    e_all, t_all, fold_all = gnn_data["e"], gnn_data["t"], gnn_data["fold"]
    n = len(e_all)

    ensemble_risk = np.full(n, np.nan)

    for fold in sorted(set(fold_all)):
        test_mask = fold_all == fold
        train_mask = ~test_mask
        risks = []
        for data, key in [(gnn_data, "cna_gnn"), (es_data, "cnv_mutation_vae_hpo_earlystop")]:
            feat = data[key]
            scaler = StandardScaler()
            f_tr = scaler.fit_transform(feat[train_mask])
            f_te = scaler.transform(feat[test_mask])
            Xtr = np.concatenate([senmo_emb[train_mask], f_tr], axis=1)
            Xte = np.concatenate([senmo_emb[test_mask], f_te], axis=1)
            tr_risk, te_risk = train_fold_frozen(Xtr, e_all[train_mask], t_all[train_mask], Xte)
            risk_scaler = StandardScaler().fit(tr_risk.reshape(-1, 1))
            risks.append(risk_scaler.transform(te_risk.reshape(-1, 1)).reshape(-1))
        ensemble_risk[test_mask] = np.mean(risks, axis=0)

    assert not np.isnan(ensemble_risk).any()

    median_risk = np.median(ensemble_risk)
    high_risk = ensemble_risk >= median_risk
    print(f"Median risk score: {median_risk:.4f}")
    print(f"High-risk group: n={high_risk.sum()}, events={e_all[high_risk].sum():.0f}")
    print(f"Low-risk group:  n={(~high_risk).sum()}, events={e_all[~high_risk].sum():.0f}")

    df = pd.DataFrame({
        "patient_id": patient_ids, "risk_score": ensemble_risk,
        "risk_group": np.where(high_risk, "High", "Low"),
        "OS": e_all, "OS_time": t_all,
    })
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    result = logrank_test(
        t_all[high_risk], t_all[~high_risk],
        event_observed_A=e_all[high_risk], event_observed_B=e_all[~high_risk],
    )
    p_value = result.p_value
    print(f"Log-rank test p-value: {p_value:.2e}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    kmf_high.fit(t_all[high_risk], event_observed=e_all[high_risk], label=f"High risk (n={high_risk.sum()})")
    kmf_low.fit(t_all[~high_risk], event_observed=e_all[~high_risk], label=f"Low risk (n={(~high_risk).sum()})")

    kmf_high.plot_survival_function(ax=ax, color="#c0392b", ci_show=True)
    kmf_low.plot_survival_function(ax=ax, color="#2471a3", ci_show=True)

    p_str = f"p = {p_value:.2e}" if p_value >= 1e-4 else f"p < 1e-4"
    ax.text(0.98, 0.05, f"Log-rank {p_str}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"))

    ax.set_title("BRCA Overall Survival by Predicted Risk Group\n"
                  "(GNN + CNV↔Mutation-VAE ensemble, out-of-fold risk, median split)", fontsize=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)

    add_at_risk_counts(kmf_high, kmf_low, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
