"""
Extended KM comparison across the model progression: SeNMo alone (no-CNV
baseline) -> SeNMo+GNN -> SeNMo+CNV-VAE(HPOEarlyStop) -> SeNMo+CNV ensemble
(the frozen final model from make_km_curves.py). For each model:

  - tertile risk stratification (Low/Mid/High, not just median split)
  - multivariate log-rank test across all 3 groups
  - hazard ratio (High vs Low tertile) with 95% CI via Cox PH, so the
    strength of stratification is quantified, not just visualized
  - its own KM panel, assembled into one side-by-side comparison figure

Same out-of-fold risk methodology as make_km_curves.py: every patient's
risk score comes from whichever of the 5 folds' models treated them as
held-out test data, using the exact frozen recipe (fixed head, lr=1e-3,
wd=3e-3, 300 epochs, seed=66) -- no re-tuning, consistent with the
project's validated final protocol.

Output: SeNMo_BRCA_CNV/results/km_comparison_senmo_vs_cnv.png
        SeNMo_BRCA_CNV/results/km_comparison_summary.csv
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
RESULTS_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results")
GNN_PKL = os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl")
ES_PKL = os.path.join(EMBED_DIR, "brca_embedding_cnv_mutation_vae_hpo_earlystop.pkl")
OUT_PNG = os.path.join(RESULTS_DIR, "km_comparison_senmo_vs_cnv.png")
OUT_CSV = os.path.join(RESULTS_DIR, "km_comparison_summary.csv")

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


def compute_oof_risk(senmo_emb, e_all, t_all, fold_all, feature_blocks):
    """feature_blocks: list of (data_dict, key) pairs to concat+standardize
    alongside senmo_emb; empty list = SeNMo alone, no CNV at all. Multiple
    blocks are averaged as a prediction-level ensemble (standardized risk),
    matching the project's established ensembling convention."""
    n = len(e_all)
    if not feature_blocks:
        risk = np.full(n, np.nan)
        for fold in sorted(set(fold_all)):
            test_mask = fold_all == fold
            train_mask = ~test_mask
            Xtr = senmo_emb[train_mask]
            Xte = senmo_emb[test_mask]
            tr_risk, te_risk = train_fold_frozen(Xtr, e_all[train_mask], t_all[train_mask], Xte)
            rs = StandardScaler().fit(tr_risk.reshape(-1, 1))
            risk[test_mask] = rs.transform(te_risk.reshape(-1, 1)).reshape(-1)
        return risk

    risk = np.full(n, np.nan)
    for fold in sorted(set(fold_all)):
        test_mask = fold_all == fold
        train_mask = ~test_mask
        block_risks = []
        for data, key in feature_blocks:
            feat = data[key]
            scaler = StandardScaler()
            f_tr = scaler.fit_transform(feat[train_mask])
            f_te = scaler.transform(feat[test_mask])
            Xtr = np.concatenate([senmo_emb[train_mask], f_tr], axis=1)
            Xte = np.concatenate([senmo_emb[test_mask], f_te], axis=1)
            tr_risk, te_risk = train_fold_frozen(Xtr, e_all[train_mask], t_all[train_mask], Xte)
            rs = StandardScaler().fit(tr_risk.reshape(-1, 1))
            block_risks.append(rs.transform(te_risk.reshape(-1, 1)).reshape(-1))
        risk[test_mask] = np.mean(block_risks, axis=0)
    return risk


def tertile_groups(risk):
    q1, q2 = np.percentile(risk, [100 / 3, 200 / 3])
    groups = np.where(risk <= q1, "Low", np.where(risk <= q2, "Mid", "High"))
    return groups


def hazard_ratio_high_vs_low(t_all, e_all, groups):
    mask = groups != "Mid"
    df = pd.DataFrame({
        "T": t_all[mask], "E": e_all[mask],
        "high": (groups[mask] == "High").astype(int),
    })
    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E")
    hr = np.exp(cph.params_["high"])
    ci_low, ci_high = np.exp(cph.confidence_intervals_.loc["high"])
    p = cph.summary.loc["high", "p"]
    return hr, ci_low, ci_high, p


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gnn_data = load(GNN_PKL)
    es_data = load(ES_PKL)
    assert list(gnn_data["patient_ids"]) == list(es_data["patient_ids"])

    senmo_emb = gnn_data["embedding"]
    e_all, t_all, fold_all = gnn_data["e"], gnn_data["t"], gnn_data["fold"]

    models = [
        ("SeNMo alone (no CNV)", []),
        ("SeNMo + GNN", [(gnn_data, "cna_gnn")]),
        ("SeNMo + CNV-VAE", [(es_data, "cnv_mutation_vae_hpo_earlystop")]),
        ("SeNMo + CNV ensemble (final)", [(gnn_data, "cna_gnn"), (es_data, "cnv_mutation_vae_hpo_earlystop")]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    summary_rows = []
    colors = {"Low": "#2471a3", "Mid": "#7d8c99", "High": "#c0392b"}

    for ax, (label, blocks) in zip(axes, models):
        print(f"\n=== {label} ===")
        risk = compute_oof_risk(senmo_emb, e_all, t_all, fold_all, blocks)
        groups = tertile_groups(risk)

        lr = multivariate_logrank_test(t_all, groups, e_all)
        hr, ci_low, ci_high, hr_p = hazard_ratio_high_vs_low(t_all, e_all, groups)
        print(f"  n per group: {pd.Series(groups).value_counts().to_dict()}")
        print(f"  multivariate log-rank p = {lr.p_value:.2e}")
        print(f"  HR (High vs Low) = {hr:.2f}  95% CI [{ci_low:.2f}, {ci_high:.2f}]  p = {hr_p:.2e}")

        for g in ["Low", "Mid", "High"]:
            mask = groups == g
            kmf = KaplanMeierFitter()
            kmf.fit(t_all[mask], event_observed=e_all[mask], label=f"{g} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, color=colors[g], ci_show=False)

        p_str = f"{lr.p_value:.1e}" if lr.p_value >= 1e-4 else "<1e-4"
        ax.text(0.97, 0.05,
                f"log-rank p={p_str}\nHR(High/Low)={hr:.2f} [{ci_low:.2f}-{ci_high:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray"))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Time (days)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left", fontsize=8)

        summary_rows.append({
            "model": label, "n": len(risk),
            "n_low": (groups == "Low").sum(), "n_mid": (groups == "Mid").sum(), "n_high": (groups == "High").sum(),
            "events_low": e_all[groups == "Low"].sum(), "events_mid": e_all[groups == "Mid"].sum(),
            "events_high": e_all[groups == "High"].sum(),
            "logrank_p": lr.p_value, "HR_high_vs_low": hr, "HR_ci_low": ci_low, "HR_ci_high": ci_high,
            "HR_p": hr_p,
        })

    axes[0].set_ylabel("Survival probability")
    fig.suptitle("BRCA Overall Survival: Model Progression, Tertile Risk Stratification (out-of-fold)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
