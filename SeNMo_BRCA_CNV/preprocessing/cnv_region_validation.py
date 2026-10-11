"""
Three complementary checks on the 34 IG regions (ig_top100_regions.csv):

1. Univariate Cox: region CNV (mean raw segment log-ratio over the region's
   bins) as the only predictor; same 747 patients, OS time, event indicator.
   HR per +1 log-ratio unit (and per SD), 95% CI, Wald p, Benjamini-Hochberg
   FDR q across all 34 regions, plus a Schoenfeld proportional-hazards test.
2. Observed CNV vs predicted risk: raw region CNV against the out-of-fold
   predicted risk of (a) the GNN branch (the IG target) and (b) the frozen
   50/50 ensemble; Spearman rho; figure for the key loci (CCND1, the chr8p11
   amplicon, the EGFR-adjacent 7p11.2 regions) plus all 34 regions.
3. Conditional model response (ICE / partial dependence): for each region,
   set ALL of its bins to the same raw value v (keeps the region internally
   coherent; every other bin and the SeNMo embedding stay at the patient's
   real values), re-run the frozen GNN branch with the patient's own OOF
   fold head, and record the predicted-risk change. Values outside the
   region's observed 1st-99th percentile are flagged as extrapolation.

Reads existing outputs; writes only to SeNMo_BRCA_CNV/results/cnv_region_validation/:
  region_cox_univariate.csv, region_cnv_vs_risk.csv, region_conditional_response.csv,
  region_conditional_curves.csv, region_validation_summary.csv/.md,
  fig_cnv_vs_risk_key_loci.png, fig_cnv_vs_risk_all_regions.png,
  fig_conditional_response_key_loci.png, run_log.txt
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from lifelines.statistics import proportional_hazard_test  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency  # noqa: E402
import make_km_curves_comparison as kmc  # noqa: E402
from ig_cna_gnn_attribution import FrozenGNNRisk, train_head_frozen  # noqa: E402

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
RESULTS_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results")
IG_DIR = os.path.join(RESULTS_DIR, "ig_cna_gnn")
OUT_DIR = os.path.join(RESULTS_DIR, "cnv_region_validation")
ENCODER_PATH = os.path.join(EMBED_DIR, "cna_gnn_encoder.pt")
GNN_PKL = os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl")
ENSEMBLE_RISK_CSV = os.path.join(RESULTS_DIR, "km_curve_frozen_ensemble_risk_scores.csv")

GRID = np.round(np.arange(-1.0, 2.0001, 0.25), 2)  # raw log-ratio values imposed on a region
EVENT_COLORS = {0: "#7d8c99", 1: "#c0392b"}
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def region_labels(regions, ann):
    labels = []
    for r in regions.itertuples():
        a = ann[(ann.chrom.astype(str) == str(r.chrom)) & (ann.start_1based > r.start_bp) &
                (ann.end_1based <= r.end_bp)]
        genes = sorted({g for s in a.gene if not s.startswith("intergenic") for g in s.split(";")})
        if genes:
            name = "/".join(genes[:3])
        else:
            near = a.gene.iloc[0].split("nearest ")[1].split(",")[0]
            name = f"near {near}"
        band = a.cytoband.iloc[0] if len(a) else ""
        labels.append(f"{band} {name} (chr{r.chrom}:{r.start_bp / 1e6:.2f}-{r.end_bp / 1e6:.2f} Mb)")
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------ data + model
    ckpt = torch.load(ENCODER_PATH, weights_only=False)
    bin_columns = ckpt["bin_columns"]
    n_bins = len(bin_columns)
    adj = build_normalized_adjacency(ckpt["edge_index"], n_bins)
    encoder = CNAGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    sc_mean, sc_std = ckpt["scaler_mean"].astype(np.float32), ckpt["scaler_std"].astype(np.float32)

    d = kmc.load(GNN_PKL)
    pids = list(d["patient_ids"])
    senmo = d["embedding"].astype(np.float32)
    e_all, t_all, fold_all = d["e"], d["t"], d["fold"]
    X_raw = d["cna"].astype(np.float32)
    folds = sorted(set(fold_all))

    models, gnn_risk = {}, np.full(len(pids), np.nan)
    for k in folds:
        te, tr = fold_all == k, fold_all != k
        fs = StandardScaler().fit(d["cna_gnn"][tr])
        Xtr = np.concatenate([senmo[tr], fs.transform(d["cna_gnn"][tr])], axis=1)
        Xte = np.concatenate([senmo[te], fs.transform(d["cna_gnn"][te])], axis=1)
        head = train_head_frozen(Xtr, e_all[tr], t_all[tr])
        with torch.no_grad():
            tr_risk = head(torch.tensor(Xtr)).numpy().reshape(-1)
            te_risk = head(torch.tensor(Xte)).numpy().reshape(-1)
        rs = StandardScaler().fit(tr_risk.reshape(-1, 1))
        models[k] = FrozenGNNRisk(encoder, adj, head, fs.mean_, fs.scale_, rs.mean_[0], rs.scale_[0]).eval()
        gnn_risk[te] = rs.transform(te_risk.reshape(-1, 1)).reshape(-1)
    ens_risk = pd.read_csv(ENSEMBLE_RISK_CSV).set_index("patient_id").loc[pids, "risk_score"].values

    def predict(X_raw_mod):
        """OOF GNN-branch risk for every patient for a modified raw CNV matrix."""
        out = np.empty(len(pids), dtype=np.float32)
        xs = (X_raw_mod - sc_mean) / sc_std
        with torch.no_grad():
            for k in folds:
                te = fold_all == k
                out[te] = models[k](torch.tensor(xs[te]).t().unsqueeze(-1), torch.tensor(senmo[te])).numpy()
        return out

    chk = np.abs(predict(X_raw) - gnn_risk).max()
    log(f"[check] rebuilt GNN-branch OOF risk, graph path vs head-on-stored-embedding: max diff {chk:.2e}")
    assert chk < 1e-4

    # ------------------------------------------------------------ regions
    regions = pd.read_csv(os.path.join(IG_DIR, "ig_top100_regions.csv"))
    ann = pd.read_csv(os.path.join(IG_DIR, "ig_top100_bins_annotated.csv"))
    bins = pd.read_csv(os.path.join(IG_DIR, "ig_bin_ranking.csv"))
    col = {c: i for i, c in enumerate(bin_columns)}
    bins["col"] = bins["bin"].map(col)
    regions["label"] = region_labels(regions, ann)
    regions["region_id"] = [f"R{i + 1:02d}" for i in range(len(regions))]
    members = []
    for r in regions.itertuples():
        m = bins[(bins.chrom.astype(str) == str(r.chrom)) & (bins.start_bp >= r.start_bp) & (bins.end_bp <= r.end_bp)]
        assert len(m) == r.n_bins
        members.append(m.col.values)
    region_cnv = np.stack([X_raw[:, idx].mean(1) for idx in members], axis=1)  # (patients, regions)
    log(f"{len(regions)} regions, {len(pids)} patients, {int(e_all.sum())} events")

    # ------------------------------------------------------------ 1. univariate Cox
    rows = []
    for j, r in enumerate(regions.itertuples()):
        df = pd.DataFrame({"T": t_all, "E": e_all, "cnv": region_cnv[:, j]})
        cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
        s = cph.summary.loc["cnv"]
        sd = df.cnv.std()
        ph = proportional_hazard_test(cph, df, time_transform="rank").summary["p"].iloc[0]
        rows.append({"region_id": r.region_id, "region": r.label, "ig_best_rank": r.best_rank, "n_bins": r.n_bins,
                     "cnv_mean": df.cnv.mean(), "cnv_sd": sd,
                     "HR_per_unit": s["exp(coef)"], "HR_ci_low": s["exp(coef) lower 95%"],
                     "HR_ci_high": s["exp(coef) upper 95%"], "p": s["p"],
                     "HR_per_SD": np.exp(s["coef"] * sd), "c_index": cph.concordance_index_, "ph_test_p": ph})
    cox = pd.DataFrame(rows)
    cox["q_BH"] = bh_fdr(cox.p)

    # 1b. Same Cox, adjusted for SeNMo-alone OOF risk (the context the GNN head sees CNV in).
    senmo_risk = kmc.compute_oof_risk(senmo, e_all, t_all, fold_all, [])
    adj_rows = []
    for j in range(len(regions)):
        df = pd.DataFrame({"T": t_all, "E": e_all, "cnv": region_cnv[:, j], "senmo_risk": senmo_risk})
        s = CoxPHFitter().fit(df, duration_col="T", event_col="E").summary.loc["cnv"]
        adj_rows.append({"HR_adj_senmo": s["exp(coef)"], "HR_adj_ci_low": s["exp(coef) lower 95%"],
                         "HR_adj_ci_high": s["exp(coef) upper 95%"], "p_adj_senmo": s["p"]})
    cox = pd.concat([cox, pd.DataFrame(adj_rows)], axis=1)
    cox["q_BH_adj_senmo"] = bh_fdr(cox.p_adj_senmo)
    cox["spearman_cnv_vs_senmo_risk"] = [spearmanr(region_cnv[:, j], senmo_risk).correlation
                                         for j in range(len(regions))]
    cox.to_csv(os.path.join(OUT_DIR, "region_cox_univariate.csv"), index=False)

    # ------------------------------------------------------------ 2. CNV vs predicted risk
    rows = []
    for j, r in enumerate(regions.itertuples()):
        rho_g, p_g = spearmanr(region_cnv[:, j], gnn_risk)
        rho_e, p_e = spearmanr(region_cnv[:, j], ens_risk)
        rows.append({"region_id": r.region_id, "spearman_cnv_vs_gnn_risk": rho_g, "p_gnn": p_g,
                     "spearman_cnv_vs_ensemble_risk": rho_e, "p_ensemble": p_e})
    cvr = pd.DataFrame(rows)
    cvr["q_BH_gnn"] = bh_fdr(cvr.p_gnn)
    cvr.to_csv(os.path.join(OUT_DIR, "region_cnv_vs_risk.csv"), index=False)

    # Key loci: CCND1 region, chr8p11 amplicon (top two regions), EGFR-adjacent chr7 regions.
    def find(chrom, pos):
        m = regions[(regions.chrom.astype(str) == chrom) & (regions.start_bp <= pos) & (regions.end_bp >= pos)]
        return int(m.index[0])
    key = [("CCND1 (11q13.3)", find("11", 69_455_855)),
           ("chr8p11 amplicon, top region", find("8", 37_260_000)),
           ("chr8p11 amplicon, ADRB3", find("8", 37_820_516)),
           ("EGFR-adjacent: VOPP1/LANCL2", find("7", 55_510_000)),
           ("EGFR-adjacent: 17 kb from EGFR", find("7", 55_065_000))]

    def scatter(ax, j, title):
        x = region_cnv[:, j]
        for ev in (0, 1):
            m = e_all == ev
            ax.scatter(x[m], gnn_risk[m], s=9 if ev == 0 else 16, alpha=0.45 if ev == 0 else 0.8,
                       color=EVENT_COLORS[ev], label="event" if ev else "censored", linewidths=0)
        qs = np.quantile(x, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, 9)
        xm = [np.median(x[idx == b]) for b in range(10) if (idx == b).any()]
        ym = [np.median(gnn_risk[idx == b]) for b in range(10) if (idx == b).any()]
        ax.plot(xm, ym, color="#1f6f78", lw=2, marker="o", ms=3, label="decile median")
        ax.axvline(0, color="#898781", lw=0.8, ls="--")
        c = cox.iloc[j]
        rho = cvr.iloc[j].spearman_cnv_vs_gnn_risk
        ax.set_title(title, fontsize=9.5)
        ax.text(0.98, 0.03, f"Spearman ρ={rho:+.2f}\nCox HR={c.HR_per_unit:.2f} q={c.q_BH:.2g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cdccc3"))
        ax.grid(alpha=0.25)

    fig, axes = plt.subplots(1, len(key), figsize=(4.2 * len(key), 4.2), sharey=True)
    for ax, (name, j) in zip(axes, key):
        scatter(ax, j, f"{name}\n{regions.label[j]}")
        ax.set_xlabel("Region CNV (mean raw log-ratio)")
    axes[0].set_ylabel("Predicted risk, GNN branch (OOF, standardized)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Observed region CNV vs model-predicted risk (747 patients, out-of-fold)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_cnv_vs_risk_key_loci.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    ncol = 6
    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.1 * nrow), sharey=True)
    for j, ax in enumerate(axes.ravel()):
        if j >= len(regions):
            ax.axis("off")
            continue
        scatter(ax, j, f"{regions.region_id[j]} (IG rank {regions.best_rank[j]})\n{regions.label[j]}")
        ax.title.set_fontsize(7)
    fig.suptitle("All 34 IG regions: observed CNV vs GNN-branch predicted risk", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_cnv_vs_risk_all_regions.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------ 3. conditional response
    curves, rows = [], []
    for j, (r, idx) in enumerate(zip(regions.itertuples(), members)):
        lo, hi = np.percentile(region_cnv[:, j], [1, 99])
        risk_grid = np.empty((len(GRID), len(pids)), dtype=np.float32)
        for g, v in enumerate(GRID):
            Xm = X_raw.copy()
            Xm[:, idx] = v
            risk_grid[g] = predict(Xm)
        delta = risk_grid - gnn_risk  # vs the patient's actual prediction
        for g, v in enumerate(GRID):
            curves.append({"region_id": r.region_id, "value": v, "in_observed_p1_p99": lo <= v <= hi,
                           "mean_risk": risk_grid[g].mean(), "mean_delta_vs_actual": delta[g].mean(),
                           "p10_delta": np.percentile(delta[g], 10), "p90_delta": np.percentile(delta[g], 90)})
        in_rng = (GRID >= lo) & (GRID <= hi)
        slope = np.polyfit(GRID[in_rng], risk_grid[in_rng].mean(1), 1)[0] if in_rng.sum() >= 2 else np.nan
        i0, i1 = list(GRID).index(0.0), list(GRID).index(1.0)
        im = list(GRID).index(-0.5)
        rows.append({"region_id": r.region_id, "observed_p1": lo, "observed_p99": hi,
                     "slope_risk_per_unit_in_observed_range": slope,
                     "delta_risk_neutral_to_+1": (risk_grid[i1] - risk_grid[i0]).mean(),
                     "delta_risk_neutral_to_-0.5": (risk_grid[im] - risk_grid[i0]).mean(),
                     "frac_patients_risk_down_at_+1": (risk_grid[i1] < risk_grid[i0]).mean()})
    curves = pd.DataFrame(curves)
    curves.to_csv(os.path.join(OUT_DIR, "region_conditional_curves.csv"), index=False)
    cond = pd.DataFrame(rows)
    cond.to_csv(os.path.join(OUT_DIR, "region_conditional_response.csv"), index=False)

    fig, axes = plt.subplots(1, len(key), figsize=(4.2 * len(key), 3.9), sharey=True)
    for ax, (name, j) in zip(axes, key):
        c = curves[curves.region_id == regions.region_id[j]]
        ax.fill_between(c.value, c.p10_delta, c.p90_delta, color="#1f6f78", alpha=0.15, label="10th-90th pct of patients")
        ax.plot(c.value, c.mean_delta_vs_actual, color="#1f6f78", lw=2, label="mean")
        obs = c[c.in_observed_p1_p99]
        ax.axvspan(obs.value.min(), obs.value.max(), color="#9c6b13", alpha=0.07, label="observed 1-99% range")
        ax.axhline(0, color="#898781", lw=0.8)
        ax.axvline(0, color="#898781", lw=0.8, ls="--")
        ax.set_title(f"{name}\n{regions.label[j]}", fontsize=9.5)
        ax.set_xlabel("CNV imposed on all region bins (raw log-ratio)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Δ predicted risk vs actual (standardized)")
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Conditional model response: one region varied, all other inputs fixed (GNN branch, OOF heads)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_conditional_response_key_loci.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------ summary
    summ = regions[["region_id", "label", "best_rank", "n_bins", "mean_signed_ig", "corr_cnv_vs_signed_ig"]] \
        .merge(cox[["region_id", "HR_per_unit", "HR_ci_low", "HR_ci_high", "p", "q_BH", "ph_test_p",
                    "HR_adj_senmo", "HR_adj_ci_low", "HR_adj_ci_high", "q_BH_adj_senmo",
                    "spearman_cnv_vs_senmo_risk"]]) \
        .merge(cvr[["region_id", "spearman_cnv_vs_gnn_risk", "spearman_cnv_vs_ensemble_risk"]]) \
        .merge(cond[["region_id", "slope_risk_per_unit_in_observed_range", "delta_risk_neutral_to_+1"]])
    summ["data_direction"] = np.where(summ.HR_per_unit > 1, "gain->worse", "gain->better")
    summ["model_direction"] = np.where(summ.slope_risk_per_unit_in_observed_range > 0, "gain->higher risk",
                                       "gain->lower risk")
    summ["agree"] = (summ.HR_per_unit > 1) == (summ.slope_risk_per_unit_in_observed_range > 0)
    summ.to_csv(os.path.join(OUT_DIR, "region_validation_summary.csv"), index=False)

    lines = ["| Region | IG rank | Cox HR per unit [95% CI] | p | q (BH) | ρ CNV vs risk | Model slope | Agree |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summ.itertuples():
        lines.append(f"| {s.region_id} {s.label} | {s.best_rank} | {s.HR_per_unit:.2f} [{s.HR_ci_low:.2f}, "
                     f"{s.HR_ci_high:.2f}] | {s.p:.3g} | {s.q_BH:.3g} | {s.spearman_cnv_vs_gnn_risk:+.2f} | "
                     f"{s.slope_risk_per_unit_in_observed_range:+.3f} | {'yes' if s.agree else 'no'} |")
    with open(os.path.join(OUT_DIR, "region_validation_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"Cox: p<0.05 in {(cox.p < 0.05).sum()}/34 regions; q<0.05 in {(cox.q_BH < 0.05).sum()}; "
        f"q<0.10 in {(cox.q_BH < 0.10).sum()}; HR>1 in {(cox.HR_per_unit > 1).sum()}; "
        f"PH violated (p<0.05) in {(cox.ph_test_p < 0.05).sum()}")
    log(f"Cox adjusted for SeNMo risk: HR>1 in {(cox.HR_adj_senmo > 1).sum()}/34; "
        f"q<0.05 in {(cox.q_BH_adj_senmo < 0.05).sum()}; median HR {cox.HR_per_unit.median():.2f} -> "
        f"{cox.HR_adj_senmo.median():.2f}; median rho(CNV, SeNMo risk) {cox.spearman_cnv_vs_senmo_risk.median():+.2f}")
    log(cox[["region_id", "HR_per_unit", "q_BH", "HR_adj_senmo", "HR_adj_ci_low", "HR_adj_ci_high",
             "q_BH_adj_senmo", "spearman_cnv_vs_senmo_risk"]].round(3).to_string(index=False))
    log(f"CNV vs GNN risk: rho<0 in {(cvr.spearman_cnv_vs_gnn_risk < 0).sum()}/34; "
        f"q<0.05 in {(cvr.q_BH_gnn < 0.05).sum()}")
    log(f"Conditional: slope<0 in {(cond.slope_risk_per_unit_in_observed_range < 0).sum()}/34; "
        f"data/model direction agree in {int(summ.agree.sum())}/34")
    log("")
    log("\n".join(lines))
    with open(os.path.join(OUT_DIR, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
