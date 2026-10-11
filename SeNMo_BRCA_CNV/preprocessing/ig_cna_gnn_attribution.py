"""
Integrated Gradients (IG) attribution of the frozen GNN branch's predicted
survival risk back to its CNV input bins.

Model being explained (exactly the frozen recipe, Frozen Model Validation §02):
  raw CNV (7460 bins, segment log-ratio)
    -> standardize with the pan-cancer (non-BRCA) scaler saved in cna_gnn_encoder.pt
    -> frozen 2-layer GCN encoder (graph connectivity fixed) -> mean+max pool (32-d)
    -> per-fold StandardScaler (fit on that fold's train patients)
    -> concat with the patient's SeNMo embedding (48-d, held fixed -- not attributed)
    -> per-fold Cox MLP head (make_km_curves_comparison.train_fold_frozen, seed 66)
    -> per-fold risk standardization (the scale the 50/50 ensemble averages on)
IG target = that standardized risk score (not the C-index). Because the final
ensemble is 0.5*(GNN-branch risk) + 0.5*(VAE-branch risk), the GNN-path IG of
the final ensemble score is exactly half of these values -- same ranking.

Baseline: raw CNV = 0 in every bin, i.e. copy-neutral log-ratio (the raw
values are continuous segment means centred on 0). In model-input space that
is (0 - scaler_mean) / scaler_std per bin -- NOT standardized zero, which
would be "average pan-cancer CNV", not neutral copy number.

Attributions are out-of-fold: each patient is explained with the head of the
fold in which they were held-out test data (the same model that produced
their reported risk score). Every patient is also explained under all five
fold heads, to separate head-to-head variability from patient sampling in the
stability analysis.

Outputs (SeNMo_BRCA_CNV/results/ig_cna_gnn/):
  ig_oof_signed.npy            747 x 7460 signed IG, out-of-fold head per patient
  ig_all_heads_signed.npy      5 x 747 x 7460 signed IG, every head x every patient
  ig_bin_ranking.csv           all 7460 bins ranked by mean |IG|, with signed stats
  ig_top100_bins.csv           top 100 bins (by mean |IG|) + signed values + stability
  ig_top100_regions.csv        top-100 bins merged into contiguous genomic regions
  ig_stability_summary.csv     fold and bootstrap stability metrics
  ig_run_log.txt               verification checks and summary
"""
import os
import pickle
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency, pool_embedding  # noqa: E402
import make_km_curves_comparison as kmc  # noqa: E402

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
RESULTS_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "ig_cna_gnn")
ENCODER_PATH = os.path.join(EMBED_DIR, "cna_gnn_encoder.pt")
GNN_PKL = os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl")
ES_PKL = os.path.join(EMBED_DIR, "brca_embedding_cnv_mutation_vae_hpo_earlystop.pkl")
GENES_CSV = os.path.join(EMBED_DIR, "gene_annotations.csv")
ENSEMBLE_RISK_CSV = os.path.join(RESULTS_DIR, "km_curve_frozen_ensemble_risk_scores.csv")

IG_STEPS = 64          # midpoint Riemann steps along the baseline -> input path
PATIENT_BATCH = 4      # patients per IG forward/backward (x IG_STEPS graph copies)
TOP_K = 100
N_BOOT = 1000
BOOT_SEED = 66
BIN_BP = 10_000        # bin coordinates in the column names are in 10 kb units

LOG_LINES = []


def log(msg=""):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# ---------------------------------------------------------------- frozen heads
def train_head_frozen(X_train, e_train, t_train, seed=kmc.SEED):
    """Same code path as kmc.train_fold_frozen, but returns the trained head."""
    torch.manual_seed(seed)
    model = kmc.make_mlp(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-3)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    etr = torch.tensor(e_train, dtype=torch.float32)
    ttr = torch.tensor(t_train, dtype=torch.float32)
    for _ in range(300):
        model.train()
        opt.zero_grad()
        loss = kmc.cox_loss(ttr, etr, model(Xtr))
        loss.backward()
        opt.step()
    model.eval()
    return model


class FrozenGNNRisk(torch.nn.Module):
    """Standardized CNV input (n_bins, batch, 1) + SeNMo emb (batch, 48) -> standardized risk (batch,)."""

    def __init__(self, encoder, adj, head, feat_mean, feat_scale, risk_mean, risk_scale):
        super().__init__()
        self.encoder, self.adj, self.head = encoder, adj, head
        self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_scale", torch.tensor(feat_scale, dtype=torch.float32))
        self.risk_mean, self.risk_scale = float(risk_mean), float(risk_scale)

    def forward(self, x_std, senmo):
        pooled = pool_embedding(self.encoder(x_std, self.adj))
        feat = (pooled - self.feat_mean) / self.feat_scale
        raw = self.head(torch.cat([senmo, feat], dim=1)).reshape(-1)
        return (raw - self.risk_mean) / self.risk_scale


def integrated_gradients(model, x_std, baseline_std, senmo, steps=IG_STEPS, batch=PATIENT_BATCH):
    """x_std: (n_patients, n_bins). Returns signed IG (n_patients, n_bins), f(x), f(baseline)."""
    n, n_bins = x_std.shape
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps  # midpoint rule
    base = torch.tensor(baseline_std, dtype=torch.float32)
    ig = np.zeros((n, n_bins), dtype=np.float32)
    f_x = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch):
        xb = torch.tensor(x_std[s:s + batch], dtype=torch.float32)  # (b, n_bins)
        sb = torch.tensor(senmo[s:s + batch], dtype=torch.float32)
        b = xb.shape[0]
        delta = xb - base  # (b, n_bins)
        path = base + alphas.view(1, steps, 1) * delta.view(b, 1, n_bins)  # (b, steps, n_bins)
        path = path.reshape(b * steps, n_bins).requires_grad_(True)
        out = model(path.t().unsqueeze(-1), sb.repeat_interleave(steps, dim=0))
        grads, = torch.autograd.grad(out.sum(), path)  # columns independent -> per-sample grads
        avg_grad = grads.reshape(b, steps, n_bins).mean(dim=1)
        ig[s:s + b] = (delta * avg_grad).detach().numpy()
        with torch.no_grad():
            f_x[s:s + b] = model(xb.t().unsqueeze(-1), sb).numpy()
    with torch.no_grad():
        f_base = model(base.view(n_bins, 1, 1).expand(n_bins, n, 1).contiguous(),
                       torch.tensor(senmo, dtype=torch.float32)).numpy()
    return ig, f_x, f_base


# ------------------------------------------------------------------- helpers
def topk_idx(scores, k=TOP_K):
    return np.argsort(-scores, kind="stable")[:k]


def overlap(a, b):
    return len(set(a) & set(b))


def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def pairwise_stability(score_rows, labels):
    rows = []
    tops = [topk_idx(s) for s in score_rows]
    for i, j in combinations(range(len(score_rows)), 2):
        rho = spearmanr(score_rows[i], score_rows[j]).correlation
        rows.append({"a": labels[i], "b": labels[j], "top100_overlap": overlap(tops[i], tops[j]),
                     "top100_jaccard": jaccard(tops[i], tops[j]), "spearman_all_bins": rho})
    return pd.DataFrame(rows)


def genes_for_bins(chroms, starts_bp, ends_bp, genes):
    out = []
    for c, s, e in zip(chroms, starts_bp, ends_bp):
        g = genes[(genes["chrom"].astype(str) == c) & (genes["start"] < e) & (genes["end"] > s)]
        out.append(";".join(sorted(set(g["external_name"].dropna()))))
    return out


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # Frozen encoder + fixed graph.
    ckpt = torch.load(ENCODER_PATH, weights_only=False)
    bin_columns = ckpt["bin_columns"]
    n_bins = len(bin_columns)
    adj = build_normalized_adjacency(ckpt["edge_index"], n_bins)
    encoder = CNAGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    sc_mean, sc_std = ckpt["scaler_mean"].astype(np.float32), ckpt["scaler_std"].astype(np.float32)

    gnn_data = kmc.load(GNN_PKL)
    es_data = kmc.load(ES_PKL)
    assert list(gnn_data["patient_ids"]) == list(es_data["patient_ids"])
    pids = list(gnn_data["patient_ids"])
    senmo = gnn_data["embedding"].astype(np.float32)
    e_all, t_all, fold_all = gnn_data["e"], gnn_data["t"], gnn_data["fold"]
    folds = sorted(set(fold_all))
    X_raw = gnn_data["cna"].astype(np.float32)
    X_std = (X_raw - sc_mean) / sc_std
    baseline_std = (0.0 - sc_mean) / sc_std  # copy-neutral log-ratio = 0
    log(f"patients={len(pids)} bins={n_bins} folds={folds} events={int(e_all.sum())}")
    log(f"raw CNV: mean={X_raw.mean():.4f} median={np.median(X_raw):.4f} "
        f"min={X_raw.min():.3f} max={X_raw.max():.3f}; baseline = raw 0 (copy-neutral)")

    # Check 1: the frozen encoder reproduces the stored pooled embedding.
    with torch.no_grad():
        pooled = pool_embedding(encoder(torch.tensor(X_std).t().unsqueeze(-1), adj)).numpy()
    err = np.abs(pooled - gnn_data["cna_gnn"]).max()
    log(f"[check] encoder re-run vs stored cna_gnn: max abs diff = {err:.2e}")
    assert err < 1e-4

    # Rebuild each fold's exact head; Check 2: reproduces the saved frozen-ensemble risks.
    fold_models, gnn_oof = {}, np.full(len(pids), np.nan)
    vae_oof = np.full(len(pids), np.nan)
    for k in folds:
        te, tr = fold_all == k, fold_all != k
        fs = StandardScaler().fit(gnn_data["cna_gnn"][tr])
        Xtr = np.concatenate([senmo[tr], fs.transform(gnn_data["cna_gnn"][tr])], axis=1)
        Xte = np.concatenate([senmo[te], fs.transform(gnn_data["cna_gnn"][te])], axis=1)
        tr_ref, te_ref = kmc.train_fold_frozen(Xtr, e_all[tr], t_all[tr], Xte)
        head = train_head_frozen(Xtr, e_all[tr], t_all[tr])
        with torch.no_grad():
            te_mine = head(torch.tensor(Xte, dtype=torch.float32)).numpy().reshape(-1)
        assert np.allclose(te_mine, te_ref, atol=1e-6), f"fold {k}: head not reproduced"
        rs = StandardScaler().fit(tr_ref.reshape(-1, 1))
        fold_models[k] = FrozenGNNRisk(encoder, adj, head, fs.mean_, fs.scale_, rs.mean_[0], rs.scale_[0]).eval()
        gnn_oof[te] = rs.transform(te_ref.reshape(-1, 1)).reshape(-1)

        vf = es_data["cnv_mutation_vae_hpo_earlystop"]
        vs = StandardScaler().fit(vf[tr])
        vtr, vte = kmc.train_fold_frozen(np.concatenate([senmo[tr], vs.transform(vf[tr])], 1), e_all[tr], t_all[tr],
                                         np.concatenate([senmo[te], vs.transform(vf[te])], 1))
        vrs = StandardScaler().fit(vtr.reshape(-1, 1))
        vae_oof[te] = vrs.transform(vte.reshape(-1, 1)).reshape(-1)

    saved = pd.read_csv(ENSEMBLE_RISK_CSV).set_index("patient_id").loc[pids, "risk_score"].values
    ens_err = np.abs(0.5 * (gnn_oof + vae_oof) - saved).max()
    log(f"[check] rebuilt 0.5*(GNN+VAE) OOF risk vs saved frozen-ensemble risk: max abs diff = {ens_err:.2e}")
    assert ens_err < 1e-4
    log(f"[check] GNN-branch OOF C-index (pooled) = {kmc_cindex(t_all, e_all, gnn_oof):.4f}")

    # IG for every patient under every fold head.
    all_ig = np.zeros((len(folds), len(pids), n_bins), dtype=np.float32)
    completeness = []
    for i, k in enumerate(folds):
        tk = time.time()
        ig, f_x, f_b = integrated_gradients(fold_models[k], X_std, baseline_std, senmo)
        all_ig[i] = ig
        gap = ig.sum(1) - (f_x - f_b)
        completeness.append(np.abs(gap))
        te = fold_all == k
        assert np.allclose(f_x[te], gnn_oof[te], atol=1e-4), "IG model output != frozen OOF risk"
        log(f"head fold {k}: IG done in {time.time() - tk:.0f}s; completeness |sum IG - (f(x)-f(x'))| "
            f"median={np.median(np.abs(gap)):.2e} max={np.abs(gap).max():.2e} "
            f"(median |f(x)-f(x')|={np.median(np.abs(f_x - f_b)):.3f})")
    np.save(os.path.join(OUT_DIR, "ig_all_heads_signed.npy"), all_ig)

    fold_index = {k: i for i, k in enumerate(folds)}
    oof_ig = np.stack([all_ig[fold_index[fold_all[p]], p] for p in range(len(pids))])
    np.save(os.path.join(OUT_DIR, "ig_oof_signed.npy"), oof_ig)

    # ---------------------------------------------------------- ranking
    abs_ig = np.abs(oof_ig)
    mean_abs = abs_ig.mean(0)
    mean_signed = oof_ig.mean(0)
    order = np.argsort(-mean_abs, kind="stable")
    rank = np.empty(n_bins, dtype=int)
    rank[order] = np.arange(1, n_bins + 1)

    chroms, starts, ends = kmc_parse_bins(bin_columns)
    starts_bp, ends_bp = np.array(starts) * BIN_BP, np.array(ends) * BIN_BP
    genes = pd.read_csv(GENES_CSV)
    # Sign interpretation: corr(raw CNV, signed IG) across patients, per bin.
    xc = X_raw - X_raw.mean(0)
    ic = oof_ig - oof_ig.mean(0)
    denom = np.sqrt((xc ** 2).sum(0) * (ic ** 2).sum(0))
    corr_cnv_ig = np.divide((xc * ic).sum(0), denom, out=np.zeros(n_bins), where=denom > 0)

    bins = pd.DataFrame({
        "rank": rank, "bin": bin_columns, "chrom": chroms, "start_bp": starts_bp, "end_bp": ends_bp,
        "mean_abs_ig": mean_abs, "mean_signed_ig": mean_signed,
        "direction_consistency": np.divide(np.abs(mean_signed), mean_abs, out=np.zeros(n_bins), where=mean_abs > 0),
        "frac_patients_ig_positive": (oof_ig > 0).mean(0),
        "mean_raw_cnv": X_raw.mean(0),
        "frac_gain_gt_0.2": (X_raw > 0.2).mean(0), "frac_loss_lt_-0.2": (X_raw < -0.2).mean(0),
        "corr_cnv_vs_signed_ig": corr_cnv_ig,
    })

    # ---------------------------------------------------------- stability
    labels = [f"fold{k}" for k in folds]
    # (a) Out-of-fold subsets: different patients AND different head.
    oof_fold_scores = [abs_ig[fold_all == k].mean(0) for k in folds]
    stab_oof = pairwise_stability(oof_fold_scores, labels).assign(comparison="OOF fold subsets (patients+head differ)")
    # (b) Same 747 patients, each fold's head: isolates head-to-head variability.
    head_scores = [np.abs(all_ig[i]).mean(0) for i in range(len(folds))]
    stab_head = pairwise_stability(head_scores, labels).assign(comparison="All patients, per-fold head")
    ref_top = topk_idx(mean_abs)
    for k, s in zip(folds, oof_fold_scores):
        bins[f"rank_oof_fold{k}"] = pd.Series(-s).rank(method="min").astype(int).values
    # (c) Bootstrap over patients (OOF attributions fixed, patients resampled).
    rng = np.random.default_rng(BOOT_SEED)
    sel_count = np.zeros(n_bins)
    boot_overlap, boot_rho = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(pids), len(pids))
        s = abs_ig[idx].mean(0)
        top = topk_idx(s)
        sel_count[top] += 1
        boot_overlap.append(overlap(top, ref_top))
        boot_rho.append(spearmanr(s, mean_abs).correlation)
    bins["bootstrap_top100_freq"] = sel_count / N_BOOT
    bins["in_top100_all_oof_folds"] = np.all([np.isin(np.arange(n_bins), topk_idx(s)) for s in oof_fold_scores], 0)
    bins["genes"] = genes_for_bins(chroms, starts_bp, ends_bp, genes)

    random_expect = TOP_K * TOP_K / n_bins
    stab = pd.concat([stab_oof, stab_head], ignore_index=True)
    boot_row = pd.DataFrame([{
        "comparison": f"Bootstrap patients (n={N_BOOT}) vs full-cohort top100", "a": "boot", "b": "full",
        "top100_overlap": np.mean(boot_overlap), "top100_jaccard": np.nan,
        "spearman_all_bins": np.mean(boot_rho),
        "overlap_p2.5": np.percentile(boot_overlap, 2.5), "overlap_p97.5": np.percentile(boot_overlap, 97.5)}])
    stab = pd.concat([stab, boot_row], ignore_index=True)
    stab.to_csv(os.path.join(OUT_DIR, "ig_stability_summary.csv"), index=False)

    bins = bins.sort_values("rank")
    bins.to_csv(os.path.join(OUT_DIR, "ig_bin_ranking.csv"), index=False)
    top = bins.head(TOP_K).copy()
    top.to_csv(os.path.join(OUT_DIR, "ig_top100_bins.csv"), index=False)

    # Contiguous regions among the top 100 (adjacent bins share a CNV segment).
    top_sorted = top.sort_values(["chrom", "start_bp"]).reset_index(drop=True)
    region_id, rid = [], 0
    for i in range(len(top_sorted)):
        if i > 0 and not (top_sorted.chrom[i] == top_sorted.chrom[i - 1]
                          and top_sorted.start_bp[i] == top_sorted.end_bp[i - 1]):
            rid += 1
        region_id.append(rid)
    top_sorted["region"] = region_id
    regions = top_sorted.groupby("region").agg(
        chrom=("chrom", "first"), start_bp=("start_bp", "min"), end_bp=("end_bp", "max"),
        n_bins=("bin", "size"), best_rank=("rank", "min"), sum_mean_abs_ig=("mean_abs_ig", "sum"),
        mean_signed_ig=("mean_signed_ig", "mean"), corr_cnv_vs_signed_ig=("corr_cnv_vs_signed_ig", "mean"),
        mean_raw_cnv=("mean_raw_cnv", "mean"), bootstrap_top100_freq=("bootstrap_top100_freq", "mean"),
        genes=("genes", lambda g: ";".join(sorted({x for s in g for x in s.split(";") if x}))),
    ).sort_values("best_rank")
    regions.to_csv(os.path.join(OUT_DIR, "ig_top100_regions.csv"), index=False)

    # ---------------------------------------------------------- summary
    log("")
    log(f"Top-{TOP_K} share of total mean|IG|: {top.mean_abs_ig.sum() / mean_abs.sum():.1%}")
    log(f"Top-{TOP_K} by chromosome: {top.chrom.value_counts().to_dict()} "
        f"(all bins: {pd.Series(chroms).value_counts().to_dict()})")
    log(f"Top-{TOP_K} collapse into {len(regions)} contiguous region(s)")
    log(f"Random expectation of top-{TOP_K} overlap: {random_expect:.2f} bins")
    for name, df in [("OOF fold subsets", stab_oof), ("per-fold heads, all patients", stab_head)]:
        log(f"{name}: top100 overlap mean={df.top100_overlap.mean():.1f} "
            f"[{df.top100_overlap.min()}-{df.top100_overlap.max()}], "
            f"Spearman mean={df.spearman_all_bins.mean():.3f} [{df.spearman_all_bins.min():.3f}-"
            f"{df.spearman_all_bins.max():.3f}]")
    log(f"bootstrap: top100 overlap with full-cohort mean={np.mean(boot_overlap):.1f} "
        f"95% [{np.percentile(boot_overlap, 2.5):.0f}, {np.percentile(boot_overlap, 97.5):.0f}], "
        f"Spearman mean={np.mean(boot_rho):.3f}")
    log(f"top-100 bins selected in >=95% of bootstraps: {(top.bootstrap_top100_freq >= 0.95).sum()}; "
        f"in top-100 of all 5 OOF folds: {int(top.in_top100_all_oof_folds.sum())}")
    log("")
    log(regions.head(20).to_string(index=False))
    log(f"\ntotal runtime {time.time() - t0:.0f}s")
    with open(os.path.join(OUT_DIR, "ig_run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")


def kmc_cindex(t, e, risk):
    from lifelines.utils import concordance_index
    return concordance_index(t, -risk, e)


def kmc_parse_bins(columns):
    from build_cna_graph import parse_bins
    return parse_bins(columns)


if __name__ == "__main__":
    main()
