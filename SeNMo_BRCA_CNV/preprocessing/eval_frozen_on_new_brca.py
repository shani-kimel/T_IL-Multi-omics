"""
Held-out evaluation of the FROZEN model on TCGA-BRCA patients that were never
part of the 747-patient development cohort (never used for architecture
choice, HPO, ensemble weighting, or any fold). Nothing about the model is
changed or re-tuned.

New-patient inputs
  - CNV: GDC SNP6 segments mapped onto the 7460-bin hg19 grid
    (map_gdc_segments_to_grid.py; variant chosen there by agreement with the
    TMO-Net grid on the 747 overlap)
  - SeNMo embedding: the released precomputed SeNMo parquet (same source as the 747)
  - Survival: TCGA-CDR OS / OS.time (verified identical to the 747's labels)

Model (Frozen Model Validation §02), trained ONCE on all 747 development
patients (their original TMO-Net grid CNV) with the frozen head recipe
(Linear(d,16)-ReLU-Dropout(0.3)-Linear(16,1), Adam lr 1e-3 wd 3e-3, 300 epochs,
seed 66); feature and risk standardization fit on the 747:
  - SeNMo alone (baseline), SeNMo+GNN, SeNMo+CNV-VAE, 50/50 ensemble (final)
Sensitivity: the five development-fold heads averaged instead of one refit.

Metrics on the held-out set: Harrell C-index with patient-bootstrap 95% CI
(2000 resamples), paired bootstrap of ensemble - SeNMo, and Kaplan-Meier /
log-rank / Cox HR with risk-group cut-offs FIXED from the 747 development
risks (median and tertiles), not re-fit on the new patients.

Outputs: SeNMo_BRCA_CNV/results/heldout_new_brca/ (default)

Optional: --patients <csv with patient_id> restricts the held-out set to a
sub-batch (e.g. complete_omics_excluded_brca.csv); --out <dir> sets the output folder.
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test, multivariate_logrank_test  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency, pool_embedding  # noqa: E402
from cnv_mrna_vae_model import Encoder  # noqa: E402
import make_km_curves_comparison as kmc  # noqa: E402

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
SEG_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_cn_segments")
VAL_CSV = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results", "gdc_grid_mapping", "validation.csv")
PARQUET = os.path.join(ROOT, "SeNMo_checkpoints", "Molecular (SeNMo).parquet")
CDR = os.path.join(SEG_DIR, "TCGA-CDR-SupplementalTableS1.xlsx")
OUT = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results", "heldout_new_brca")
N_BOOT = 2000
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def gnn_embed(X_raw, ckpt):
    adj = build_normalized_adjacency(ckpt["edge_index"], len(ckpt["bin_columns"]))
    enc = CNAGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.eval()
    xs = (X_raw - ckpt["scaler_mean"]) / ckpt["scaler_std"]
    with torch.no_grad():
        return pool_embedding(enc(torch.tensor(xs, dtype=torch.float32).t().unsqueeze(-1), adj)).numpy()


def vae_embed(X_raw, ckpt):
    enc = Encoder(ckpt["cna_dim"], ckpt["hidden_dims"], ckpt["latent_dim"], dropout=0.0)
    enc.load_state_dict(ckpt["encoder_cnv_state_dict"])
    enc.eval()
    xs = (X_raw - ckpt["cna_scaler_mean"]) / ckpt["cna_scaler_std"]
    with torch.no_grad():
        return enc(torch.tensor(xs, dtype=torch.float32))[0].numpy()


def fit_branch(dev_senmo, dev_feat, e, t, new_senmo, new_feat):
    """Frozen head recipe on all dev patients -> standardized risk for dev and new."""
    if dev_feat is None:
        Xd, Xn = dev_senmo, new_senmo
    else:
        fs = StandardScaler().fit(dev_feat)
        Xd = np.concatenate([dev_senmo, fs.transform(dev_feat)], 1)
        Xn = np.concatenate([new_senmo, fs.transform(new_feat)], 1)
    rd, rn = kmc.train_fold_frozen(Xd, e, t, Xn)
    rs = StandardScaler().fit(rd.reshape(-1, 1))
    return rs.transform(rd.reshape(-1, 1)).ravel(), rs.transform(rn.reshape(-1, 1)).ravel()


def fold_avg_branch(dev_senmo, dev_feat, e, t, fold, new_senmo, new_feat):
    """Sensitivity: the five development-fold heads, standardized risks averaged."""
    outs = []
    for k in sorted(set(fold)):
        tr = fold != k
        f_tr = None if dev_feat is None else dev_feat[tr]
        outs.append(fit_branch(dev_senmo[tr], f_tr, e[tr], t[tr], new_senmo, new_feat)[1])
    return np.mean(outs, 0)


def boot_cindex(t, e, risks, rng):
    n = len(t)
    out = {k: [] for k in risks}
    out["delta_ens_minus_senmo"] = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if e[i].sum() < 2:
            continue
        cs = {k: concordance_index(t[i], -r[i], e[i]) for k, r in risks.items()}
        for k, v in cs.items():
            out[k].append(v)
        out["delta_ens_minus_senmo"].append(cs["SeNMo + CNV ensemble (final)"] - cs["SeNMo alone"])
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", help="CSV with a patient_id column: restrict the held-out set")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(66)

    # ---------------- development cohort (exactly the frozen model's training data)
    dev = kmc.load(os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl"))
    dev_vae = kmc.load(os.path.join(EMBED_DIR, "brca_embedding_cnv_mutation_vae_hpo_earlystop.pkl"))
    assert list(dev["patient_ids"]) == list(dev_vae["patient_ids"])
    dev_ids = set(dev["patient_ids"])
    ds, de, dt, dfold = dev["embedding"].astype(np.float32), dev["e"], dev["t"], dev["fold"]
    d_gnn, d_vae = dev["cna_gnn"], dev_vae["cnv_mutation_vae_hpo_earlystop"]

    # ---------------- held-out patients
    variant = pd.read_csv(VAL_CSV).sort_values("gnn_risk_pearson_vs_ref", ascending=False).variant.iloc[0]
    g = torch.load(os.path.join(SEG_DIR, f"brca_grid_from_gdc_{variant}.pt"), weights_only=False)
    emb = pd.read_parquet(PARQUET)
    emb["ID"] = emb.PatientID.astype(str).str[:12]
    emb = emb.drop_duplicates("ID").set_index("ID")  # same rule as build_embedding_dataset.py
    cdr = pd.read_excel(CDR, sheet_name=0)
    cdr = cdr[cdr.type == "BRCA"].drop_duplicates("bcr_patient_barcode").set_index("bcr_patient_barcode")
    cdr["OS"] = pd.to_numeric(cdr.OS, errors="coerce")
    cdr["OS.time"] = pd.to_numeric(cdr["OS.time"], errors="coerce")

    cand = [p for p in g["patient_ids"] if p not in dev_ids]
    reasons = {"no SeNMo embedding": 0, "no OS / OS.time": 0, "OS.time <= 0": 0}
    new = []
    for p in cand:
        if p not in emb.index:
            reasons["no SeNMo embedding"] += 1
        elif p not in cdr.index or pd.isna(cdr.loc[p, "OS"]) or pd.isna(cdr.loc[p, "OS.time"]):
            reasons["no OS / OS.time"] += 1
        elif cdr.loc[p, "OS.time"] <= 0:
            reasons["OS.time <= 0"] += 1
        else:
            new.append(p)
    if args.patients:
        keep = set(pd.read_csv(args.patients).patient_id)
        log(f"restricting to {args.patients}: {len(keep)} listed, {len(keep & set(new))} in the held-out set "
            f"({len(keep - set(cand))} are in the dev cohort or not GDC-mapped, "
            f"{len(keep & set(cand)) - len(keep & set(new))} lack an embedding or OS)")
        new = [p for p in new if p in keep]
    log(f"GDC-mapped BRCA patients: {len(g['patient_ids'])}; not in dev cohort: {len(cand)}; "
        f"excluded: {reasons}; HELD-OUT SET: {len(new)}")
    gi = {p: i for i, p in enumerate(g["patient_ids"])}
    nX = g["cna"][[gi[p] for p in new]]
    ns = np.stack([np.asarray(emb.loc[p, "Embeddings"], dtype=np.float32) for p in new])
    ne = cdr.loc[new, "OS"].values.astype(float)
    nt = cdr.loc[new, "OS.time"].values.astype(float)
    log(f"held-out events: {int(ne.sum())} ({ne.mean():.1%}); dev events: {int(de.sum())} ({de.mean():.1%}); "
        f"median follow-up held-out {np.median(nt):.0f} d vs dev {np.median(dt):.0f} d")
    pd.DataFrame({"patient_id": new, "sample_barcode": [g["sample_barcodes"][gi[p]] for p in new],
                  "OS": ne, "OS_time": nt}).to_csv(os.path.join(OUT, "heldout_patients.csv"), index=False)

    gnn_ck = torch.load(os.path.join(EMBED_DIR, "cna_gnn_encoder.pt"), weights_only=False)
    vae_ck = torch.load(os.path.join(EMBED_DIR, "cnv_mutation_vae_hpo_earlystop_best_encoder.pt"), weights_only=False)
    n_gnn, n_vae = gnn_embed(nX, gnn_ck), vae_embed(nX, vae_ck)

    # ---------------- risks (single refit on all 747 = primary; fold-average = sensitivity)
    r_dev, r_new, r_new_fa = {}, {}, {}
    r_dev["SeNMo alone"], r_new["SeNMo alone"] = fit_branch(ds, None, de, dt, ns, None)
    r_dev["SeNMo + GNN"], r_new["SeNMo + GNN"] = fit_branch(ds, d_gnn, de, dt, ns, n_gnn)
    r_dev["SeNMo + CNV-VAE"], r_new["SeNMo + CNV-VAE"] = fit_branch(ds, d_vae, de, dt, ns, n_vae)
    r_dev["SeNMo + CNV ensemble (final)"] = 0.5 * (r_dev["SeNMo + GNN"] + r_dev["SeNMo + CNV-VAE"])
    r_new["SeNMo + CNV ensemble (final)"] = 0.5 * (r_new["SeNMo + GNN"] + r_new["SeNMo + CNV-VAE"])
    r_new_fa["SeNMo alone"] = fold_avg_branch(ds, None, de, dt, dfold, ns, None)
    fa_g = fold_avg_branch(ds, d_gnn, de, dt, dfold, ns, n_gnn)
    fa_v = fold_avg_branch(ds, d_vae, de, dt, dfold, ns, n_vae)
    r_new_fa["SeNMo + GNN"], r_new_fa["SeNMo + CNV-VAE"] = fa_g, fa_v
    r_new_fa["SeNMo + CNV ensemble (final)"] = 0.5 * (fa_g + fa_v)

    boot = boot_cindex(nt, ne, r_new, rng)
    rows = []
    for k in r_new:
        c = concordance_index(nt, -r_new[k], ne)
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append({"model": k, "heldout_cindex": c, "ci_low": lo, "ci_high": hi,
                     "heldout_cindex_foldavg_heads": concordance_index(nt, -r_new_fa[k], ne),
                     "dev_apparent_cindex_(train_fit)": concordance_index(dt, -r_dev[k], de)})
    res = pd.DataFrame(rows)
    dlt = np.array(boot["delta_ens_minus_senmo"])
    d_point = res.set_index("model").heldout_cindex
    d_point = d_point["SeNMo + CNV ensemble (final)"] - d_point["SeNMo alone"]
    res.to_csv(os.path.join(OUT, "heldout_cindex.csv"), index=False)
    log(res.round(4).to_string(index=False))
    log(f"paired bootstrap ensemble - SeNMo: delta={d_point:+.4f} 95% CI [{np.percentile(dlt, 2.5):+.4f}, "
        f"{np.percentile(dlt, 97.5):+.4f}]; P(delta>0)={np.mean(dlt > 0):.3f}")

    # ---------------- KM with cut-offs fixed from development risks
    colors = {"Low": "#2471a3", "Mid": "#7d8c99", "High": "#c0392b"}
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)
    km_rows = []
    for col, k in enumerate(r_new):
        med = np.median(r_dev[k])
        q1, q2 = np.percentile(r_dev[k], [100 / 3, 200 / 3])
        for row_i, (scheme, groups) in enumerate([
                ("median", np.where(r_new[k] > med, "High", "Low")),
                ("tertile", np.where(r_new[k] <= q1, "Low", np.where(r_new[k] <= q2, "Mid", "High")))]):
            ax = axes[row_i, col]
            for gname in ["Low", "Mid", "High"]:
                m = groups == gname
                if m.sum() == 0:
                    continue
                KaplanMeierFitter().fit(nt[m], ne[m], label=f"{gname} (n={m.sum()}, ev={int(ne[m].sum())})") \
                    .plot_survival_function(ax=ax, color=colors[gname], ci_show=False)
            p = logrank_test(nt[groups == "High"], nt[groups == "Low"], ne[groups == "High"], ne[groups == "Low"]).p_value \
                if scheme == "median" else multivariate_logrank_test(nt, groups, ne).p_value
            hm = groups != "Mid"
            hr_df = pd.DataFrame({"T": nt[hm], "E": ne[hm], "high": (groups[hm] == "High").astype(int)})
            cph = CoxPHFitter().fit(hr_df, "T", "E")
            hr = np.exp(cph.params_["high"])
            hlo, hhi = np.exp(cph.confidence_intervals_.loc["high"])
            km_rows.append({"model": k, "scheme": scheme, "logrank_p": p, "HR_high_vs_low": hr, "HR_ci_low": hlo,
                            "HR_ci_high": hhi, **{f"n_{x}": int((groups == x).sum()) for x in ["Low", "Mid", "High"]},
                            **{f"events_{x}": int(ne[groups == x].sum()) for x in ["Low", "Mid", "High"]}})
            ax.text(0.97, 0.05, f"log-rank p={p:.2g}\nHR(High/Low)={hr:.2f} [{hlo:.2f}-{hhi:.2f}]",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray"))
            ax.set_title(f"{k} - {scheme} split (dev cut-offs)", fontsize=10)
            ax.set_xlabel("Time (days)")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            ax.legend(loc="lower left", fontsize=8)
    axes[0, 0].set_ylabel("Survival probability")
    axes[1, 0].set_ylabel("Survival probability")
    fig.suptitle(f"Held-out TCGA-BRCA patients never used in development{' - ' + os.path.basename(args.patients) if args.patients else ''} (n={len(new)}, "
                 f"{int(ne.sum())} events), frozen model, cut-offs from development cohort", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "km_heldout.png"), dpi=170, bbox_inches="tight")
    km = pd.DataFrame(km_rows)
    km.to_csv(os.path.join(OUT, "km_heldout_summary.csv"), index=False)
    log(km.round(4).to_string(index=False))
    pd.DataFrame({"patient_id": new, **{k: v for k, v in r_new.items()}}).to_csv(
        os.path.join(OUT, "heldout_risk_scores.csv"), index=False)
    with open(os.path.join(OUT, "eval_log.txt"), "w") as f:
        f.write(f"CNV source variant: {variant}\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
