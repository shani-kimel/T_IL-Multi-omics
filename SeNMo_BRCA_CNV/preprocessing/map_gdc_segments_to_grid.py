"""
Map GDC TCGA-BRCA SNP6 segment files (GRCh38, log2 Segment_Mean) onto the
frozen model's 7460-bin hg19 CNV grid, so BRCA patients the TMO-Net grid
never included can be scored by the frozen model WITHOUT changing it.

Steps
  1. Lift each hg19 bin [start, end) to GRCh38 (pyliftover, UCSC chain);
     a bin whose two edges land on a different chromosome/strand or whose
     length changes by >50% is marked unmappable.
  2. Per patient, one tumor sample: primary tumor (-01) preferred, then the
     lowest sample barcode. Bin value = overlap-length-weighted mean of the
     segments' Segment_Mean; bins in segment gaps (or unmappable) are
     filled from the nearest covered bin of the same contiguous run.
  3. Done for both GDC variants (unmasked / germline-masked).
  4. Validation on the 747 patients present in both sources:
       a) value agreement with the TMO-Net grid (Pearson, MAE, per-patient r)
       b) model agreement: the frozen GNN branch (same OOF fold heads) run on
          GDC-derived vs TMO-Net inputs -- risk correlation and C-index.
     The variant with the better agreement is the one used downstream.

Outputs
  SeNMo_BRCA_CNV/data/gdc_cn_segments/brca_grid_from_gdc_{unmasked,masked}.pt
      dict: patient_ids, sample_barcodes, cna (n x 7460 float32, raw log2), bin_columns,
            unmappable_bins, filled_fraction
  SeNMo_BRCA_CNV/results/gdc_grid_mapping/validation.csv, mapping_log.txt,
      fig_gdc_vs_tmonet.png
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from pyliftover import LiftOver  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cna_graph import parse_bins  # noqa: E402
from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency  # noqa: E402
import make_km_curves_comparison as kmc  # noqa: E402
from ig_cna_gnn_attribution import FrozenGNNRisk, train_head_frozen  # noqa: E402

ROOT = "/Users/yariv/Desktop/shani project"
SEG_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_cn_segments")
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
RES_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results", "gdc_grid_mapping")
BIN_BP = 10_000
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def lift_bins(bin_columns):
    lo = LiftOver("hg19", "hg38")
    chroms, starts, ends = parse_bins(bin_columns)
    rows = []
    for c, s, e in zip(chroms, starts, ends):
        s_bp, e_bp = s * BIN_BP, e * BIN_BP
        a = lo.convert_coordinate(f"chr{c}", s_bp)
        b = lo.convert_coordinate(f"chr{c}", e_bp - 1)
        ok = bool(a) and bool(b) and a[0][0] == b[0][0] == f"chr{c}" and a[0][2] == b[0][2] == "+"
        if ok:
            ns, ne = a[0][1], b[0][1] + 1
            ok = ne > ns and abs((ne - ns) - (e_bp - s_bp)) <= 0.5 * (e_bp - s_bp)
        rows.append({"chrom": c, "start38": ns if ok else -1, "end38": ne if ok else -1, "ok": ok})
    return pd.DataFrame(rows)


def run_ids(chroms, starts, ends):
    rid, out = 0, [0]
    for i in range(1, len(chroms)):
        if not (chroms[i] == chroms[i - 1] and starts[i] == ends[i - 1]):
            rid += 1
        out.append(rid)
    return np.array(out)


def segments_to_bins(seg, bins38, runs):
    """seg: one sample's segments. Returns (n_bins,) overlap-weighted Segment_Mean, gaps filled."""
    vals = np.full(len(bins38), np.nan, dtype=np.float64)
    seg = seg.assign(Chromosome=seg.Chromosome.astype(str))
    for c, bidx in bins38[bins38.ok].groupby("chrom").groups.items():
        s = seg[seg.Chromosome == c].sort_values("Start")
        if s.empty:
            continue
        ss, se, sm = s.Start.values, s.End.values + 1, s.Segment_Mean.values  # seg End is inclusive
        for i in bidx:
            b0, b1 = bins38.start38[i], bins38.end38[i]
            lo_i = max(np.searchsorted(se, b0, side="right"), 0)
            hi_i = np.searchsorted(ss, b1, side="left")
            if hi_i <= lo_i:
                continue
            ov = np.minimum(se[lo_i:hi_i], b1) - np.maximum(ss[lo_i:hi_i], b0)
            m = ov > 0
            if m.any():
                vals[i] = np.average(sm[lo_i:hi_i][m], weights=ov[m])
    missing = np.isnan(vals)
    if missing.any():  # nearest covered bin within the same contiguous run
        for r in np.unique(runs[missing]):
            idx = np.where(runs == r)[0]
            have = idx[~np.isnan(vals[idx])]
            if len(have) == 0:
                continue
            for i in idx[np.isnan(vals[idx])]:
                vals[i] = vals[have[np.argmin(np.abs(have - i))]]
    return vals.astype(np.float32), missing.mean()


def build_variant(variant, files, bins38, runs, bin_columns):
    f = files[files.data_type == ("Masked Copy Number Segment" if variant == "masked" else "Copy Number Segment")]
    f = f.assign(primary=f.sample_barcode.str[13:15] == "01") \
         .sort_values(["case_id", "primary", "sample_barcode"], ascending=[True, False, True]) \
         .drop_duplicates("case_id")
    mats, filled = [], []
    for r in f.itertuples():
        seg = pd.read_csv(os.path.join(SEG_DIR, "brca", variant, f"{r.file_id}.seg.txt"), sep="\t")
        v, fr = segments_to_bins(seg, bins38, runs)
        mats.append(v)
        filled.append(fr)
    cna = np.stack(mats)
    out = {"patient_ids": f.case_id.tolist(), "sample_barcodes": f.sample_barcode.tolist(), "cna": cna,
           "bin_columns": bin_columns, "unmappable_bins": np.where(~bins38.ok.values)[0],
           "filled_fraction": np.array(filled), "source": f"GDC SNP6 DNAcopy {variant}, GRCh38 -> hg19 grid"}
    torch.save(out, os.path.join(SEG_DIR, f"brca_grid_from_gdc_{variant}.pt"))
    log(f"[{variant}] {cna.shape[0]} patients; bins filled from neighbours: median "
        f"{np.median(filled):.2%}, max {np.max(filled):.2%}; remaining NaN {np.isnan(cna).mean():.3%}")
    return out


def gnn_oof_risk(X_raw_by_pid, d, encoder, adj, sc_mean, sc_std):
    """Frozen GNN branch with the original OOF fold heads (trained on TMO-Net-input embeddings)."""
    pids, senmo, e, t, fold = d["patient_ids"], d["embedding"].astype(np.float32), d["e"], d["t"], d["fold"]
    X = np.stack([X_raw_by_pid[p] for p in pids])
    xs = (X - sc_mean) / sc_std
    risk = np.full(len(pids), np.nan, dtype=np.float32)
    for k in sorted(set(fold)):
        te, tr = fold == k, fold != k
        fs = StandardScaler().fit(d["cna_gnn"][tr])
        Xtr = np.concatenate([senmo[tr], fs.transform(d["cna_gnn"][tr])], axis=1)
        head = train_head_frozen(Xtr, e[tr], t[tr])
        with torch.no_grad():
            tr_risk = head(torch.tensor(Xtr)).numpy().reshape(-1)
        rs = StandardScaler().fit(tr_risk.reshape(-1, 1))
        model = FrozenGNNRisk(encoder, adj, head, fs.mean_, fs.scale_, rs.mean_[0], rs.scale_[0]).eval()
        with torch.no_grad():
            risk[te] = model(torch.tensor(xs[te]).t().unsqueeze(-1), torch.tensor(senmo[te])).numpy()
    return risk


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    ckpt = torch.load(os.path.join(EMBED_DIR, "cna_gnn_encoder.pt"), weights_only=False)
    bin_columns = ckpt["bin_columns"]
    chroms, starts, ends = parse_bins(bin_columns)
    runs = run_ids(chroms, starts, ends)
    bins38 = lift_bins(bin_columns)
    log(f"liftover hg19->GRCh38: {bins38.ok.sum()}/{len(bins38)} bins mappable")

    files = pd.read_csv(os.path.join(SEG_DIR, "brca", "files.csv"))
    d = kmc.load(os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl"))
    tmo = dict(zip(d["patient_ids"], d["cna"].astype(np.float32)))
    adj = build_normalized_adjacency(ckpt["edge_index"], len(bin_columns))
    encoder = CNAGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    sc_mean, sc_std = ckpt["scaler_mean"].astype(np.float32), ckpt["scaler_std"].astype(np.float32)
    ref_risk = gnn_oof_risk(tmo, d, encoder, adj, sc_mean, sc_std)
    ref_c = concordance_index(d["t"], -ref_risk, d["e"])
    log(f"reference (TMO-Net input) GNN-branch OOF C-index: {ref_c:.4f}")

    rows, plot_data = [], {}
    for variant in ("unmasked", "masked"):
        out = build_variant(variant, files, bins38, runs, bin_columns)
        gdc = dict(zip(out["patient_ids"], out["cna"]))
        common = [p for p in d["patient_ids"] if p in gdc]
        A = np.stack([tmo[p] for p in common]).astype(np.float64)  # float64: polyfit is unstable in float32
        B = np.stack([gdc[p] for p in common]).astype(np.float64)
        per_pat = [np.corrcoef(a, b)[0, 1] for a, b in zip(A, B) if a.std() > 0 and b.std() > 0]
        slope, icpt = np.polyfit(B.ravel(), A.ravel(), 1)
        risk = gnn_oof_risk({**tmo, **{p: gdc[p] for p in common}}, d, encoder, adj, sc_mean, sc_std) \
            if len(common) == len(d["patient_ids"]) else None
        row = {"variant": variant, "n_common": len(common), "pearson_all_values": np.corrcoef(A.ravel(), B.ravel())[0, 1],
               "median_per_patient_r": np.median(per_pat), "p10_per_patient_r": np.percentile(per_pat, 10),
               "mae": np.abs(A - B).mean(), "fit_slope_tmo_on_gdc": slope, "fit_intercept": icpt,
               "frac_patients_r_gt_0.9": np.mean(np.array(per_pat) > 0.9)}
        if risk is not None:
            row.update({"gnn_risk_pearson_vs_ref": np.corrcoef(risk, ref_risk)[0, 1],
                        "gnn_oof_cindex_gdc_input": concordance_index(d["t"], -risk, d["e"]),
                        "gnn_oof_cindex_tmonet_input": ref_c})
        rows.append(row)
        plot_data[variant] = (A, B, risk)
        log(pd.Series(row).to_string())

    val = pd.DataFrame(rows)
    val.to_csv(os.path.join(RES_DIR, "validation.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    A, B, _ = plot_data["unmasked"]
    rng = np.random.default_rng(0)
    i = rng.choice(A.size, 60000, replace=False)
    axes[0].hexbin(B.ravel()[i], A.ravel()[i], gridsize=70, bins="log", cmap="viridis", extent=(-2, 3, -2, 3))
    axes[0].plot([-2, 3], [-2, 3], color="white", lw=0.8, ls="--")
    axes[0].set_xlabel("GDC segments -> grid (unmasked)")
    axes[0].set_ylabel("TMO-Net grid value")
    axes[0].set_title("Bin values, 747 overlap patients (60k sampled)", fontsize=10)
    for v, col in (("unmasked", "#1f6f78"), ("masked", "#9c6b13")):
        a, b, _ = plot_data[v]
        r = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b) if x.std() > 0 and y.std() > 0]
        axes[1].hist(r, bins=np.linspace(0, 1, 41), alpha=0.6, color=col, label=v)
    axes[1].set_xlabel("Per-patient Pearson r (GDC-derived vs TMO-Net)")
    axes[1].set_title("Per-patient agreement", fontsize=10)
    axes[1].legend()
    best = val.sort_values("gnn_risk_pearson_vs_ref", ascending=False).variant.iloc[0]
    axes[2].scatter(ref_risk, plot_data[best][2], s=6, alpha=0.5, color="#1f6f78")
    lim = [min(ref_risk.min(), plot_data[best][2].min()), max(ref_risk.max(), plot_data[best][2].max())]
    axes[2].plot(lim, lim, color="#898781", lw=0.8, ls="--")
    axes[2].set_xlabel("GNN-branch risk, TMO-Net input")
    axes[2].set_ylabel(f"GNN-branch risk, GDC input ({best})")
    axes[2].set_title("Frozen model output agreement (OOF heads)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig_gdc_vs_tmonet.png"), dpi=160, bbox_inches="tight")
    log(f"best-agreeing variant: {best}")
    with open(os.path.join(RES_DIR, "mapping_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
