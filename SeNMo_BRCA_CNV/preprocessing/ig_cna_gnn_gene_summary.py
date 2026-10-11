"""
Coordinate mapping + gene-level summary for the GNN-branch IG attributions
produced by ig_cna_gnn_attribution.py (reads its outputs, changes nothing).

Coordinate mapping
  Bin column "143010_chr8_3726to3727_10k" = chr8, 10 kb units 3726..3727
  -> GRCh37/hg19 chr8:37,260,001-37,270,000 (1-based, inclusive; build as
  established in fetch_gene_annotations.py: MYC spot-check matches hg19).
  Cytoband from Ensembl GRCh37 /overlap (feature=band).

Gene annotation (per bin)
  Ensembl GRCh37 genes within each top-100 region +/- 1 Mb (all biotypes
  fetched, cached). "Gene" column = protein-coding genes overlapping the bin;
  if none, the nearest protein-coding gene and its distance. Overlapping
  lncRNAs are listed separately.

Gene-level summary
  For each protein-coding gene overlapping >= 1 of the 7460 bins (project's
  gene_annotations.csv, which covers every bin run): per patient, gene IG =
  sum of that patient's signed IG over the gene's bins (IG is additive, so
  this is the gene's contribution to that patient's risk vs. the neutral
  baseline). Then mean |gene IG| (ranking), mean signed gene IG
  (direction), and a patient bootstrap of gene rank.

Outputs (SeNMo_BRCA_CNV/results/ig_cna_gnn/):
  ensembl_grch37_genes_top_regions.csv   cached Ensembl genes near top regions
  ig_top100_bins_annotated.csv / .md     Rank, Chr, Start, End, Cytoband, Gene, CNV value, Mean |IG|, Signed IG, ...
  ig_gene_level_summary.csv              all covered protein-coding genes, ranked
  ig_gene_level_top25.md                 top 25 genes table
"""
import os
import time

import numpy as np
import pandas as pd
import requests

ROOT = "/Users/yariv/Desktop/shani project"
EMBED_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "results", "ig_cna_gnn")
GENES_CSV = os.path.join(EMBED_DIR, "gene_annotations.csv")
GNN_PKL = os.path.join(EMBED_DIR, "brca_embedding_cna_gnn.pkl")
ENSEMBL = "https://grch37.rest.ensembl.org/overlap/region/human/{chrom}:{start}-{end}"
FLANK = 1_000_000
CACHE = os.path.join(OUT_DIR, "ensembl_grch37_genes_top_regions.csv")
BAND_CACHE = os.path.join(OUT_DIR, "ensembl_grch37_bands_top_regions.csv")
N_BOOT = 1000
BOOT_SEED = 66
TOP_GENES = 25


def ensembl(chrom, start, end, feature):
    url = ENSEMBL.format(chrom=chrom, start=start, end=end)
    for attempt in range(4):
        try:
            r = requests.get(url, params={"feature": feature, "content-type": "application/json"}, timeout=30)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1)


def fetch_windows(regions):
    """Gene + band features for each region +/- FLANK (<=5 Mb per Ensembl request)."""
    if os.path.exists(CACHE) and os.path.exists(BAND_CACHE):
        return pd.read_csv(CACHE), pd.read_csv(BAND_CACHE)
    genes, bands = {}, {}
    for chrom, s, e in regions:
        ws, we = max(1, s - FLANK), e + FLANK
        for g in ensembl(chrom, ws, we, "gene"):
            genes[g["id"]] = {"gene_id": g["id"], "name": g.get("external_name") or g["id"],
                              "biotype": g.get("biotype"), "chrom": str(g["seq_region_name"]),
                              "start": g["start"], "end": g["end"]}
        for b in ensembl(chrom, s + 1, e, "band"):
            bands[(str(b["seq_region_name"]), b["start"])] = {
                "chrom": str(b["seq_region_name"]), "start": b["start"], "end": b["end"], "band": b["id"]}
    g, b = pd.DataFrame(genes.values()), pd.DataFrame(bands.values())
    g.to_csv(CACHE, index=False)
    b.to_csv(BAND_CACHE, index=False)
    return g, b


def annotate_bin(chrom, s1, e1, genes, bands):
    """s1/e1: 1-based inclusive bin coordinates."""
    gc = genes[genes.chrom == chrom]
    ov = gc[(gc.start <= e1) & (gc.end >= s1)]
    pc = sorted(set(ov[ov.biotype == "protein_coding"].name))
    lnc = sorted(set(ov[ov.biotype == "lncRNA"].name)) if "lncRNA" in set(ov.biotype) else \
        sorted(set(ov[ov.biotype == "lincRNA"].name))
    if pc:
        gene, nearest, dist = ";".join(pc), "", 0
    else:
        p = gc[gc.biotype == "protein_coding"]
        d = np.maximum(p.start - e1, s1 - p.end).clip(lower=0)
        i = d.idxmin()
        gene = f"intergenic (nearest {p.name[i]}, {d[i] / 1e3:.0f} kb)"
        nearest, dist = p.name[i], int(d[i])
    bc = bands[(bands.chrom == chrom) & (bands.start <= e1) & (bands.end >= s1)]
    band = "/".join(f"{chrom}{x}" for x in bc.sort_values("start").band) if len(bc) else ""
    return gene, ";".join(lnc), nearest, dist, band


def md_table(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    bins = pd.read_csv(os.path.join(OUT_DIR, "ig_bin_ranking.csv"))
    regions = pd.read_csv(os.path.join(OUT_DIR, "ig_top100_regions.csv"))
    oof_ig = np.load(os.path.join(OUT_DIR, "ig_oof_signed.npy"))  # (747, 7460), original bin order
    data = pd.read_pickle(GNN_PKL)
    X_raw = data["cna"].astype(np.float32)
    n_pat, n_bins = oof_ig.shape

    # Original column order (ranking csv is sorted by rank).
    import torch
    bin_columns = torch.load(os.path.join(EMBED_DIR, "cna_graph.pt"), weights_only=False)["bin_columns"]
    pos = {c: i for i, c in enumerate(bin_columns)}
    bins["col_idx"] = bins["bin"].map(pos)
    bins["chrom"] = bins["chrom"].astype(str)
    bins["start_1based"] = bins["start_bp"] + 1   # 10 kb units * 10,000 is the 0-based start
    bins["end_1based"] = bins["end_bp"]

    # ---------------------------------------------------------------- top-100 bins
    genes_win, bands = fetch_windows(
        [(str(r.chrom), int(r.start_bp), int(r.end_bp)) for r in regions.itertuples()])
    genes_win["chrom"] = genes_win["chrom"].astype(str)
    bands["chrom"] = bands["chrom"].astype(str)
    print(f"Ensembl GRCh37: {len(genes_win)} genes "
          f"({(genes_win.biotype == 'protein_coding').sum()} protein-coding) near {len(regions)} regions")

    top = bins.nsmallest(100, "rank").copy()
    ann = [annotate_bin(r.chrom, r.start_1based, r.end_1based, genes_win, bands) for r in top.itertuples()]
    top["gene"], top["overlapping_lncRNA"], top["nearest_protein_coding"], top["nearest_distance_bp"], \
        top["cytoband"] = zip(*ann)
    keep = ["rank", "chrom", "start_1based", "end_1based", "cytoband", "gene", "overlapping_lncRNA",
            "mean_raw_cnv", "frac_gain_gt_0.2", "frac_loss_lt_-0.2", "mean_abs_ig", "mean_signed_ig",
            "direction_consistency", "corr_cnv_vs_signed_ig", "bootstrap_top100_freq",
            "in_top100_all_oof_folds", "bin"]
    top[keep].to_csv(os.path.join(OUT_DIR, "ig_top100_bins_annotated.csv"), index=False)

    fmt = {"Rank": str, "Chr": str, "Start": lambda v: f"{int(v):,}", "End": lambda v: f"{int(v):,}",
           "CNV value": lambda v: f"{v:+.3f}", "Mean |IG|": lambda v: f"{v:.5f}",
           "Signed IG": lambda v: f"{v:+.5f}", "Boot. freq": lambda v: f"{v:.2f}"}
    t = top.rename(columns={"rank": "Rank", "chrom": "Chr", "start_1based": "Start", "end_1based": "End",
                            "cytoband": "Cytoband", "gene": "Gene", "mean_raw_cnv": "CNV value",
                            "mean_abs_ig": "Mean |IG|", "mean_signed_ig": "Signed IG",
                            "bootstrap_top100_freq": "Boot. freq"})
    cols = ["Rank", "Chr", "Start", "End", "Cytoband", "Gene", "CNV value", "Mean |IG|", "Signed IG", "Boot. freq"]
    with open(os.path.join(OUT_DIR, "ig_top100_bins_annotated.md"), "w") as f:
        f.write("Top 100 CNV bins by mean |IG| (GNN branch, frozen recipe). Coordinates GRCh37/hg19, 1-based. "
                "CNV value = mean raw segment log-ratio across 747 patients (0 = copy-neutral). "
                "Signed IG = mean signed attribution (standardized risk units; <0 lowers predicted risk).\n\n")
        f.write(md_table(t, cols, fmt))

    # ---------------------------------------------------------------- gene level
    gp = pd.read_csv(GENES_CSV)
    gp["chrom"] = gp["chrom"].astype(str)
    bchrom = bins.set_index("col_idx").sort_index()
    b_chrom = bchrom["chrom"].values
    b_s, b_e = bchrom["start_1based"].values, bchrom["end_1based"].values
    rank_by_col = bchrom["rank"].values

    members, rows = [], []
    for g in gp.drop_duplicates("gene_id").itertuples():
        idx = np.where((b_chrom == g.chrom) & (b_s <= g.end) & (b_e >= g.start))[0]
        if len(idx) == 0:
            continue
        covered = sum(min(b_e[i], g.end) - max(b_s[i], g.start) + 1 for i in idx)
        members.append(idx)
        rows.append({"gene": g.external_name, "gene_id": g.gene_id, "chrom": g.chrom,
                     "gene_start": g.start, "gene_end": g.end, "n_bins": len(idx),
                     "frac_gene_covered_by_bins": covered / (g.end - g.start + 1),
                     "best_bin_rank": int(rank_by_col[idx].min())})
    gene_ig = np.stack([oof_ig[:, idx].sum(1) for idx in members], axis=1)  # (patients, genes)
    gene_cnv = np.stack([X_raw[:, idx].mean(1) for idx in members], axis=1)
    gdf = pd.DataFrame(rows)
    gdf["mean_abs_gene_ig"] = np.abs(gene_ig).mean(0)
    gdf["mean_signed_gene_ig"] = gene_ig.mean(0)
    gdf["mean_abs_ig_per_bin"] = gdf["mean_abs_gene_ig"] / gdf["n_bins"]
    gdf["direction_consistency"] = np.abs(gdf.mean_signed_gene_ig) / gdf.mean_abs_gene_ig
    gdf["mean_raw_cnv"] = gene_cnv.mean(0)
    gdf["frac_gain_gt_0.2"] = (gene_cnv > 0.2).mean(0)
    gdf["frac_loss_lt_-0.2"] = (gene_cnv < -0.2).mean(0)
    gdf["corr_cnv_vs_gene_ig"] = [np.corrcoef(gene_cnv[:, j], gene_ig[:, j])[0, 1] for j in range(len(gdf))]
    gdf["frac_patients_gene_ig_negative"] = (gene_ig < 0).mean(0)

    rng = np.random.default_rng(BOOT_SEED)
    in_top = np.zeros(len(gdf))
    ranks = np.zeros((N_BOOT, len(gdf)))
    for bi in range(N_BOOT):
        s = np.abs(gene_ig[rng.integers(0, n_pat, n_pat)]).mean(0)
        r = (-s).argsort().argsort() + 1
        ranks[bi] = r
        in_top[r <= TOP_GENES] += 1
    gdf[f"bootstrap_top{TOP_GENES}_freq"] = in_top / N_BOOT
    gdf["bootstrap_rank_p2.5"] = np.percentile(ranks, 2.5, axis=0)
    gdf["bootstrap_rank_p97.5"] = np.percentile(ranks, 97.5, axis=0)
    gdf = gdf.sort_values("mean_abs_gene_ig", ascending=False).reset_index(drop=True)
    gdf.insert(0, "gene_rank", np.arange(1, len(gdf) + 1))
    gdf.to_csv(os.path.join(OUT_DIR, "ig_gene_level_summary.csv"), index=False)

    gt = gdf.head(TOP_GENES).rename(columns={
        "gene_rank": "Rank", "chrom": "Chr", "gene_start": "Start", "gene_end": "End", "gene": "Gene",
        "mean_raw_cnv": "CNV value", "mean_abs_gene_ig": "Mean |IG|", "mean_signed_gene_ig": "Signed IG",
        "n_bins": "Bins", f"bootstrap_top{TOP_GENES}_freq": "Boot. freq"})
    gt["Boot. rank 95%"] = [f"{a:.0f}-{b:.0f}" for a, b in zip(gt["bootstrap_rank_p2.5"], gt["bootstrap_rank_p97.5"])]
    with open(os.path.join(OUT_DIR, "ig_gene_level_top25.md"), "w") as f:
        f.write(f"Top {TOP_GENES} protein-coding genes by mean |gene IG| (per-patient sum of bin IG over the gene's "
                f"bins, then mean absolute value across 747 patients). {len(gdf)} genes overlap the 7460 bins.\n\n")
        f.write(md_table(gt, ["Rank", "Chr", "Start", "End", "Gene", "Bins", "CNV value", "Mean |IG|",
                              "Signed IG", "Boot. freq", "Boot. rank 95%"], fmt))

    print(open(os.path.join(OUT_DIR, "ig_top100_bins_annotated.md")).read())
    print(open(os.path.join(OUT_DIR, "ig_gene_level_top25.md")).read())


if __name__ == "__main__":
    main()
