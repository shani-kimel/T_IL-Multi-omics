"""
Genome-wide gene-level copy number for the 747-patient BRCA survival cohort,
from the GDC files already downloaded by minds/pull_brca_cnv_expression.py.

Unlike the TMO-Net 7460-bin grid (chr7/8/11 windows only, hg19 log-ratio),
these are GDC "Gene Level Copy Number" files: all chromosomes, 60,623
GENCODE v36 genes, GRCh38, absolute integer copy number (2 = diploid).

Workflow: ASCAT2 (the GDC default "*.gene_level_copy_number.v36.tsv";
the ascat3 / absolute_liftover variants are not used). One tumor sample per
patient: the GDC API is asked which tumor sample each local file was called
from; primary tumor (-01) is preferred, then the lowest sample/vial barcode.

Outputs (SeNMo_BRCA_CNV/data/gdc_gene_cn/):
  brca_gene_cn_ascat2.pt        dict: patient_ids, sample_barcodes, file_ids, gene table
                                (gene_id, gene_name, chromosome, start, end), cn (747 x 60623 float32)
  brca_gene_cn_chr17_chr13.csv  patients x genes (chr17 + chr13 only), column = gene_name|gene_id
  brca_key_genes_cn.csv         ERBB2, TP53, BRCA1, BRCA2, RB1 + grid genes, with copy-number calls
  build_log.txt
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import requests
import torch

ROOT = "/Users/yariv/Desktop/shani project"
RAW = os.path.join(ROOT, "SeNMo_BRCA_CNV", "minds", "brca_cohort", "raw")
COHORT_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna_gnn.pkl")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_gene_cn")
GDC_FILES = "https://api.gdc.cancer.gov/files"
KEY_GENES = ["ERBB2", "GRB7", "TP53", "BRCA1", "BRCA2", "RB1", "CCND1", "MYC", "EGFR", "FGFR1", "ZNF703"]
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def gdc_tumor_samples(file_ids):
    """file_id -> (tumor sample barcode, sample type, workflow) via the GDC API."""
    out = {}
    for s in range(0, len(file_ids), 500):
        chunk = file_ids[s:s + 500]
        body = {"filters": {"op": "in", "content": {"field": "file_id", "value": chunk}},
                "fields": "file_id,analysis.workflow_type,cases.samples.submitter_id,cases.samples.sample_type",
                "size": len(chunk)}
        r = requests.post(GDC_FILES, json=body, timeout=120)
        r.raise_for_status()
        for h in r.json()["data"]["hits"]:
            tumors = [x for x in h["cases"][0]["samples"] if "normal" not in x["sample_type"].lower()]
            t = sorted(tumors, key=lambda x: x["submitter_id"])[0]
            out[h["file_id"]] = (t["submitter_id"], t["sample_type"], h["analysis"]["workflow_type"])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(COHORT_PKL, "rb") as f:
        pids = list(pickle.load(f)["patient_ids"])

    # Local ASCAT2 (default-named) files per patient.
    local = {}
    for pid in pids:
        for p in glob.glob(os.path.join(RAW, pid, "Gene Level Copy Number", "*", "*.gene_level_copy_number.v36.tsv")):
            if ".ascat3." in p or ".absolute_liftover." in p:
                continue
            local.setdefault(pid, []).append((os.path.basename(os.path.dirname(p)), p))
    log(f"cohort {len(pids)} patients; with local ASCAT2 gene-level CN: {len(local)}; "
        f"files: {sum(len(v) for v in local.values())}")

    meta = gdc_tumor_samples([fid for v in local.values() for fid, _ in v])
    rows = []
    for pid in pids:
        cands = []
        for fid, path in local.get(pid, []):
            barcode, stype, wf = meta[fid]
            assert wf == "ASCAT2", (fid, wf)
            primary = barcode[13:15] == "01"
            cands.append((not primary, barcode, fid, path, stype))
        if not cands:
            rows.append({"patient_id": pid, "file_id": None})
            continue
        cands.sort()
        _, barcode, fid, path, stype = cands[0]
        rows.append({"patient_id": pid, "file_id": fid, "sample_barcode": barcode, "sample_type": stype,
                     "path": path, "n_candidate_files": len(cands)})
    sel = pd.DataFrame(rows)
    log(f"selected samples by type: {sel.sample_type.value_counts(dropna=False).to_dict()}; "
        f"patients with >1 candidate file: {(sel.n_candidate_files > 1).sum()}")

    log(f"no ASCAT2 file (row left NaN): {sel.patient_id[sel.file_id.isna()].tolist()}")
    genes, cn = None, np.full((len(pids), 0), np.nan, dtype=np.float32)
    for i, r in sorted(enumerate(sel.itertuples()), key=lambda x: pd.isna(x[1].path)):
        if pd.isna(r.path):
            continue
        df = pd.read_csv(r.path, sep="\t")
        if genes is None:
            genes = df[["gene_id", "gene_name", "chromosome", "start", "end"]].copy()
            cn = np.full((len(pids), len(genes)), np.nan, dtype=np.float32)
        assert (df.gene_id.values == genes.gene_id.values).all(), f"gene order differs: {r.path}"
        cn[i] = df.copy_number.values.astype(np.float32)
    log(f"matrix {cn.shape}; missing copy_number: {np.isnan(cn).mean():.2%}")
    log(f"genes per chromosome (chr17, chr13): {(genes.chromosome == 'chr17').sum()}, "
        f"{(genes.chromosome == 'chr13').sum()}")

    torch.save({"patient_ids": pids, "sample_barcodes": sel.sample_barcode.tolist(), "file_ids": sel.file_id.tolist(),
                "genes": genes.reset_index(drop=True), "cn": cn, "workflow": "ASCAT2", "build": "GRCh38/GENCODE v36"},
               os.path.join(OUT_DIR, "brca_gene_cn_ascat2.pt"))
    sel.drop(columns="path").to_csv(os.path.join(OUT_DIR, "brca_gene_cn_sample_selection.csv"), index=False)

    m = genes.chromosome.isin(["chr17", "chr13"]).values
    sub = pd.DataFrame(cn[:, m], index=pd.Index(pids, name="patient_id"),
                       columns=(genes.gene_name[m] + "|" + genes.gene_id[m]).values)
    sub.to_csv(os.path.join(OUT_DIR, "brca_gene_cn_chr17_chr13.csv"))
    log(f"chr17+chr13 subset: {sub.shape}")

    key_rows = []
    for g in KEY_GENES:
        j = np.where(genes.gene_name.values == g)[0]
        if len(j) == 0:
            continue
        v = cn[:, j[0]]
        ok = ~np.isnan(v)
        key_rows.append({"gene": g, "chromosome": genes.chromosome.iloc[j[0]], "start": genes.start.iloc[j[0]],
                         "n_called": int(ok.sum()), "median_cn": np.nanmedian(v),
                         "frac_amp_ge6": (v[ok] >= 6).mean(), "frac_gain_3_5": ((v[ok] >= 3) & (v[ok] <= 5)).mean(),
                         "frac_loss_1": (v[ok] == 1).mean(), "frac_homdel_0": (v[ok] == 0).mean()})
    key = pd.DataFrame(key_rows)
    key.to_csv(os.path.join(OUT_DIR, "brca_key_genes_cn.csv"), index=False)
    log(key.round(3).to_string(index=False))
    with open(os.path.join(OUT_DIR, "build_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
