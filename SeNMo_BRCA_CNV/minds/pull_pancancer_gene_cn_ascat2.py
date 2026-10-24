"""
Pull genome-wide GDC Gene Level Copy Number (ASCAT2, GRCh38 / GENCODE v36)
for every non-BRCA TCGA case -- the pan-cancer pretraining pool counterpart
of SeNMo_BRCA_CNV/data/gdc_gene_cn/brca_gene_cn_ascat2.pt. BRCA is excluded
at the query level, so no BRCA patient can enter pretraining.

One tumor sample per case: primary tumor (-01) or primary blood-derived
cancer (-03) preferred, otherwise the case's other tumor sample (e.g. SKCM
metastatic -06); ties broken by lowest sample barcode.

Disk-light: files are fetched in 50-file bulk requests (GDC drops many
parallel single-file connections), each ~3.4 MB TSV is streamed, its gene order checked against the
BRCA reference, and only the copy_number column is kept (float16, ~120 KB)
in cache/<file_id>.npy; the TSV is never written to disk. Resumable: cached
files are skipped on re-run. A final pass assembles one matrix.

Outputs (SeNMo_BRCA_CNV/data/gdc_gene_cn/pancancer_nonbrca/):
  selected_files.csv                 case, project, sample barcode/type, file_id
  cache/<file_id>.npy                per-file copy_number vectors
  pancancer_nonbrca_gene_cn_ascat2.npy   (n_cases x 60623) float16, rows = selected_files.csv order
  genes.csv                          gene_id, gene_name, chromosome, start, end
  pull_log.txt
"""
import io
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import torch

ROOT = "/Users/yariv/Desktop/shani project"
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_gene_cn", "pancancer_nonbrca")
CACHE = os.path.join(OUT_DIR, "cache")
BRCA_REF = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_gene_cn", "brca_gene_cn_ascat2.pt")
GDC = "https://api.gdc.cancer.gov"
WORKERS = 3   # parallel bulk requests
BATCH = 50    # files per bulk request
LOG_PATH = os.path.join(OUT_DIR, "pull_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def query_files():
    filt = {"op": "and", "content": [
        {"op": "=", "content": {"field": "data_type", "value": "Gene Level Copy Number"}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "ASCAT2"}},
        {"op": "=", "content": {"field": "cases.project.program.name", "value": "TCGA"}},
        {"op": "!=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}}]}
    fields = "file_id,file_size,cases.submitter_id,cases.project.project_id,cases.samples.submitter_id,cases.samples.sample_type"
    hits, start = [], 0
    while True:
        r = requests.post(f"{GDC}/files", json={"filters": filt, "fields": fields, "size": 2000, "from": start},
                          timeout=180)
        r.raise_for_status()
        d = r.json()["data"]
        hits += d["hits"]
        start += 2000
        if start >= d["pagination"]["total"]:
            return hits


def select(hits):
    rows = []
    for h in hits:
        c = h["cases"][0]
        tumors = sorted((s for s in c["samples"] if "normal" not in s["sample_type"].lower()),
                        key=lambda s: s["submitter_id"])
        if not tumors:
            continue
        t = tumors[0]
        rows.append({"case_id": c["submitter_id"], "project": c["project"]["project_id"],
                     "sample_barcode": t["submitter_id"], "sample_type": t["sample_type"],
                     "primary": t["submitter_id"][13:15] in ("01", "03"),
                     "file_id": h["file_id"], "file_size": h["file_size"]})
    df = pd.DataFrame(rows)
    assert not (df.project == "TCGA-BRCA").any()
    df = df.sort_values(["case_id", "primary", "sample_barcode", "file_id"], ascending=[True, False, True, True])
    return df.drop_duplicates("case_id").reset_index(drop=True)


def fetch_batch(file_ids, ref_gene_ids):
    """One bulk POST /data request (GDC returns a tar.gz stream of <file_id>/<name>.tsv);
    GDC drops connections when many single-file requests run in parallel from one IP."""
    status = {f: "missing_from_bundle" for f in file_ids}
    for attempt in range(5):
        try:
            r = requests.post(f"{GDC}/data", json={"ids": file_ids}, stream=True, timeout=600)
            r.raise_for_status()
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                for m in tf:
                    if not (m.isfile() and m.name.endswith(".tsv")):
                        continue
                    fid = m.name.split("/")[0]
                    df = pd.read_csv(io.BytesIO(tf.extractfile(m).read()), sep="\t",
                                     usecols=["gene_id", "copy_number"])
                    if not (df.gene_id.values == ref_gene_ids).all():
                        status[fid] = "gene_order_mismatch"
                        continue
                    out = os.path.join(CACHE, f"{fid}.npy")
                    tmp = out + ".tmp.npy"
                    np.save(tmp, df.copy_number.values.astype(np.float16))
                    os.replace(tmp, out)
                    status[fid] = "ok"
            return status
        except Exception as e:  # noqa: BLE001 -- retry transient network/stream errors
            if attempt == 4:
                return {f: (s if s == "ok" else f"error: {e}") for f, s in status.items()}
            file_ids = [f for f in file_ids if status[f] != "ok"]
            time.sleep(5 * (attempt + 1))


def main():
    os.makedirs(CACHE, exist_ok=True)
    ref = torch.load(BRCA_REF, weights_only=False)["genes"]
    ref_ids = ref.gene_id.values
    ref.to_csv(os.path.join(OUT_DIR, "genes.csv"), index=False)

    sel_path = os.path.join(OUT_DIR, "selected_files.csv")
    if os.path.exists(sel_path):
        sel = pd.read_csv(sel_path)
    else:
        hits = query_files()
        sel = select(hits)
        sel.to_csv(sel_path, index=False)
        log(f"GDC: {len(hits)} non-BRCA ASCAT2 files -> {len(sel)} cases selected "
            f"({sel.primary.mean():.1%} primary); {sel.project.nunique()} projects; "
            f"~{sel.file_size.sum() / 1e9:.1f} GB to stream")
        log(f"sample types: {sel.sample_type.value_counts().to_dict()}")

    todo = [f for f in sel.file_id if not os.path.exists(os.path.join(CACHE, f"{f}.npy"))]
    log(f"{len(sel) - len(todo)} cached, {len(todo)} to download")
    status = {}
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_batch, b, ref_ids) for b in batches]
        for i, fut in enumerate(as_completed(futs), 1):
            status.update(fut.result())
            if i % 5 == 0 or i == len(batches):
                bad = sum(1 for s in status.values() if s != "ok")
                log(f"progress {len(status)}/{len(todo)} files  failed so far: {bad}")
    failed = {k: v for k, v in status.items() if v not in ("ok", "cached")}
    if failed:
        log(f"FAILED {len(failed)} files (re-run to retry): {list(failed.items())[:5]}")
        sys.exit(1)

    mat = np.stack([np.load(os.path.join(CACHE, f"{f}.npy")) for f in sel.file_id])
    np.save(os.path.join(OUT_DIR, "pancancer_nonbrca_gene_cn_ascat2.npy"), mat)
    log(f"DONE matrix {mat.shape} float16; missing {np.isnan(mat.astype(np.float32)).mean():.2%}")


if __name__ == "__main__":
    main()
