"""
Pull TCGA-BRCA copy-number SEGMENT files (Affymetrix SNP6, DNAcopy, GRCh38)
for every tumor sample, both GDC variants:
  - "Copy Number Segment"        (*.grch38.seg.v2.txt)
  - "Masked Copy Number Segment" (*.nocnv_grch38.seg.v2.txt, germline CNVs removed)
Segment_Mean is a log2 ratio -- the same kind of value as the TMO-Net
7460-bin grid -- so these can be mapped onto that grid for the ~350 BRCA
patients the grid never included (see preprocessing/map_gdc_segments_to_grid.py).

Normal samples are excluded at the query level. Files are small (~5-50 KB)
and fetched in bulk requests (GDC drops many parallel single-file
connections). Resumable: files already on disk are skipped.

Outputs (SeNMo_BRCA_CNV/data/gdc_cn_segments/brca/):
  files.csv                    file_id, data_type, sample barcode/type, case
  {masked,unmasked}/<file_id>.seg.txt
"""
import io
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = "/Users/yariv/Desktop/shani project"
OUT = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_cn_segments", "brca")
GDC = "https://api.gdc.cancer.gov"
BATCH = 200
WORKERS = 3
SUBDIR = {"Copy Number Segment": "unmasked", "Masked Copy Number Segment": "masked"}


def query():
    filt = {"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "in", "content": {"field": "data_type", "value": list(SUBDIR)}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "DNAcopy"}}]}
    fields = "file_id,file_name,data_type,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"
    r = requests.post(f"{GDC}/files", json={"filters": filt, "fields": fields, "size": 10000}, timeout=180)
    r.raise_for_status()
    rows = []
    for h in r.json()["data"]["hits"]:
        s = h["cases"][0]["samples"]
        assert len(s) == 1, h  # SNP6 segment files are per sample
        rows.append({"file_id": h["file_id"], "file_name": h["file_name"], "data_type": h["data_type"],
                     "case_id": h["cases"][0]["submitter_id"], "sample_barcode": s[0]["submitter_id"],
                     "sample_type": s[0]["sample_type"]})
    df = pd.DataFrame(rows)
    return df[~df.sample_type.str.lower().str.contains("normal")].reset_index(drop=True)


def path_for(row):
    return os.path.join(OUT, SUBDIR[row.data_type], f"{row.file_id}.seg.txt")


def fetch(batch):
    ids = [r.file_id for r in batch]
    dest = {r.file_id: path_for(r) for r in batch}
    for attempt in range(5):
        try:
            resp = requests.post(f"{GDC}/data", json={"ids": ids}, stream=True, timeout=600)
            resp.raise_for_status()
            if len(ids) == 1:  # GDC returns a single file un-tarred
                content = resp.content
                with open(dest[ids[0]], "wb") as f:
                    f.write(content)
                return 0
            with tarfile.open(fileobj=resp.raw, mode="r|gz") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith(".txt"):
                        fid = m.name.split("/")[0]
                        if fid in dest:
                            with open(dest[fid], "wb") as f:
                                f.write(tf.extractfile(m).read())
            return sum(not os.path.exists(p) for p in dest.values())
        except Exception as e:  # noqa: BLE001 -- retry transient network errors
            print(f"  retry ({e})", flush=True)
            time.sleep(5 * (attempt + 1))
    return sum(not os.path.exists(p) for p in dest.values())


def main():
    for d in SUBDIR.values():
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    files = query()
    files.to_csv(os.path.join(OUT, "files.csv"), index=False)
    print(f"{len(files)} tumor segment files for {files.case_id.nunique()} cases: "
          f"{files.data_type.value_counts().to_dict()}; {files.sample_type.value_counts().to_dict()}", flush=True)
    todo = [r for r in files.itertuples() if not os.path.exists(path_for(r))]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(files) - len(todo)} on disk, {len(todo)} to download in {len(batches)} bulk requests", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        missing = sum(ex.map(fetch, batches))
    print(f"DONE; missing after download: {missing}", flush=True)


if __name__ == "__main__":
    main()
