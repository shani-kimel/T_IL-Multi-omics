"""
Fetches real gene coordinates (GRCh37/hg19 -- TCGA PanCanAtlas convention)
for the 7 contiguous genomic runs covered by the CNA bins, via Ensembl's
GRCh37 REST API (grch37.rest.ensembl.org/overlap/region). Confirmed
reachable and fast (~0.4s/query) in this environment, unlike string-db.org
directly (timed out) -- Ensembl's overlap endpoint caps each request at 5Mb,
so each run is queried in <=5Mb chunks and results are deduplicated by
Ensembl gene ID.

NOTE on genome build: nothing in this project's files states which build
the CNA bin coordinates use. TCGA PanCanAtlas conventionally used GRCh37/
hg19, and a spot-check (MYC found at chr8:128.7-128.8Mb, matching hg19)
is consistent with that, so GRCh37 is used here. If it later turns out to
be GRCh38, gene-to-bin assignments would shift slightly (mostly at region
edges) but the overall approach doesn't change.

Restricts to protein_coding genes only (drop lncRNA/pseudogene/etc.) --
that's what has PPI/pathway annotations downstream anyway, and keeps the
node count from being dominated by non-coding transcripts.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/gene_annotations.csv
  columns: gene_id, external_name, chrom, start, end, strand
"""
import os
import time

import pandas as pd
import requests
import torch

ROOT = "/Users/yariv/Desktop/shani project"
GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_annotations.csv")

ENSEMBL_URL = "https://grch37.rest.ensembl.org/overlap/region/human/{region}"
CHUNK_SIZE = 4_900_000  # stay under Ensembl's 5,000,000 bp cap


def get_runs():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    run_id = graph["run_id"].numpy()
    bin_columns = graph["bin_columns"]
    import re
    bin_re = re.compile(r"\d+_chr([0-9XY]+)_(\d+)to(\d+)_10k")
    parsed = [bin_re.match(c).groups() for c in bin_columns]

    runs = []
    for r in range(graph["n_runs"]):
        idx = [i for i, x in enumerate(run_id) if x == r]
        chrom = parsed[idx[0]][0]
        start_bp = int(parsed[idx[0]][1]) * 10000
        end_bp = int(parsed[idx[-1]][2]) * 10000
        runs.append((chrom, start_bp, end_bp))
    return runs


def fetch_region(chrom, start, end, retries=3):
    url = ENSEMBL_URL.format(region=f"{chrom}:{start}-{end}")
    params = {"feature": "gene", "content-type": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry ({e})")
            time.sleep(1)


def main():
    runs = get_runs()
    genes = {}
    for chrom, run_start, run_end in runs:
        chunk_start = run_start
        while chunk_start < run_end:
            chunk_end = min(chunk_start + CHUNK_SIZE, run_end)
            print(f"querying chr{chrom}:{chunk_start:,}-{chunk_end:,} ...")
            result = fetch_region(chrom, chunk_start, chunk_end)
            for g in result:
                if g.get("biotype") != "protein_coding":
                    continue
                gid = g["gene_id"]
                genes[gid] = {
                    "gene_id": gid,
                    "external_name": g.get("external_name", gid),
                    "chrom": chrom,
                    "start": g["start"],
                    "end": g["end"],
                    "strand": g["strand"],
                }
            chunk_start = chunk_end
            time.sleep(0.1)  # be polite to the public API

    df = pd.DataFrame(genes.values()).sort_values(["chrom", "start"]).reset_index(drop=True)
    print(f"\n{len(df)} protein-coding genes found across {len(runs)} runs")
    print(df["chrom"].value_counts())
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
