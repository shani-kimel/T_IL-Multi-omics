"""
Identify the TCGA-BRCA patients that TMO-Net left out of its released
dataset even though all four of its omics exist for them:
  - CNV (Genome_Wide_SNP_6), expression (IlluminaHiSeq_RNASeqV2) and
    methylation (HumanMethylation450 or 27): a PanCanAtlas tumor aliquot not
    flagged Do_not_use (merged_sample_quality_annotations.tsv)
  - mutation: a GDC Masked Somatic Mutation file (proxy for MC3)
These are the "test batch" for the frozen model: never in the development
cohort and without any missing-data or QC reason for exclusion.

Output: SeNMo_BRCA_CNV/data/gdc_cn_segments/complete_omics_excluded_brca.csv
        (patient_id, methylation_platform)
"""
import json
import os

import pandas as pd
import requests

ROOT = "/Users/yariv/Desktop/shani project"
SEG_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "gdc_cn_segments")
QUALITY = os.path.join(SEG_DIR, "pancan_quality.tsv")
QUALITY_URL = "https://api.gdc.cancer.gov/data/1a7d7be8-675d-4e60-a105-19d4121bdebf"  # merged_sample_quality_annotations.tsv
MANIFEST = os.path.join(ROOT, "SeNMo_BRCA_CNV", "minds", "brca_cohort", "manifest.json")
TMONET_CLIN = os.path.join(ROOT, "TCGA_PanCancer_Dataset", "TCGA_PanCaner_Cinical_Info_230109.csv")
OUT = os.path.join(SEG_DIR, "complete_omics_excluded_brca.csv")
KIND = {"Genome_Wide_SNP_6": "CNV", "IlluminaHiSeq_RNASeqV2": "EXPR",
        "HumanMethylation450": "METH", "HumanMethylation27": "METH"}


def main():
    if not os.path.exists(QUALITY):
        r = requests.get(QUALITY_URL, timeout=300)
        r.raise_for_status()
        with open(QUALITY, "wb") as f:
            f.write(r.content)
    q = pd.read_csv(QUALITY, sep="\t", low_memory=False)
    q = q[(q["cancer type"] == "BRCA") & q.platform.notna() & (q.Do_not_use == False)]  # noqa: E712
    q = q[q.aliquot_barcode.str[13:15].astype(int) < 10]  # tumor aliquots only
    plat = q.groupby("patient_barcode").platform.agg(set)

    clin = pd.read_csv(TMONET_CLIN)
    in_tmonet = set(clin[clin.Cancer_Type == "BRCA"].ID)
    with open(MANIFEST) as f:
        manifest = json.load(f)

    rows = []
    for p in manifest:
        pid = p["PatientID"]
        if pid in in_tmonet:
            continue
        s = plat.get(pid, set())
        kinds = {KIND[x] for x in s if x in KIND}
        if {"CNV", "EXPR", "METH"} <= kinds and p.get("Masked Somatic Mutation"):
            rows.append({"patient_id": pid,
                         "methylation_platform": "450K" if "HumanMethylation450" in s else "27K"})
    out = pd.DataFrame(rows).sort_values("patient_id")
    out.to_csv(OUT, index=False)
    print(f"GDC BRCA {len(manifest)}; not in TMO-Net {len(manifest) - len(in_tmonet & {p['PatientID'] for p in manifest})}; "
          f"complete omics yet excluded: {len(out)} {out.methylation_platform.value_counts().to_dict()}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
