"""
Cohort-level feature matrices for Protein (RPPA) and miRNA, built jointly
across all downloaded patients (not per-patient in isolation, unlike
SeNMo's reference package_classes/*_preprocess.py scripts, which reduce
dimensionality via per-column "all values below threshold" rules that only
make sense applied to a whole cohort matrix at once -- see plan notes).

Scans one or more MINDS download roots (each containing raw/{PatientID}/...)
and produces two CSVs, indexed by the same 12-char TCGA barcode 'ID'
convention used everywhere else in this project:
  SeNMo_BRCA_CNV/data/raw_features/protein_features.csv
  SeNMo_BRCA_CNV/data/raw_features/mirna_features.csv

A patient with multiple RPPA/miRNA files (multiple aliquots/samples) has
their values averaged. Columns present for too few patients are dropped;
remaining gaps are mean-imputed (mirrors SeNMo's stated tolerance for
missing modalities, done consistently at the cohort level instead of
ad hoc per patient).
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV"
RAW_ROOTS = [
    os.path.join(ROOT, "minds", "brca_cohort", "raw"),
    os.path.join(ROOT, "minds", "pancancer_nonbrca_cohort", "raw"),
]
OUT_DIR = os.path.join(ROOT, "data", "raw_features")

MIN_COVERAGE = 0.70  # drop a feature column present for fewer than this fraction of patients


def collect_patient_dirs():
    patient_dirs = {}
    for raw_root in RAW_ROOTS:
        if not os.path.isdir(raw_root):
            print(f"WARNING: raw root not found, skipping: {raw_root}")
            continue
        for patient_id in os.listdir(raw_root):
            full = os.path.join(raw_root, patient_id)
            if os.path.isdir(full):
                patient_dirs.setdefault(patient_id, []).append(full)
    return patient_dirs


def parse_rppa(patient_dir):
    files = glob.glob(os.path.join(patient_dir, "Protein Expression Quantification", "*", "*_RPPA_data.tsv"))
    if not files:
        return None
    series_list = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        s = df.set_index("AGID")["protein_expression"]
        series_list.append(s)
    combined = pd.concat(series_list, axis=1).astype(float).mean(axis=1)
    return combined


def parse_mirna(patient_dir):
    files = glob.glob(os.path.join(patient_dir, "miRNA Expression Quantification", "*", "*.mirnas.quantification.txt"))
    if not files:
        return None
    series_list = []
    for f in files:
        df = pd.read_csv(f, sep="\t")
        s = df.set_index("miRNA_ID")["reads_per_million_miRNA_mapped"]
        series_list.append(s)
    combined = pd.concat(series_list, axis=1).astype(float).mean(axis=1)
    return np.log1p(combined)  # standard log1p transform for skewed RPM counts


def build_matrix(patient_dirs, parse_fn, label):
    rows = {}
    for patient_id, dirs in patient_dirs.items():
        # a patient could in principle appear under more than one raw root; use the first that has data
        for d in dirs:
            series = parse_fn(d)
            if series is not None:
                rows[patient_id] = series
                break
    print(f"{label}: {len(rows)}/{len(patient_dirs)} patients have data")
    matrix = pd.DataFrame(rows).T  # patients x features
    matrix.index.name = "ID"

    coverage = matrix.notna().mean(axis=0)
    keep_cols = coverage[coverage >= MIN_COVERAGE].index
    dropped = matrix.shape[1] - len(keep_cols)
    matrix = matrix[keep_cols]
    print(f"{label}: kept {matrix.shape[1]} features (dropped {dropped} below {MIN_COVERAGE:.0%} coverage)")

    # mean-impute remaining gaps (patients missing this specific feature, or missing the whole modality)
    matrix = matrix.fillna(matrix.mean(axis=0))
    n_fully_missing = matrix.isna().all(axis=1).sum()
    if n_fully_missing:
        print(f"{label}: {n_fully_missing} patients have zero coverage even after imputation (will be dropped)")
        matrix = matrix.dropna(how="all")
    return matrix.astype("float32")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patient_dirs = collect_patient_dirs()
    print(f"Found {len(patient_dirs)} patient directories across {len(RAW_ROOTS)} raw root(s)")

    protein = build_matrix(patient_dirs, parse_rppa, "Protein (RPPA)")
    protein.to_csv(os.path.join(OUT_DIR, "protein_features.csv"))
    print(f"Wrote protein_features.csv: {protein.shape}")

    mirna = build_matrix(patient_dirs, parse_mirna, "miRNA")
    mirna.to_csv(os.path.join(OUT_DIR, "mirna_features.csv"))
    print(f"Wrote mirna_features.csv: {mirna.shape}")


if __name__ == "__main__":
    main()
