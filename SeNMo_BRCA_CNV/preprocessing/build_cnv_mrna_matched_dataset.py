"""
Builds a gene-level CNV + mRNA matched BRCA dataset from the raw GDC pull in
minds/brca_cohort/raw/ (Gene Level Copy Number + Gene Expression
Quantification), instead of the pre-curated TCGA_PanCancer_Dataset that
build_embedding_dataset.py uses -- that resource is capped at 758 BRCA
patients because it requires all 4 omics (CNA+expression+methylation+
mutation) simultaneously. This pipeline only needs CNV+mRNA, so it recovers
close to the full GDC BRCA cohort (~1089 patients with both).

Per patient, Gene Level Copy Number has up to 3 pipeline variants (ascat3,
absolute_liftover, and an unlabeled legacy one); ascat3 is GDC's current
default pipeline and is used preferentially, falling back to the other two
for the patients missing an ascat3 file. When a patient has multiple Gene
Expression Quantification files (technical replicates), the largest is used.

The feature set is the intersection of protein-coding genes (by GENCODE v36
Ensembl gene_id, derived from the expression files' gene_type column) present
in both CNV and expression data -- this is the gene universe a CNV<->mRNA
cross-modal model would train on.

For downstream survival evaluation this larger cohort is cross-referenced
against (a) the existing 758-patient clinical/fold assignment
(TMO-Net/data/TCGA_PanCancer_Data_cleaned) for OS/OS.time + fold, and (b) the
released SeNMo pretrained embedding parquet -- both are None/NaN for patients
outside those smaller sets, since a CNV<->mRNA VAE can still be pretrained
unsupervised on the larger set even where survival labels or SeNMo's
embedding aren't available.

Output: SeNMo_BRCA_CNV/data/cnv_mrna_matched/brca_cnv_mrna_matched.pkl
  -- dict with keys:
     patient_ids (list[str], len N)
     gene_ids, gene_names (list[str], len G -- shared CNV/mRNA gene order)
     cnv (N x G float32, copy_number; NaN where that gene/patient is missing)
     mrna (N x G float32, log1p(tpm_unstranded))
     cnv_pipeline_used (list[str], which of ascat3/absolute_liftover/legacy
       was used per patient, for auditing)
     os, os_time (N float32, NaN where no label -- from the 758-patient set)
     fold (N float32, NaN where not in the existing 5-fold split)
     senmo_embedding (N x 48 float32, NaN row where not in the SeNMo parquet)
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

ROOT = "/Users/yariv/Desktop/shani project"
RAW_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "minds", "brca_cohort", "raw")
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
PARQUET_PATH = os.path.join(ROOT, "SeNMo_checkpoints", "Molecular (SeNMo).parquet")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "cnv_mrna_matched")
OUT_PATH = os.path.join(OUT_DIR, "brca_cnv_mrna_matched.pkl")

CNV_MODALITY = "Gene Level Copy Number"
EXPR_MODALITY = "Gene Expression Quantification"
CNV_PIPELINE_PRIORITY = ["ascat3", "absolute_liftover", "legacy"]

EXPR_COLUMNS = [
    "gene_id", "gene_name", "gene_type", "unstranded", "stranded_first",
    "stranded_second", "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded",
]


def find_cnv_file(patient_dir):
    """Pick one Gene Level Copy Number file per patient: ascat3 preferred,
    then absolute_liftover, then the unlabeled legacy pipeline."""
    files = glob.glob(os.path.join(patient_dir, CNV_MODALITY, "*", "*.tsv"))
    if not files:
        return None, None
    by_pipeline = {}
    for f in files:
        name = os.path.basename(f)
        if "ascat3" in name:
            by_pipeline.setdefault("ascat3", f)
        elif "absolute_liftover" in name:
            by_pipeline.setdefault("absolute_liftover", f)
        else:
            by_pipeline.setdefault("legacy", f)
    for pipeline in CNV_PIPELINE_PRIORITY:
        if pipeline in by_pipeline:
            return by_pipeline[pipeline], pipeline
    return None, None


def find_expr_file(patient_dir):
    """Pick the largest Gene Expression Quantification file (technical
    replicates are near-identical; size is a simple, deterministic tiebreak)."""
    files = glob.glob(os.path.join(patient_dir, EXPR_MODALITY, "*", "*.tsv"))
    if not files:
        return None
    return max(files, key=os.path.getsize)


def load_cnv(path):
    df = pd.read_csv(path, sep="\t")
    return df.set_index("gene_id")["copy_number"]


def load_expr(path):
    df = pd.read_csv(path, sep="\t", skiprows=6, header=None, names=EXPR_COLUMNS)
    return df.set_index("gene_id")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patient_ids_all = sorted(os.listdir(RAW_DIR))

    # Canonical protein-coding gene universe: derived once from the first
    # patient with an expression file (GENCODE v36 gene model is shared
    # across all patients pulled in this cohort).
    canonical_genes = None
    for pid in patient_ids_all:
        expr_path = find_expr_file(os.path.join(RAW_DIR, pid))
        if expr_path is None:
            continue
        expr_df = load_expr(expr_path)
        canonical_genes = expr_df.index[expr_df["gene_type"] == "protein_coding"]
        canonical_gene_names = expr_df.loc[canonical_genes, "gene_name"]
        break
    if canonical_genes is None:
        raise RuntimeError("No expression files found under " + RAW_DIR)
    print(f"Canonical protein-coding gene universe: {len(canonical_genes)} genes")

    patient_ids, cnv_rows, mrna_rows, pipeline_used = [], [], [], []
    n_skipped_no_cnv, n_skipped_no_expr = 0, 0

    for pid in patient_ids_all:
        pdir = os.path.join(RAW_DIR, pid)
        cnv_path, pipeline = find_cnv_file(pdir)
        expr_path = find_expr_file(pdir)
        if cnv_path is None:
            n_skipped_no_cnv += 1
            continue
        if expr_path is None:
            n_skipped_no_expr += 1
            continue

        cnv_series = load_cnv(cnv_path).reindex(canonical_genes)
        expr_series = load_expr(expr_path)["tpm_unstranded"].reindex(canonical_genes)

        patient_ids.append(pid)
        cnv_rows.append(cnv_series.values.astype(np.float32))
        mrna_rows.append(np.log1p(expr_series.values.astype(np.float32)))
        pipeline_used.append(pipeline)

    print(f"Matched patients (CNV + mRNA): {len(patient_ids)} "
          f"(skipped {n_skipped_no_cnv} missing CNV, {n_skipped_no_expr} missing expression)")

    cnv = np.stack(cnv_rows)
    mrna = np.stack(mrna_rows)
    print(f"cnv matrix: {cnv.shape}, missing values: {np.isnan(cnv).sum()} "
          f"({np.isnan(cnv).mean():.2%})")
    print(f"mrna matrix: {mrna.shape}, missing values: {np.isnan(mrna).sum()} "
          f"({np.isnan(mrna).mean():.2%})")

    # Cross-reference with the existing 758-patient clinical/fold set and the
    # SeNMo pretrained embedding, for the subset usable in a supervised
    # survival comparison. NaN/None for everyone else -- still usable for
    # unsupervised CNV<->mRNA pretraining.
    clinical = pd.read_csv(os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")).set_index("ID", drop=False)
    test_index = pd.read_csv(os.path.join(CLEANED_DIR, "test_data.csv"))
    fold_lookup = test_index.set_index("Sample_ID")["Fold"].to_dict()

    emb_df = pd.read_parquet(PARQUET_PATH)
    emb_df["ID"] = emb_df["PatientID"].astype(str).str[:12]
    emb_df = emb_df.drop_duplicates(subset="ID").set_index("ID")

    os_arr = np.full(len(patient_ids), np.nan, dtype=np.float32)
    os_time_arr = np.full(len(patient_ids), np.nan, dtype=np.float32)
    fold_arr = np.full(len(patient_ids), np.nan, dtype=np.float32)
    senmo_emb_arr = np.full((len(patient_ids), 48), np.nan, dtype=np.float32)

    n_with_labels, n_with_fold, n_with_senmo = 0, 0, 0
    for i, pid in enumerate(patient_ids):
        if pid in clinical.index:
            row = clinical.loc[pid]
            if pd.notna(row["OS"]) and pd.notna(row["OS.time"]):
                os_arr[i] = row["OS"]
                os_time_arr[i] = row["OS.time"]
                n_with_labels += 1
        if pid in fold_lookup:
            fold_arr[i] = fold_lookup[pid]
            n_with_fold += 1
        if pid in emb_df.index:
            senmo_emb_arr[i] = np.asarray(emb_df.loc[pid, "Embeddings"], dtype=np.float32)
            n_with_senmo += 1

    print(f"Of {len(patient_ids)} matched patients: {n_with_labels} have OS/OS.time, "
          f"{n_with_fold} have a fold assignment, {n_with_senmo} have a SeNMo embedding")

    data = {
        "patient_ids": patient_ids,
        "gene_ids": list(canonical_genes),
        "gene_names": list(canonical_gene_names),
        "cnv": cnv,
        "mrna": mrna,
        "cnv_pipeline_used": pipeline_used,
        "os": os_arr,
        "os_time": os_time_arr,
        "fold": fold_arr,
        "senmo_embedding": senmo_emb_arr,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
