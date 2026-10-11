"""
Stage 1 (baseline, NO CNV): builds the SeNMo-format BRCA dataset from
Expression + Methylation + Mutation only.

This script intentionally does NOT read, import, or reference the CNA/CNV
file. CNV is added in a separate script (build_dataset_cnv.py) once this
baseline is validated, so the two configs stay physically isolated.

Reuses (does not regenerate):
- TMO-Net/data/TCGA_PanCancer_Data_cleaned/train_data.csv, test_data.csv
  (the existing stratified 5-fold CV split, seed 66)
- TMO-Net/data/TCGA_PanCancer_Data_cleaned/{Expression,Methylation,Mutation}
  _230109_modified.csv (already ID-normalized to 12-char TCGA barcodes)
- TMO-Net/data/TCGA_PanCancer_Data_cleaned/cleaned_clinical_info.csv (OS, OS.time)

Output: for each fold 1..5, a pickle at
  SeNMo_BRCA_CNV/data/BRCA_noCNV/train/omic_10cv_fold_{fold}.pkl
matching SeNMo_Training.py's expected structure:
  {'cv_splits': {fold: {'train': {x_patname, x_omic, e, t},
                         'test':  {x_patname, x_omic, e, t}}}}
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "BRCA_noCNV", "train")

CANCER_TYPE = "BRCA"
NUM_FOLDS = 5

# Blocks for the NO-CNV baseline. Continuous blocks get standardized
# (fit on train rows of each fold only); mutation is binary, left as-is.
CONTINUOUS_BLOCKS = ["Expression", "Methylation"]
BINARY_BLOCKS = ["Mutation"]


def load_omics(name):
    path = os.path.join(CLEANED_DIR, f"TCGA_PanCancerAtlas_{name}_230109_modified.csv")
    df = pd.read_csv(path)
    df = df.set_index("ID")
    return df.astype("float32")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    clinical = pd.read_csv(os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv"))
    clinical = clinical.set_index("ID", drop=False)

    train_index = pd.read_csv(os.path.join(CLEANED_DIR, "train_data.csv"))
    test_index = pd.read_csv(os.path.join(CLEANED_DIR, "test_data.csv"))

    omics = {name: load_omics(name) for name in CONTINUOUS_BLOCKS + BINARY_BLOCKS}
    block_order = CONTINUOUS_BLOCKS + BINARY_BLOCKS
    total_dim = sum(omics[b].shape[1] for b in block_order)
    print(f"Baseline (no CNV) blocks: {block_order}, total feature dim = {total_dim}")

    for fold in range(1, NUM_FOLDS + 1):
        train_ids = train_index[
            (train_index["Cancer_Type"] == CANCER_TYPE) & (train_index["Fold"] == fold)
        ]["Sample_ID"].tolist()
        test_ids = test_index[
            (test_index["Cancer_Type"] == CANCER_TYPE) & (test_index["Fold"] == fold)
        ]["Sample_ID"].tolist()

        # Sanity: every patient must have a row in every omics block and in clinical.
        for name, df in omics.items():
            missing = (set(train_ids) | set(test_ids)) - set(df.index)
            if missing:
                raise ValueError(f"Fold {fold}: {len(missing)} BRCA patients missing from {name}: {sorted(missing)[:5]}...")

        def build_split(ids, scalers=None, fit=False):
            blocks = []
            fitted_scalers = {} if fit else scalers
            for name in CONTINUOUS_BLOCKS:
                X = omics[name].loc[ids].values
                if fit:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    fitted_scalers[name] = scaler
                else:
                    X = scalers[name].transform(X)
                blocks.append(X)
            for name in BINARY_BLOCKS:
                blocks.append(omics[name].loc[ids].values)
            X_omic = np.concatenate(blocks, axis=1).astype(np.float32)

            clin = clinical.loc[ids]
            e = clin["OS"].astype(np.float32).values
            t = clin["OS.time"].astype(np.float32).values

            if np.isnan(X_omic).any():
                raise ValueError("NaNs found in combined omics matrix")
            if np.isnan(e).any() or np.isnan(t).any():
                raise ValueError("NaNs found in survival labels (OS/OS.time)")

            return {"x_patname": list(ids), "x_omic": [row for row in X_omic], "e": list(e), "t": list(t)}, fitted_scalers

        train_split, scalers = build_split(train_ids, fit=True)
        test_split, _ = build_split(test_ids, scalers=scalers, fit=False)

        assert train_split["x_omic"][0].shape[0] == total_dim
        assert test_split["x_omic"][0].shape[0] == total_dim

        data_cv = {"cv_splits": {fold: {"train": train_split, "test": test_split}}}
        out_path = os.path.join(OUT_DIR, f"omic_10cv_fold_{fold}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data_cv, f, protocol=4)

        print(
            f"Fold {fold}: train={len(train_ids)} test={len(test_ids)} "
            f"dim={total_dim} -> {out_path}"
        )

    print("Done. Total feature dim (no CNV):", total_dim)


if __name__ == "__main__":
    main()
