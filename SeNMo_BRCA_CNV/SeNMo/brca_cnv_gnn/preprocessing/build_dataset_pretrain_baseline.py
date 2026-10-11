"""
Pretraining dataset (NO CNV, NO BRCA): builds a single pan-cancer training
set from every cancer type EXCEPT BRCA, using Expression + Methylation +
Mutation (same blocks/feature space as the BRCA_noCNV fine-tuning data, so
the pretrained encoder's input layer matches).

BRCA is excluded entirely from pretraining -- on purpose -- so that when we
later fine-tune per-fold on BRCA, none of that fold's held-out BRCA test
patients (or even the other folds' BRCA train patients) have been seen
during pretraining. This keeps the per-fold BRCA test C-index honest.

Does NOT read or reference the CNA/CNV file (same isolation rule as
build_dataset_baseline.py).

Output: SeNMo_BRCA_CNV/data/PanCancer_noBRCA_noCNV/train/omic_10cv_fold_1.pkl
(a single "fold" -- this is one pretraining run, not a 5-fold CV).
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "PanCancer_noBRCA_noCNV", "train")

EXCLUDE_CANCER_TYPE = "BRCA"
MONITOR_FOLD = 1  # which existing fold assignment to use as the train/val split for monitoring pretraining

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

    train_ids = train_index[
        (train_index["Cancer_Type"] != EXCLUDE_CANCER_TYPE) & (train_index["Fold"] == MONITOR_FOLD)
    ]["Sample_ID"].tolist()
    test_ids = test_index[
        (test_index["Cancer_Type"] != EXCLUDE_CANCER_TYPE) & (test_index["Fold"] == MONITOR_FOLD)
    ]["Sample_ID"].tolist()

    print(f"Pretraining set (non-BRCA, fold {MONITOR_FOLD} split): train={len(train_ids)} test={len(test_ids)}")
    assert EXCLUDE_CANCER_TYPE not in clinical.loc[train_ids]["Cancer_Type"].unique()
    assert EXCLUDE_CANCER_TYPE not in clinical.loc[test_ids]["Cancer_Type"].unique()

    omics = {name: load_omics(name) for name in CONTINUOUS_BLOCKS + BINARY_BLOCKS}
    block_order = CONTINUOUS_BLOCKS + BINARY_BLOCKS
    total_dim = sum(omics[b].shape[1] for b in block_order)
    print(f"Pretrain blocks: {block_order}, total feature dim = {total_dim}")

    for name, df in omics.items():
        missing = (set(train_ids) | set(test_ids)) - set(df.index)
        if missing:
            raise ValueError(f"{len(missing)} patients missing from {name}: {sorted(missing)[:5]}...")

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

    data_cv = {"cv_splits": {1: {"train": train_split, "test": test_split}}}
    out_path = os.path.join(OUT_DIR, "omic_10cv_fold_1.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data_cv, f, protocol=4)

    print(f"Wrote {out_path} (train={len(train_ids)} test={len(test_ids)} dim={total_dim})")


if __name__ == "__main__":
    main()
