"""
Builds the combined dataset for the embeddings-based experiment:
  - Stage 1 (no CNV): the 48-dim precomputed SeNMo embedding (from the
    ACTUAL pretrained SeNMo model, released as
    SeNMo_checkpoints/Molecular (SeNMo).parquet) for each BRCA patient.
  - Stage 2 (+CNV): that same embedding concatenated with the patient's
    CNA features (7460 dims, from the local cleaned pan-cancer dataset).

Restricts to the intersection of: (a) BRCA patients in the local cleaned
clinical/CNA data, (b) patients with a precomputed embedding in the SeNMo
parquet, so Stage 1 and Stage 2 are compared on the identical patient set.
Keeps the existing 5-fold split (train_data.csv/test_data.csv) as-is.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna.pkl
  -- a dict with keys: patient_ids, embedding (Nx48), cna (Nx7460), e, t, fold
"""
import os
import pickle

import numpy as np
import pandas as pd

ROOT = "/Users/yariv/Desktop/shani project"
CLEANED_DIR = os.path.join(ROOT, "TMO-Net", "data", "TCGA_PanCancer_Data_cleaned")
PARQUET_PATH = os.path.join(ROOT, "SeNMo_checkpoints", "Molecular (SeNMo).parquet")
OUT_DIR = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment")

CANCER_TYPE = "BRCA"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    clinical = pd.read_csv(os.path.join(CLEANED_DIR, "cleaned_clinical_info.csv")).set_index("ID", drop=False)
    cna = pd.read_csv(os.path.join(CLEANED_DIR, "TCGA_PanCancerAtlas_CNA_230109_modified.csv")).set_index("ID").astype("float32")

    train_index = pd.read_csv(os.path.join(CLEANED_DIR, "train_data.csv"))
    test_index = pd.read_csv(os.path.join(CLEANED_DIR, "test_data.csv"))
    # test_data.csv has exactly one row per patient per fold-they're-held-out-in;
    # build a single Sample_ID -> Fold (held-out fold) lookup from it.
    fold_lookup = test_index.set_index("Sample_ID")["Fold"].to_dict()

    brca_ids_local = set(clinical[clinical["Cancer_Type"] == CANCER_TYPE]["ID"])
    print(f"Local BRCA patients: {len(brca_ids_local)}")

    emb_df = pd.read_parquet(PARQUET_PATH)
    emb_df["ID"] = emb_df["PatientID"].astype(str).str[:12]
    emb_df = emb_df.drop_duplicates(subset="ID").set_index("ID")

    overlap = sorted(brca_ids_local & set(emb_df.index) & set(cna.index) & set(fold_lookup.keys()))
    print(f"Patients with embedding + CNA + fold assignment: {len(overlap)} "
          f"(dropped {len(brca_ids_local) - len(overlap)} missing one of these)")

    embedding = np.stack([np.asarray(emb_df.loc[pid, "Embeddings"], dtype=np.float32) for pid in overlap])
    cna_matrix = cna.loc[overlap].values.astype(np.float32)
    e = clinical.loc[overlap, "OS"].astype(np.float32).values
    t = clinical.loc[overlap, "OS.time"].astype(np.float32).values
    fold = np.array([fold_lookup[pid] for pid in overlap], dtype=int)

    assert embedding.shape == (len(overlap), 48)
    assert cna_matrix.shape[1] == 7460
    assert not np.isnan(embedding).any()
    assert not np.isnan(cna_matrix).any()
    assert not np.isnan(e).any() and not np.isnan(t).any()

    print("Fold distribution (each patient's held-out test fold):")
    print(pd.Series(fold).value_counts().sort_index())

    data = {
        "patient_ids": overlap,
        "embedding": embedding,
        "cna": cna_matrix,
        "e": e,
        "t": t,
        "fold": fold,
    }
    out_path = os.path.join(OUT_DIR, "brca_embedding_cna.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
