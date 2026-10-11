"""
Pull Gene Level Copy Number + Gene Expression Quantification for the existing
BRCA cohort (brca_cohort/manifest.json already lists these modalities for
1094/1095 of its 1098 patients -- only miRNA + Protein were ever actually
downloaded for this cohort, so this fetches the two modalities needed for a
CNV<->mRNA cross-modal model on close to the full GDC BRCA cohort, instead of
the 758-patient intersection the project has been limited to so far).

Reuses brca_cohort/manifest.json as-is (no DB query, no new manifest build).
Same multi-pass, collision-safe download/organize logic as
pull_pancancer_protein_mirna.py, since a single GDC API pass reliably drops
a fraction of requests under load.

Resumable: safe to re-run: patients whose files are already organized under
raw/<patient>/<data_type>/ are skipped on subsequent passes.
"""
import os
import shutil

from med_minds.downloader import GDCFileDownloader

OUTPUT_DIR = "./brca_cohort"
INCLUDE_MODALITIES = ["Gene Level Copy Number", "Gene Expression Quantification"]
DOWNLOAD_PASSES = 3
MAX_WORKERS = 4


def organize_safely(d, output_dir):
    """Same logic as GDCFileDownloader.organize_files(), but skips (rather than
    crashes on) a destination file that already exists from a prior pass."""
    moved, skipped = 0, 0
    for patient in d.manifest:
        patient_id = patient["PatientID"]
        for data_type, files in patient.items():
            if data_type in ("PatientID", "gdc_case_id"):
                continue
            if not any(isinstance(f, dict) and "id" in f for f in files):
                continue
            if d.include and data_type not in d.include:
                continue
            if d.exclude and data_type in d.exclude:
                continue
            for file in files:
                file_uuid = file["id"]
                file_name = file["file_name"]
                target_dir = os.path.join(output_dir, "raw", patient_id, data_type, file_uuid)
                dest_path = os.path.join(target_dir, file_name)
                source_path = os.path.join(output_dir, file_name)
                # Extracted tar.gz bundles land each file's contents in a
                # <file_uuid>/ subfolder instead of flat in output_dir.
                source_folder_path = os.path.join(output_dir, file_uuid)
                if os.path.exists(dest_path):
                    skipped += 1
                    if os.path.isfile(source_path):
                        os.remove(source_path)
                    if os.path.isdir(source_folder_path):
                        shutil.rmtree(source_folder_path)
                    continue
                if os.path.isfile(source_path):
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(source_path, target_dir)
                    moved += 1
                elif os.path.isdir(source_folder_path):
                    os.makedirs(target_dir, exist_ok=True)
                    for item in os.listdir(source_folder_path):
                        shutil.move(os.path.join(source_folder_path, item), target_dir)
                    os.rmdir(source_folder_path)
                    moved += 1
    return moved, skipped


def patient_is_incomplete(patient, output_dir):
    """True if this patient still has at least one included-modality file not yet organized."""
    patient_id = patient["PatientID"]
    for data_type in INCLUDE_MODALITIES:
        for file in patient.get(data_type, []):
            if "id" not in file:
                continue
            dest = os.path.join(output_dir, "raw", patient_id, data_type, file["id"], file["file_name"])
            if not os.path.exists(dest):
                return True
    return False


def main():
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} not found -- expected the existing brca_cohort manifest")

    for attempt in range(1, DOWNLOAD_PASSES + 1):
        d = GDCFileDownloader(OUTPUT_DIR, MAX_WORKERS=MAX_WORKERS, include=INCLUDE_MODALITIES)
        full_manifest = d.manifest
        d.manifest = [p for p in full_manifest if patient_is_incomplete(p, OUTPUT_DIR)]
        print(f"pass {attempt}: {len(d.manifest)}/{len(full_manifest)} patients still incomplete", flush=True)
        if not d.manifest:
            print("all patients already complete, skipping remaining passes")
            break
        try:
            d.multi_download()
        except Exception as e:
            print(f"pass {attempt}: multi_download error (continuing): {e}")
        # GDC bundles a patient's files into a .tar.gz when a data_type has more
        # than one file id (happens for some patients here); extract those into
        # flat files *before* organizing, or organize_safely will never find them
        # (and the cleanup below would delete the archive with the data still in it).
        d.extract_files(".tar.gz", "r:gz")
        d.extract_files(".tar", "r:")
        d.manifest = full_manifest
        moved, skipped = organize_safely(d, OUTPUT_DIR)
        print(f"pass {attempt}/{DOWNLOAD_PASSES}: moved={moved} already_had={skipped}", flush=True)

    remaining_archives = [fn for fn in os.listdir(OUTPUT_DIR) if fn.endswith(".tar.gz") or fn.endswith(".tar")]
    if remaining_archives:
        print(f"WARNING: {len(remaining_archives)} archive(s) left unextracted/unorganized after all passes, "
              f"NOT deleting them: {remaining_archives[:5]}{'...' if len(remaining_archives) > 5 else ''}")
    for fn in os.listdir(OUTPUT_DIR):
        full = os.path.join(OUTPUT_DIR, fn)
        if os.path.isfile(full) and (fn.endswith(".tsv") or fn.endswith(".txt")):
            os.remove(full)

    n_cnv = sum(1 for p in os.listdir(os.path.join(OUTPUT_DIR, "raw"))
                if os.path.isdir(os.path.join(OUTPUT_DIR, "raw", p, "Gene Level Copy Number")))
    n_expr = sum(1 for p in os.listdir(os.path.join(OUTPUT_DIR, "raw"))
                 if os.path.isdir(os.path.join(OUTPUT_DIR, "raw", p, "Gene Expression Quantification")))
    print(f"DONE. patients with Gene Level Copy Number: {n_cnv}, with Gene Expression Quantification: {n_expr}")


if __name__ == "__main__":
    main()
