"""
Pull Protein Expression Quantification + miRNA Expression Quantification for
all non-BRCA TCGA projects, one project at a time, GDC-only (skips the
IDC/TCIA imaging aggregation step entirely -- we don't need imaging, and it
appears to be a major contributor to the connection-reset spiral we hit
pulling all ~10,330 cases in one shot). Lower concurrency than the default
(2 workers instead of 8) to be gentler on GDC's API at this volume.

Resumable: skips a project entirely if its manifest.json already exists.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import med_minds
from med_minds.aggregator import Aggregator, numpy_to_python

OUT_ROOT = "./pancancer_by_project"
MAX_WORKERS = 2
INCLUDE_MODALITIES = ["Protein Expression Quantification", "miRNA Expression Quantification"]

NON_BRCA_PROJECTS = [
    "TCGA-ACC", "TCGA-BLCA", "TCGA-CESC", "TCGA-CHOL", "TCGA-COAD", "TCGA-DLBC",
    "TCGA-ESCA", "TCGA-GBM", "TCGA-HNSC", "TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP",
    "TCGA-LAML", "TCGA-LGG", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-MESO",
    "TCGA-OV", "TCGA-PAAD", "TCGA-PCPG", "TCGA-PRAD", "TCGA-READ", "TCGA-SARC",
    "TCGA-SKCM", "TCGA-STAD", "TCGA-TGCT", "TCGA-THCA", "TCGA-THYM", "TCGA-UCEC",
    "TCGA-UCS", "TCGA-UVM",
]


def gdc_only_manifest(project, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print(f"[{project}] manifest already exists, skipping build")
        return

    query = (
        "SELECT DISTINCT project_project_id, cases_case_id, cases_submitter_id "
        f"FROM clinical WHERE project_project_id = '{project}'"
    )
    cohort_data = med_minds.db.get_minds_cohort(query, use_status=False)
    print(f"[{project}] {len(cohort_data)} cases")

    aggregator = Aggregator(cohort_data, output_dir, max_workers=MAX_WORKERS)

    responses = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(aggregator.gdc_files, case_id, case_submitter_id): case_id
            for case_id, case_submitter_id in zip(aggregator.case_ids, aggregator.case_submitter_ids)
        }
        done = 0
        for future in as_completed(futures):
            responses.append(future.result())
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"[{project}] manifest progress: {done}/{len(futures)}")

    processed = [numpy_to_python(r) for r in responses]
    with open(manifest_path, "w") as f:
        json.dump(processed, f, indent=4)
    print(f"[{project}] manifest written: {manifest_path}")


import shutil

DOWNLOAD_PASSES = 3  # GDC's API drops a meaningful fraction of requests under load;
                      # repeating recovers most of what failed on the previous pass.
                      # NOTE: the library's own process_cases()/organize_files() is NOT
                      # safe to call repeatedly -- it raises FileExistsError (aborting
                      # the whole pass) when a file it already moved on a prior pass gets
                      # redownloaded and collides with the existing copy. We call
                      # multi_download() directly and do our own collision-safe organize.


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
                if os.path.exists(dest_path):
                    skipped += 1
                    if os.path.isfile(source_path):
                        os.remove(source_path)  # redundant re-download of an already-organized file
                    continue
                if os.path.isfile(source_path):
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(source_path, target_dir)
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


def download_project(project, output_dir):
    from med_minds.downloader import GDCFileDownloader

    for attempt in range(1, DOWNLOAD_PASSES + 1):
        d = GDCFileDownloader(output_dir, MAX_WORKERS=4, include=INCLUDE_MODALITIES)
        full_manifest = d.manifest
        d.manifest = [p for p in full_manifest if patient_is_incomplete(p, output_dir)]
        print(f"[{project}] pass {attempt}: {len(d.manifest)}/{len(full_manifest)} patients still incomplete")
        if not d.manifest:
            print(f"[{project}] all patients already complete, skipping remaining passes")
            break
        try:
            d.multi_download()
        except Exception as e:
            print(f"[{project}] pass {attempt}: multi_download error (continuing): {e}")
        d.manifest = full_manifest  # organize needs to see everyone (redundant flat leftovers etc.)
        moved, skipped = organize_safely(d, output_dir)
        rppa = sum(1 for _ in os_walk_matches(output_dir, "_RPPA_data.tsv"))
        mirna = sum(1 for _ in os_walk_matches(output_dir, "mirnas.quantification.txt"))
        print(f"[{project}] pass {attempt}/{DOWNLOAD_PASSES}: moved={moved} already_had={skipped} -> rppa={rppa} mirna={mirna}")
    # final cleanup: drop any stray unorganized files left flat (failed all passes' organize match)
    for fn in os.listdir(output_dir):
        full = os.path.join(output_dir, fn)
        if os.path.isfile(full) and (fn.endswith(".tsv") or fn.endswith(".txt")):
            os.remove(full)
    print(f"[{project}] download complete")


def os_walk_matches(root, suffix):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(suffix):
                yield os.path.join(dirpath, fn)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for project in NON_BRCA_PROJECTS:
        project_dir = os.path.join(OUT_ROOT, project)
        done_marker = os.path.join(project_dir, "DOWNLOAD_DONE")
        if os.path.exists(done_marker):
            print(f"[{project}] already fully done, skipping")
            continue
        try:
            gdc_only_manifest(project, project_dir)
            download_project(project, project_dir)
            with open(done_marker, "w") as f:
                f.write("done\n")
        except Exception as e:
            print(f"[{project}] FAILED: {e}", file=sys.stderr)
            time.sleep(10)
            continue
    print("ALL PROJECTS DONE")


if __name__ == "__main__":
    main()
