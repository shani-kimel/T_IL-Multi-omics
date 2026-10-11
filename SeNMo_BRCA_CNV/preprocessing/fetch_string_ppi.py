"""
Fetches real protein-protein interaction edges from the STRING database
for the 344 protein-coding genes in gene_cna_graph.pt. Confirmed reachable
via STRING's public REST API (string-db.org/api/json/network), unlike a
bare browser-style request to string-db.org's homepage (timed out earlier
in this session -- the API endpoint is a different, script-friendly path).

Uses STRING's own gene-symbol resolution (submitting our Ensembl-derived
external_name symbols directly; STRING's API resolves near-exact synonyms
itself, so this doesn't require a separate ID-mapping step for the large
majority of genes -- a small number may fail to resolve and are simply
absent from the edge list, which is fine for a graph that already has
genomic-neighbor edges as a fallback).

required_score=400 is STRING's own "medium confidence" default cutoff
(scores are 0-1000). This is the standard threshold used in the field
when no stronger prior justifies a stricter one.

Output: SeNMo_BRCA_CNV/data/embedding_experiment/gene_ppi_edges.csv
  columns: gene_a, gene_b, combined_score (0-1000)
"""
import os
import time

import pandas as pd
import requests
import torch

ROOT = "/Users/yariv/Desktop/shani project"
GENE_GRAPH_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_cna_graph.pt")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_ppi_edges.csv")

STRING_URL = "https://string-db.org/api/json/network"
REQUIRED_SCORE = 400
SPECIES = 9606  # human


def main():
    gene_graph = torch.load(GENE_GRAPH_PATH, weights_only=False)
    gene_names = gene_graph["gene_names"]
    print(f"Querying STRING for interactions among {len(gene_names)} genes...")

    resp = requests.post(
        STRING_URL,
        data={
            "identifiers": "%0d".join(gene_names),
            "species": SPECIES,
            "required_score": REQUIRED_SCORE,
            "caller_identity": "shani_project_cna_gnn_research",
        },
        timeout=60,
    )
    resp.raise_for_status()
    interactions = resp.json()
    print(f"STRING returned {len(interactions)} interactions (score >= {REQUIRED_SCORE})")

    name_set = set(gene_names)
    rows = []
    for edge in interactions:
        a, b = edge["preferredName_A"], edge["preferredName_B"]
        if a in name_set and b in name_set and a != b:
            rows.append((a, b, edge["score"]))

    df = pd.DataFrame(rows, columns=["gene_a", "gene_b", "combined_score"])
    n_resolved = len(set(df["gene_a"]) | set(df["gene_b"]))
    print(f"{len(df)} edges among {n_resolved} of {len(gene_names)} genes resolved by STRING")
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
