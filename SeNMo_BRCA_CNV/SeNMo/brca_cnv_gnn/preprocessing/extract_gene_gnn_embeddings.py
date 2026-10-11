"""
Runs the frozen, pan-cancer-pretrained gene-level CNA-GNN encoder over the
BRCA cohort and writes a pkl with the same schema as brca_embedding_cna.pkl,
plus a new "cna_gene_gnn" key (Nx32: mean+max pooled 16-dim node
embeddings over the 344-gene, genomic+PPI graph).
"""
import os
import pickle

import numpy as np
import torch

from cna_gnn_model import CNAGraphEncoder, build_normalized_adjacency, pool_embedding

ROOT = "/Users/yariv/Desktop/shani project"
GENE_CNA_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_cna_full.pt")
ENCODER_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "gene_gnn_encoder.pt")
EXISTING_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_gene_gnn.pkl")


def main():
    ckpt = torch.load(ENCODER_PATH, weights_only=False)
    gene_data = torch.load(GENE_CNA_PATH, weights_only=False)
    assert gene_data["gene_names"] == ckpt["gene_names"]
    adj = build_normalized_adjacency(ckpt["edge_index"], len(ckpt["gene_names"]))

    encoder = CNAGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    patient_ids = existing["patient_ids"]

    pid_to_idx = {pid: i for i, pid in enumerate(gene_data["patient_ids"])}
    rows = [pid_to_idx[pid] for pid in patient_ids]
    X = gene_data["gene_cna"][rows]
    X = (X - ckpt["scaler_mean"]) / ckpt["scaler_std"]

    x_in = torch.tensor(X).t().unsqueeze(-1)
    with torch.no_grad():
        node_embeddings = encoder(x_in, adj)
        cna_gene_gnn = pool_embedding(node_embeddings).numpy()

    print(f"cna_gene_gnn embedding shape: {cna_gene_gnn.shape}")
    assert not np.isnan(cna_gene_gnn).any()

    out = dict(existing)
    out["cna_gene_gnn"] = cna_gene_gnn
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
