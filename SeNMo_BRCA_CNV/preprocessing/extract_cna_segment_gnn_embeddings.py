"""
Runs the frozen, pan-cancer-pretrained segment-GNN encoder over the BRCA
cohort and writes a pkl with the same schema as brca_embedding_cna.pkl,
plus a new "cna_segment_gnn" key (Nx32: mean+max pooled 16-dim node
embeddings), for Config E in train_embedding_experiment.py.
"""
import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cna_segment_gnn_model import SegmentGraphEncoder, pool_embedding, segments_to_graph

ROOT = "/Users/yariv/Desktop/shani project"
SEGMENTS_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segments.pkl")
ENCODER_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "cna_segment_gnn_encoder.pt")
EXISTING_PKL = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna.pkl")
OUT_PATH = os.path.join(ROOT, "SeNMo_BRCA_CNV", "data", "embedding_experiment", "brca_embedding_cna_segment_gnn.pkl")


def main():
    ckpt = torch.load(ENCODER_PATH, weights_only=False)
    mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]

    encoder = SegmentGraphEncoder(hidden_dim=ckpt["hidden_dim"], dropout=0.0)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    with open(EXISTING_PKL, "rb") as f:
        existing = pickle.load(f)
    patient_ids = existing["patient_ids"]

    with open(SEGMENTS_PATH, "rb") as f:
        seg_data = pickle.load(f)
    id_to_idx = {pid: i for i, pid in enumerate(seg_data["patient_ids"])}
    n_runs = seg_data["n_runs"]

    graphs = []
    for pid in patient_ids:
        seg = seg_data["segments"][id_to_idx[pid]].copy()
        seg[:, 0] = (seg[:, 0] - mean[0]) / std[0]
        seg[:, 1] = (seg[:, 1] - mean[1]) / std[1]
        graphs.append(segments_to_graph(seg, n_runs))

    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            node_embeddings = encoder(batch.x, batch.edge_index)
            pooled = pool_embedding(node_embeddings, batch.batch)
            all_embeddings.append(pooled.numpy())
    cna_segment_gnn = np.concatenate(all_embeddings, axis=0)

    print(f"cna_segment_gnn embedding shape: {cna_segment_gnn.shape}")
    assert cna_segment_gnn.shape[0] == len(patient_ids)
    assert not np.isnan(cna_segment_gnn).any()

    out = dict(existing)
    out["cna_segment_gnn"] = cna_segment_gnn
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
