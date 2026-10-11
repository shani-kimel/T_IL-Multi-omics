"""
GNN encoder for the "CNV segments as nodes" graph. Unlike the fixed-bin
grid (cna_gnn_model.py), every patient has their own graph here -- a
different number of segments/nodes, different edges -- so we use
PyTorch Geometric's Data/Batch machinery instead of a single shared dense
adjacency.

Node features per segment: [value, log1p(length_in_bins)] (run_id is used
only to build edges, not as a feature). Edges: chain consecutive segments
within a run, plus boundary edges linking each run's first segment to the
next run's first segment (mirroring cna_graph.py's cross-region hubs, at
segment granularity instead of bin granularity).
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool


def segments_to_graph(segment_array, n_runs):
    """segment_array: (n_segments, 3) = [value, log_length, run_id]. Returns a
    torch_geometric.data.Data with x=[value, log_length] and chain+hub edges."""
    n_seg = segment_array.shape[0]
    run_id = segment_array[:, 2].astype(np.int64)

    edges = []
    for i in range(1, n_seg):
        if run_id[i] == run_id[i - 1]:
            edges.append((i - 1, i))

    run_start_idx = {}
    for i, r in enumerate(run_id):
        if r not in run_start_idx:
            run_start_idx[r] = i
    starts = [run_start_idx[r] for r in sorted(run_start_idx)]
    for a in range(len(starts) - 1):
        edges.append((starts[a], starts[a + 1]))

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    x = torch.tensor(segment_array[:, :2], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)


class SegmentGraphEncoder(nn.Module):
    """2-layer GraphSAGE encoder: 2 -> hidden -> hidden, per-node embedding."""

    def __init__(self, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.gc1 = SAGEConv(2, hidden_dim)
        self.gc2 = SAGEConv(hidden_dim, hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.act(self.gc1(x, edge_index))
        h = self.dropout(h)
        h = self.act(self.gc2(h, edge_index))
        return h


class SegmentGraphAutoencoder(nn.Module):
    """Encoder + a linear decoder back to [value, log_length] per node, for
    masked-node reconstruction pretraining."""

    def __init__(self, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.encoder = SegmentGraphEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        recon = self.decoder(h)
        return recon, h


def pool_embedding(node_embeddings, batch_index):
    mean_pool = global_mean_pool(node_embeddings, batch_index)
    max_pool = global_max_pool(node_embeddings, batch_index)
    return torch.cat([mean_pool, max_pool], dim=-1)
