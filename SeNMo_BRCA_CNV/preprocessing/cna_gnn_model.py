"""
Shared GCN encoder/decoder for the CNA genomic-adjacency graph.

All patients share the exact same graph topology (the 7460 genomic bins and
their genomic-adjacency edges from build_cna_graph.py); only the per-bin
CNA values differ per patient. That means a GCN layer's propagation step
(normalized_adj @ X) can be computed once per batch as a single sparse-dense
matmul across all patients in the batch simultaneously, with X shaped
(n_bins, batch_size * feature_dim) -- no per-graph looping needed.
"""
import torch
import torch.nn as nn


def build_normalized_adjacency(edge_index, n_nodes):
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    self_loops = torch.arange(n_nodes, dtype=torch.long)
    self_loop_index = torch.stack([self_loops, self_loops], dim=0)
    full_edge_index = torch.cat([edge_index, self_loop_index], dim=1)

    row, col = full_edge_index[0], full_edge_index[1]
    deg = torch.zeros(n_nodes)
    deg.scatter_add_(0, row, torch.ones(row.shape[0]))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(full_edge_index, values, (n_nodes, n_nodes)).coalesce()
    return adj


class GCNLayer(nn.Module):
    """x: (n_nodes, batch_size, in_dim) -> (n_nodes, batch_size, out_dim)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        n_nodes, batch_size, in_dim = x.shape
        x_flat = x.reshape(n_nodes, batch_size * in_dim)
        propagated = torch.sparse.mm(adj, x_flat)
        propagated = propagated.reshape(n_nodes, batch_size, in_dim)
        return self.lin(propagated)


class CNAGraphEncoder(nn.Module):
    """2-layer GCN encoder: 1 -> hidden -> hidden, producing a per-node embedding."""

    def __init__(self, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.gc1 = GCNLayer(1, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (n_nodes, batch_size, 1)
        h = self.act(self.gc1(x, adj))
        h = self.dropout(h)
        h = self.act(self.gc2(h, adj))
        return h  # (n_nodes, batch_size, hidden_dim)


class CNAGraphAutoencoder(nn.Module):
    """Encoder + a 1-layer GCN decoder back to a scalar per bin, for masked reconstruction pretraining."""

    def __init__(self, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.encoder = CNAGraphEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = GCNLayer(hidden_dim, 1)

    def forward(self, x, adj):
        h = self.encoder(x, adj)
        recon = self.decoder(h, adj)
        return recon, h


def pool_embedding(node_embeddings):
    """(n_nodes, batch_size, hidden_dim) -> (batch_size, 2*hidden_dim) via mean+max pooling over nodes."""
    mean_pool = node_embeddings.mean(dim=0)
    max_pool = node_embeddings.max(dim=0).values
    return torch.cat([mean_pool, max_pool], dim=-1)


class CNAGraphCrossModalAutoencoder(nn.Module):
    """CNAGraphAutoencoder's masked-bin self-reconstruction, fused with a
    second head that predicts mRNA expression from the SAME pooled graph
    embedding used downstream -- one representation shaped by both the
    genomic-adjacency prior (what the plain GNN already had) and an
    expression-predictiveness prior (what the separate MLP-based cross-modal
    VAE had, but without any genomic structure), instead of training the two
    priors in separate models and gluing them together post-hoc at inference.

    No Cox multi-task term here: pretrain_cna_gnn_coxmtl.py's own results
    showed the pooled pan-cancer Cox loss made the plain GNN *worse*
    (0.7629 vs 0.7813 downstream test C-index), so it's left out rather than
    carried forward into a new model on the strength of the same evidence.
    """

    def __init__(self, hidden_dim=16, mrna_dim=6016, mrna_hidden=(128,), dropout=0.1):
        super().__init__()
        self.encoder = CNAGraphEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.self_decoder = GCNLayer(hidden_dim, 1)

        cross_layers = []
        dims = [2 * hidden_dim] + list(mrna_hidden)
        for i in range(len(dims) - 1):
            cross_layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        cross_layers.append(nn.Linear(dims[-1], mrna_dim))
        self.cross_decoder = nn.Sequential(*cross_layers)

    def forward(self, x, adj):
        node_emb = self.encoder(x, adj)  # (n_bins, batch_size, hidden_dim)
        self_recon = self.self_decoder(node_emb, adj)  # (n_bins, batch_size, 1)
        pooled = pool_embedding(node_emb)  # (batch_size, 2*hidden_dim)
        mrna_pred = self.cross_decoder(pooled)  # (batch_size, mrna_dim)
        return self_recon, pooled, mrna_pred
