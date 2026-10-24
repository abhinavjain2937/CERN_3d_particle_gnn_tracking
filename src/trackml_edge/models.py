"""Edge classifiers.

Both score a candidate edge from the embeddings of its two endpoint hits and its own
edge features. They differ only in whether node embeddings see their neighbourhood.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def _block(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.ReLU())


class EdgeMLPClassifier(nn.Module):
    """Edge-feature MLP. NOTE: no message passing, so this is not a GNN."""

    def __init__(self, node_in: int = 6, edge_in: int = 6, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.node_enc = _block(node_in, hidden)
        self.edge_enc = _block(edge_in, hidden)
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def node_embedding(self, x, edge_index):
        return self.node_enc(x)

    def forward(self, x, edge_index, edge_attr):
        h = self.node_embedding(x, edge_index)
        src, dst = edge_index
        z = torch.cat([h[src], h[dst], self.edge_enc(edge_attr)], dim=-1)
        return self.head(z).squeeze(-1)          # logits


class GCNEdgeClassifier(EdgeMLPClassifier):
    """Same head, but node embeddings pass through two GCNConv layers (message passing)."""

    def __init__(self, node_in: int = 6, edge_in: int = 6, hidden: int = 64, dropout: float = 0.2):
        super().__init__(node_in, edge_in, hidden, dropout)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def node_embedding(self, x, edge_index):
        h = self.node_enc(x)
        h = F.relu(self.conv1(h, edge_index))
        return F.relu(self.conv2(h, edge_index))


MODEL_REGISTRY = {"edge_mlp": EdgeMLPClassifier, "gcn": GCNEdgeClassifier}


def build_model(name: str, hidden: int = 64, dropout: float = 0.2) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](hidden=hidden, dropout=dropout)
