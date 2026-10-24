"""k-NN hit-graph construction, feature standardisation and PyG conversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

NODE_FEATURES = ["x", "y", "z", "r", "phi", "eta"]
EDGE_FEATURES = ["dx", "dy", "dz", "dr", "dphi", "dist"]


@dataclass
class RawGraph:
    """Unscaled graph arrays for one event."""
    event_id: str
    node_raw: np.ndarray      # (N, 6)
    edge_index: np.ndarray    # (2, E)  src -> dst
    edge_raw: np.ndarray      # (E, 6)
    label: np.ndarray         # (E,)    1 = same non-noise particle


class Standardizer:
    """Minimal z-score scaler whose state is just two arrays (easy to checkpoint)."""

    def __init__(self, mean: np.ndarray | None = None, scale: np.ndarray | None = None):
        self.mean = mean
        self.scale = scale

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Standardizer is not fitted")
        return (x - self.mean) / self.scale

    def state(self) -> dict:
        return {"mean": np.asarray(self.mean), "scale": np.asarray(self.scale)}

    @classmethod
    def from_state(cls, state: dict) -> "Standardizer":
        return cls(np.asarray(state["mean"]), np.asarray(state["scale"]))


def build_raw_graph(df: pd.DataFrame, k: int, event: str = "event") -> RawGraph:
    """Build the k-NN candidate graph for one event table (see data.load_event_table).

    Neighbours are found in (r, 100*phi, 0.1*z) space. Node features are
    (x, y, z, r, phi, eta); edge features are (dx, dy, dz, dr, dphi, dist).
    An edge is positive iff both hits share the same non-noise particle id.
    """
    n = len(df)
    if n <= k:
        raise ValueError(f"{event}: {n} hits is not enough for k={k} neighbours")

    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    eta = np.arcsinh(z / (r + 1e-6))
    node_raw = np.column_stack([x, y, z, r, phi, eta])

    space = np.column_stack([r, phi * 100.0, z * 0.1])
    idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(space).kneighbors(space, return_distance=False)

    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()                     # column 0 is the hit itself

    dx, dy, dz = x[dst] - x[src], y[dst] - y[src], z[dst] - z[src]
    dr = r[dst] - r[src]
    dphi = np.arctan2(np.sin(phi[dst] - phi[src]), np.cos(phi[dst] - phi[src]))
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    edge_raw = np.column_stack([dx, dy, dz, dr, dphi, dist])

    pid = df["particle_id"].to_numpy()
    label = ((pid[src] == pid[dst]) & (pid[src] != 0)).astype(np.float32)

    return RawGraph(event, node_raw, np.vstack([src, dst]), edge_raw, label)


def fit_standardizers(raws: List[RawGraph]) -> tuple[Standardizer, Standardizer]:
    """Fit node/edge scalers on the given (training) graphs only."""
    node_std = Standardizer().fit(np.vstack([g.node_raw for g in raws]))
    edge_std = Standardizer().fit(np.vstack([g.edge_raw for g in raws]))
    return node_std, edge_std


def to_pyg(raw: RawGraph, node_std: Standardizer, edge_std: Standardizer) -> Data:
    return Data(
        x=torch.tensor(node_std.transform(raw.node_raw), dtype=torch.float),
        edge_index=torch.tensor(raw.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_std.transform(raw.edge_raw), dtype=torch.float),
        y=torch.tensor(raw.label, dtype=torch.float),
    )


def graph_stats(raw: RawGraph) -> dict:
    return {
        "event": raw.event_id,
        "nodes": int(raw.node_raw.shape[0]),
        "edges": int(raw.edge_index.shape[1]),
        "positive_edges": int(raw.label.sum()),
        "positive_rate": float(raw.label.mean()),
    }
