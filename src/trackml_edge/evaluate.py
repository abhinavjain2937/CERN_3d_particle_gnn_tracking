"""Prediction, threshold selection and edge-level metrics."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

METRIC_NAMES = ["roc_auc", "precision", "recall", "f1"]
DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


@torch.no_grad()
def predict(model: torch.nn.Module, graph) -> Tuple[np.ndarray, np.ndarray]:
    """Edge probabilities and labels for one graph."""
    model.eval()
    probs = torch.sigmoid(model(graph.x, graph.edge_index, graph.edge_attr))
    return probs.cpu().numpy(), graph.y.cpu().numpy()


def pooled_predictions(model, graphs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    preds = [predict(model, g) for g in graphs]
    return np.concatenate([p for p, _ in preds]), np.concatenate([y for _, y in preds])


def safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def edge_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": safe_roc_auc(y, p),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray, grid: Sequence[float] = DEFAULT_GRID) -> float:
    """Threshold maximising F1 on the given (validation) predictions."""
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate_graphs(model, graphs: Sequence, event_ids: List[str], threshold: float) -> dict:
    """Per-event metrics plus a pooled confusion matrix at a fixed threshold."""
    rows, ys, ps = [], [], []
    for eid, g in zip(event_ids, graphs):
        p, y = predict(model, g)
        rows.append({"event": eid, **edge_metrics(y, p, threshold)})
        ys.append(y); ps.append(p)
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    cm = confusion_matrix(y_all, (p_all >= threshold).astype(int), labels=[0, 1])
    per_event = pd.DataFrame(rows)
    return {
        "per_event": per_event,
        "mean": per_event[METRIC_NAMES].mean().to_dict(),
        "std": per_event[METRIC_NAMES].std(ddof=0).to_dict(),
        "confusion_matrix": cm,
        "y": y_all,
        "p": p_all,
    }
