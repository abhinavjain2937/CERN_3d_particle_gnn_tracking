"""Matplotlib figures (headless-safe)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_loss(histories: Dict[str, dict], path: str | Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, h in histories.items():
        ax[0].plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=name)
        if h["val_auc"]:
            ep, auc = zip(*h["val_auc"])
            ax[1].plot(ep, auc, marker="o", label=name)
    ax[0].set(title="Training loss (class-weighted BCE)", xlabel="epoch", ylabel="loss")
    ax[1].set(title="Validation ROC-AUC", xlabel="epoch", ylabel="ROC-AUC")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(test_preds: Dict[str, tuple], path: str | Path) -> None:
    """ROC and precision-recall curves on the pooled test events."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, (y, p) in test_preds.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=name)
        ax[1].plot(rec, prec, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC (pooled test events)", xlabel="false positive rate", ylabel="true positive rate")
    ax[1].set(title="Precision-recall (pooled test events)", xlabel="recall", ylabel="precision")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tracks_3d(df, path: str | Path, n_particles: int = 15) -> None:
    """3D view of a few truth trajectories from one event table (columns x, y, z, particle_id)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ids = df.loc[df["particle_id"] != 0, "particle_id"].unique()[:n_particles]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for pid in ids:
        p = df[df["particle_id"] == pid].sort_values("z")
        ax.plot(p["x"], p["y"], p["z"], marker="o", ms=2, lw=1)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)",
           title=f"{len(ids)} simulated particle trajectories (TrackML)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
