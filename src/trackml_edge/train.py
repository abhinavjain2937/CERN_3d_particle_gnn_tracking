"""Training loop with validation-based checkpoint selection."""
from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn

from .config import Config
from .evaluate import pooled_predictions, safe_roc_auc
from .utils import get_logger


def compute_pos_weight(graphs: Sequence, cap: float, device: torch.device) -> torch.Tensor:
    pos = sum(float(g.y.sum()) for g in graphs)
    neg = sum(float((g.y == 0).sum()) for g in graphs)
    return torch.tensor([min(neg / max(pos, 1.0), cap)], device=device)


def train_model(model: nn.Module, train_graphs: Sequence, val_graphs: Sequence,
                cfg: Config, device: torch.device) -> dict:
    """Full-batch training (one optimiser step per training event per epoch).

    Every ``cfg.eval_every`` epochs the pooled validation ROC-AUC is computed and the
    best weights are restored at the end, so only validation events influence selection.
    """
    log = get_logger()
    pos_weight = compute_pos_weight(train_graphs, cfg.pos_weight_cap, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {"train_loss": [], "val_auc": []}
    best_auc, best_epoch, best_state = float("-inf"), 0, None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for g in train_graphs:
            optimizer.zero_grad()
            loss = criterion(model(g.x, g.edge_index, g.edge_attr), g.y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        history["train_loss"].append(total / len(train_graphs))

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            p, y = pooled_predictions(model, val_graphs)
            auc = safe_roc_auc(y, p)
            history["val_auc"].append((epoch, auc))
            score = auc if auc == auc else float("-inf")      # NaN-safe
            if score > best_auc or best_state is None:
                best_auc, best_epoch = score, epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            log.info("epoch %3d/%d | loss %.4f | val ROC-AUC %.4f", epoch, cfg.epochs,
                     history["train_loss"][-1], auc)

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_auc
    return history
