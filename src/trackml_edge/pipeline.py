"""End-to-end pipeline: data -> graphs -> train -> select -> evaluate -> report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from .config import Config
from .data import (event_id, find_event_files, load_event_table, resolve_data_root,
                   split_events)
from .evaluate import (choose_threshold, evaluate_graphs, pooled_predictions, safe_roc_auc)
from .graph import (Standardizer, build_raw_graph, fit_standardizers,
                    graph_stats, to_pyg)
from .models import build_model
from .report import metrics_table_md
from .train import train_model
from .utils import get_device, get_logger, load_json, save_json, set_seed, setup_logging
from .visualize import plot_curves, plot_loss


def _build_split_graphs(files, cfg: Config):
    return [build_raw_graph(load_event_table(f, cfg.r_max), cfg.k_neighbors, event_id(f)) for f in files]


def run_pipeline(cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "run.log")
    cfg.save(out / "config.yaml")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log.info("device: %s | output: %s", device, out)

    # ---- data -----------------------------------------------------------------------
    files = find_event_files(resolve_data_root(cfg.data_root))
    splits = split_events(files, cfg.n_train, cfg.n_val, cfg.n_test)
    save_json({s: [p.name for p in fs] for s, fs in splits.items()}, out / "splits.json")
    log.info("found %d events -> train %d / val %d / test %d", len(files),
             cfg.n_train, cfg.n_val, cfg.n_test)

    raws: Dict[str, list] = {s: _build_split_graphs(fs, cfg) for s, fs in splits.items()}
    node_std, edge_std = fit_standardizers(raws["train"])          # training events only
    graphs = {s: [to_pyg(r, node_std, edge_std).to(device) for r in rs] for s, rs in raws.items()}
    ids = {s: [r.event_id for r in rs] for s, rs in raws.items()}

    rows = [{"split": s, **graph_stats(r)} for s, rs in raws.items() for r in rs]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    dataset = {"n_train": cfg.n_train, "n_val": cfg.n_val, "n_test": cfg.n_test,
               "mean_nodes": float(summary["nodes"].mean()), "mean_edges": float(summary["edges"].mean()),
               "mean_positive_rate": float(summary["positive_rate"].mean())}
    log.info("mean graph: %.0f hits, %.0f edges, %.1f%% positive", dataset["mean_nodes"],
             dataset["mean_edges"], 100 * dataset["mean_positive_rate"])

    # ---- train / select / evaluate ---------------------------------------------------
    results, histories, test_preds, models = {}, {}, {}, {}
    for name in cfg.models:
        log.info("=== training %s ===", name)
        set_seed(cfg.seed)
        model = build_model(name, cfg.hidden, cfg.dropout).to(device)
        hist = train_model(model, graphs["train"], graphs["val"], cfg, device)
        histories[name], models[name] = hist, model

        yv_p, yv = pooled_predictions(model, graphs["val"])
        thr = choose_threshold(yv, yv_p)
        val_auc = safe_roc_auc(yv, yv_p)
        test = evaluate_graphs(model, graphs["test"], ids["test"], thr)
        train = evaluate_graphs(model, graphs["train"], ids["train"], thr)
        test_preds[name] = (test["y"], test["p"])
        results[name] = {
            "threshold": thr, "val_roc_auc": val_auc, "best_epoch": hist["best_epoch"],
            "test_mean": test["mean"], "test_std": test["std"],
            "test_per_event": test["per_event"].to_dict(orient="records"),
            "test_confusion_matrix": test["confusion_matrix"].tolist(),
            "train_mean": train["mean"],
        }
        log.info("%s | thr %.2f | val AUC %.4f | test AUC %.4f ± %.4f | P %.3f R %.3f F1 %.3f",
                 name, thr, val_auc, test["mean"]["roc_auc"], test["std"]["roc_auc"],
                 test["mean"]["precision"], test["mean"]["recall"], test["mean"]["f1"])
        torch.save(model.state_dict(), out / f"weights_{name}.pt")

    final = max(results, key=lambda n: results[n]["val_roc_auc"]
                if results[n]["val_roc_auc"] == results[n]["val_roc_auc"] else float("-inf"))
    log.info("final model (by validation ROC-AUC): %s", final)

    torch.save({
        "model": final, "state_dict": models[final].state_dict(), "threshold": results[final]["threshold"],
        "node_std": {k: torch.as_tensor(v) for k, v in node_std.state().items()},
        "edge_std": {k: torch.as_tensor(v) for k, v in edge_std.state().items()},
        "config": cfg.to_dict(),
    }, out / "best_model.pt")

    metrics = {"final_model": final, "models": results, "dataset": dataset,
               "splits": {s: ids[s] for s in ids}, "config": cfg.to_dict()}
    save_json(metrics, out / "metrics.json")
    (out / "metrics.md").write_text(metrics_table_md(metrics) + "\n", encoding="utf-8")
    plot_loss(histories, out / "training_curves.png")
    plot_curves(test_preds, out / "test_curves.png")
    log.info("wrote metrics.json, metrics.md and figures to %s", out)
    return metrics


def load_checkpoint(path: str | Path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = build_model(ckpt["model"], cfg.hidden, cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    node_std = Standardizer.from_state({k: v.numpy() for k, v in ckpt["node_std"].items()})
    edge_std = Standardizer.from_state({k: v.numpy() for k, v in ckpt["edge_std"].items()})
    return model, ckpt["threshold"], node_std, edge_std, cfg


def evaluate_run(run_dir: str | Path, data_root: str | Path, device: str = "auto") -> dict:
    """Re-evaluate a finished run's selected model on its recorded test events."""
    run_dir = Path(run_dir)
    log = setup_logging()
    dev = get_device(device)
    model, thr, node_std, edge_std, cfg = load_checkpoint(run_dir / "best_model.pt")
    model.to(dev)
    names = load_json(run_dir / "splits.json")["test"]
    by_name = {p.name: p for p in find_event_files(resolve_data_root(data_root))}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise FileNotFoundError(f"Test events missing from data root: {missing}")
    raws = _build_split_graphs([by_name[n] for n in names], cfg)
    graphs = [to_pyg(r, node_std, edge_std).to(dev) for r in raws]
    res = evaluate_graphs(model, graphs, [r.event_id for r in raws], thr)
    saved = load_json(run_dir / "metrics.json")
    out = {"model": saved["final_model"], "threshold": thr,
           "test_mean": res["mean"], "test_std": res["std"],
           "test_per_event": res["per_event"].to_dict(orient="records")}
    ref = saved["models"][saved["final_model"]]["test_mean"]["roc_auc"]
    out["reproduces_saved_roc_auc"] = bool(abs(ref - res["mean"]["roc_auc"]) < 1e-4)
    save_json(out, run_dir / "evaluation.json")
    log.info("re-evaluated %s on %d test events | ROC-AUC %.4f (saved run: %.4f) | reproduces: %s",
             out["model"], len(names), res["mean"]["roc_auc"], ref, out["reproduces_saved_roc_auc"])
    return out
