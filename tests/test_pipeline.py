import json

import torch

from trackml_edge.cli import main
from trackml_edge.config import Config
from trackml_edge.pipeline import evaluate_run, run_pipeline
from trackml_edge.report import metrics_table_md, resume_macros


def _cfg(root, out):
    return Config(data_root=str(root), out_dir=str(out), n_train=2, n_val=2, n_test=3, epochs=6,
                  eval_every=3, hidden=16, device="cpu")


def test_full_pipeline_artifacts_and_reproducibility(synthetic_root, tmp_path):
    out = tmp_path / "run"
    metrics = run_pipeline(_cfg(synthetic_root, out))

    for name in ["config.yaml", "splits.json", "dataset_summary.csv", "metrics.json", "metrics.md",
                 "best_model.pt", "weights_edge_mlp.pt", "weights_gcn.pt", "training_curves.png", "test_curves.png"]:
        assert (out / name).exists(), name

    splits = json.loads((out / "splits.json").read_text())
    assert not (set(splits["train"]) & set(splits["val"]) | set(splits["train"]) & set(splits["test"])
                | set(splits["val"]) & set(splits["test"]))                       # no leakage across splits
    assert metrics["final_model"] in ("edge_mlp", "gcn")
    best = metrics["models"][metrics["final_model"]]
    assert 0.5 < best["test_mean"]["roc_auc"] <= 1.0

    # report helpers
    assert "Test ROC-AUC" in metrics_table_md(metrics)
    tex = resume_macros(metrics)
    assert "\\newcommand{\\trkAUC}" in tex and "\\newcommand{\\trkNtest}{3}" in tex

    # re-evaluating the saved checkpoint reproduces the stored numbers
    res = evaluate_run(out, synthetic_root, device="cpu")
    assert res["reproduces_saved_roc_auc"]


def test_deterministic_given_seed(synthetic_root, tmp_path):
    a = run_pipeline(_cfg(synthetic_root, tmp_path / "a"))
    b = run_pipeline(_cfg(synthetic_root, tmp_path / "b"))
    for name in a["models"]:
        assert abs(a["models"][name]["test_mean"]["roc_auc"] - b["models"][name]["test_mean"]["roc_auc"]) < 1e-6


def test_cli_run_and_synthetic(tmp_path):
    data = tmp_path / "data"
    main(["make-synthetic", "--out", str(data), "--n-events", "6"])
    main(["run", "--config", "does_not_exist.yaml", "--data-root", str(data), "--out-dir", str(tmp_path / "r"),
          "--n-train", "2", "--n-val", "1", "--n-test", "2", "--epochs", "3", "--device", "cpu",
          "--models", "edge_mlp"])
    assert (tmp_path / "r" / "metrics.json").exists()
