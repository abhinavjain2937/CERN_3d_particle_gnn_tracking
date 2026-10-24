"""Command-line interface:  trackml-edge {run,evaluate,plot-tracks,make-synthetic}"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config


def _load_config(args) -> Config:
    path = Path(args.config)
    cfg = Config.from_yaml(path) if path.exists() else Config()
    return cfg.override(
        data_root=args.data_root, out_dir=args.out_dir, n_train=args.n_train, n_val=args.n_val,
        n_test=args.n_test, epochs=args.epochs, device=args.device, seed=args.seed,
        models=args.models, k_neighbors=args.k_neighbors, r_max=args.r_max)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="trackml-edge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="train, select, evaluate and report")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--data-root"); run.add_argument("--out-dir")
    run.add_argument("--n-train", type=int); run.add_argument("--n-val", type=int)
    run.add_argument("--n-test", type=int); run.add_argument("--epochs", type=int)
    run.add_argument("--device"); run.add_argument("--seed", type=int)
    run.add_argument("--k-neighbors", type=int); run.add_argument("--r-max", type=float)
    run.add_argument("--models", nargs="+")

    ev = sub.add_parser("evaluate", help="re-evaluate a finished run on its recorded test events")
    ev.add_argument("--run", required=True); ev.add_argument("--data-root", required=True)
    ev.add_argument("--device", default="auto")

    pt = sub.add_parser("plot-tracks", help="3D plot of a few truth trajectories")
    pt.add_argument("--data-root", required=True); pt.add_argument("--out", default="docs/tracks_3d.png")
    pt.add_argument("--event-index", type=int, default=0); pt.add_argument("--n-particles", type=int, default=15)

    sy = sub.add_parser("make-synthetic", help="write SYNTHETIC events for smoke tests (not real data)")
    sy.add_argument("--out", default="data/synthetic"); sy.add_argument("--n-events", type=int, default=12)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .pipeline import run_pipeline
        run_pipeline(_load_config(args))
    elif args.cmd == "evaluate":
        from .pipeline import evaluate_run
        evaluate_run(args.run, args.data_root, args.device)
    elif args.cmd == "plot-tracks":
        from .data import find_event_files, load_event_table, resolve_data_root
        from .visualize import plot_tracks_3d
        files = find_event_files(resolve_data_root(args.data_root))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plot_tracks_3d(load_event_table(files[args.event_index], r_max=None), args.out, args.n_particles)
        print("wrote", args.out)
    elif args.cmd == "make-synthetic":
        from .synthetic import make_synthetic_dataset
        print("wrote synthetic events to", make_synthetic_dataset(args.out, args.n_events))


if __name__ == "__main__":
    main()
