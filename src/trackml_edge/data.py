"""Locating TrackML events on disk, splitting them, and reading one event."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HITS_SUFFIX = "-hits.csv"
TRUTH_SUFFIX = "-truth.csv"


def resolve_data_root(data_root: str | Path, cache_dir: str | Path = ".cache/trackml") -> Path:
    """Return a directory containing events; extract a .zip into ``cache_dir`` if needed."""
    path = Path(data_root).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    if path.is_file() and path.suffix == ".zip":
        target = Path(cache_dir) / path.stem
        if not target.exists():
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
        return target
    return path


def find_event_files(root: str | Path) -> List[Path]:
    """All ``event*-hits.csv`` files under ``root``, sorted by name (deterministic)."""
    files = sorted(Path(root).rglob(f"event*{HITS_SUFFIX}"))
    if not files:
        raise FileNotFoundError(f"No event*{HITS_SUFFIX} files found under {root}")
    return files


def event_id(hits_path: str | Path) -> str:
    return Path(hits_path).name.split("-")[0]


def truth_path(hits_path: str | Path) -> Path:
    hits_path = Path(hits_path)
    return hits_path.with_name(hits_path.name.replace(HITS_SUFFIX, TRUTH_SUFFIX))


def split_events(files: List[Path], n_train: int, n_val: int, n_test: int) -> Dict[str, List[Path]]:
    """Disjoint train / val / test split by position in the sorted file list."""
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("n_train, n_val and n_test must each be >= 1")
    need = n_train + n_val + n_test
    if len(files) < need:
        raise ValueError(f"Need {need} events (train+val+test) but found {len(files)}")
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:need],
    }


def load_event_table(hits_path: str | Path, r_max: float | None = 200.0) -> pd.DataFrame:
    """Hits joined with truth particle ids; optionally keep only r < r_max (mm).

    Columns: hit_id, x, y, z, particle_id (0 = noise).
    """
    hits_path = Path(hits_path)
    tpath = truth_path(hits_path)
    if not tpath.exists():
        raise FileNotFoundError(f"Missing truth file for {hits_path.name}: {tpath}")
    hits = pd.read_csv(hits_path, usecols=["hit_id", "x", "y", "z"])
    truth = pd.read_csv(tpath, usecols=["hit_id", "particle_id"])
    df = hits.merge(truth, on="hit_id", how="inner")
    if len(df) != len(hits):
        raise ValueError(f"{hits_path.name}: {len(hits) - len(df)} hits have no truth row")
    if r_max is not None:
        df = df[np.hypot(df["x"], df["y"]) < r_max]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{hits_path.name}: no hits left after r < {r_max} mm cut")
    return df
