"""Synthetic TrackML-format events for smoke tests and demos.

These are NOT real TrackML data: helical-ish tracks plus noise hits, written in the same
CSV layout (event*-hits.csv / event*-truth.csv). Metrics obtained on them mean nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RADII = np.array([25, 35, 45, 55, 70, 85, 100, 115, 130, 145, 160, 175, 190,
                   260, 340, 500, 700, 1000.0])


def make_synthetic_dataset(out_dir: str | Path, n_events: int = 8, n_particles: int = 400,
                           n_noise: int = 600, first_event: int = 1000, seed: int = 0) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for e in range(first_event, first_event + n_events):
        rng = np.random.default_rng(seed + e)
        rows, truth, hid = [], [], 1
        for pid in range(1, n_particles + 1):
            phi0 = rng.uniform(-np.pi, np.pi)
            curv = rng.normal(0, 0.0008)
            cot = rng.normal(0, 1.2)
            z0 = rng.normal(0, 40)
            for r in _RADII:
                if rng.random() < 0.08:
                    continue
                phi = phi0 + curv * r
                rows.append((hid, r * np.cos(phi) + rng.normal(0, 0.2), r * np.sin(phi) + rng.normal(0, 0.2),
                             z0 + r * cot, 7, 2, 1))
                truth.append((hid, pid))
                hid += 1
        for _ in range(n_noise):
            r, phi = rng.choice(_RADII), rng.uniform(-np.pi, np.pi)
            rows.append((hid, r * np.cos(phi), r * np.sin(phi), rng.normal(0, 300), 7, 2, 1))
            truth.append((hid, 0))
            hid += 1
        hits = pd.DataFrame(rows, columns=["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"])
        tr = pd.DataFrame(truth, columns=["hit_id", "particle_id"])
        for c in ["tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"]:
            tr[c] = 0.0
        hits.sample(frac=1, random_state=e).to_csv(out / f"event{e:09d}-hits.csv", index=False)
        tr.to_csv(out / f"event{e:09d}-truth.csv", index=False)
    return out
