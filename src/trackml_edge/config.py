"""Experiment configuration (dataclass + YAML)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    seed: int = 42

    # data
    data_root: str = "data/train_sample"
    out_dir: str = "runs/default"
    r_max: float = 200.0
    k_neighbors: int = 5
    n_train: int = 5
    n_val: int = 2
    n_test: int = 5

    # models
    models: List[str] = field(default_factory=lambda: ["edge_mlp", "gcn"])
    hidden: int = 64
    dropout: float = 0.2

    # optimisation
    epochs: int = 60
    lr: float = 3e-3
    weight_decay: float = 1e-4
    pos_weight_cap: float = 12.0
    eval_every: int = 5

    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def override(self, **kwargs) -> "Config":
        """Return a copy with every non-None keyword replaced."""
        data = asdict(self)
        for key, value in kwargs.items():
            if value is None:
                continue
            if key not in data:
                raise ValueError(f"Unknown config key: {key}")
            data[key] = value
        return Config(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
