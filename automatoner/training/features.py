"""Feature spec shared by training, evaluation and the generated Lambda handler.

transform(): numeric columns are median-filled and standardised, categorical columns are one-hot encoded
(unknown values become all zeros). The spec is saved as features.json next to the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureSpec:
    def __init__(self, numeric: list[str], categorical: dict[str, list[str]], median: dict[str, float],
                 mean: dict[str, float], std: dict[str, float]) -> None:
        self.numeric, self.categorical, self.median, self.mean, self.std = numeric, categorical, median, mean, std

    @classmethod
    def fit(cls, df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> "FeatureSpec":
        median = {c: float(df[c].median()) for c in numeric}
        filled = {c: df[c].fillna(median[c]).astype(float) for c in numeric}
        mean = {c: float(filled[c].mean()) for c in numeric}
        std = {c: (float(filled[c].std(ddof=0)) or 1.0) for c in numeric}
        cats = {c: sorted(str(v) for v in df[c].dropna().unique()) for c in categorical}
        return cls(numeric, cats, median, mean, std)

    @property
    def names(self) -> list[str]:
        return list(self.numeric) + [f"{c}={v}" for c, vals in self.categorical.items() for v in vals]

    @property
    def dim(self) -> int:
        return len(self.numeric) + sum(len(v) for v in self.categorical.values())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            x = df[c].astype(float).fillna(self.median[c]).to_numpy()
            cols.append((x - self.mean[c]) / self.std[c])
        for c, vals in self.categorical.items():
            s = df[c].astype(str).to_numpy()
            for v in vals:
                cols.append((s == v).astype(float))
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.stack(cols, axis=1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical, "median": self.median,
                "mean": self.mean, "std": self.std, "names": self.names}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSpec":
        return cls(d["numeric"], d["categorical"], d["median"], d["mean"], d["std"])
