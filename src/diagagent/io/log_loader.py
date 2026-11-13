"""Load time-series logs into a standard DataFrame and keep them in a store.

Standard form: a float column "t" in seconds from the first sample, sorted, plus
numeric signal columns. The agent never passes raw arrays through the LLM; it
passes a dataset_id and the tools look the data up in the DatasetStore.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LogFormatError(ValueError):
    pass


def standardize(
    df: pd.DataFrame,
    time_col: str = "t",
    rename: dict[str, str] | None = None,
    time_unit: str = "s",
) -> pd.DataFrame:
    """Convert the time column to seconds from start, sort, keep numeric columns."""
    if time_col not in df.columns:
        raise LogFormatError(f"time column '{time_col}' not found; columns: {list(df.columns)}")
    df = df.copy()
    tc = df[time_col]
    if not pd.api.types.is_numeric_dtype(tc):
        ts = pd.to_datetime(tc, errors="coerce", utc=True)
        if ts.isna().all():
            raise LogFormatError(f"cannot parse time column '{time_col}'")
        t = (ts - ts.min()).dt.total_seconds()
    else:
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[time_unit]
        t = tc.astype(float) * scale
        t = t - t.min()
    df = df.drop(columns=[time_col])
    if rename:
        df = df.rename(columns=rename)
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().any()]
    if num.shape[1] == 0:
        raise LogFormatError("no numeric signal columns found")
    num.insert(0, "t", t.to_numpy())
    num = num.dropna(subset=["t"]).sort_values("t", kind="stable")
    num = num.drop_duplicates(subset="t", keep="first").reset_index(drop=True)
    return num


def load_log(
    path: str | Path,
    time_col: str = "t",
    rename: dict[str, str] | None = None,
    time_unit: str = "s",
    **read_kwargs: Any,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt", ".log"}:
        raw = pd.read_csv(p, **read_kwargs)
    elif suffix == ".tsv":
        raw = pd.read_csv(p, sep="\t", **read_kwargs)
    elif suffix == ".parquet":
        raw = pd.read_parquet(p, **read_kwargs)
    elif suffix == ".json":
        raw = pd.read_json(p, **read_kwargs)
    else:
        raise LogFormatError(f"unsupported file type: {suffix}")
    return standardize(raw, time_col=time_col, rename=rename, time_unit=time_unit)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    dt = np.diff(df["t"].to_numpy())
    return {
        "n_samples": int(len(df)),
        "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]) if len(df) else 0.0,
        "nominal_dt_s": float(np.median(dt)) if len(dt) else None,
        "signals": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()),
                "mean": float(df[c].mean()), "n_nan": int(df[c].isna().sum())}
            for c in df.columns if c != "t"
        },
    }


@dataclass
class DatasetStore:
    """In-memory registry of loaded datasets, keyed by short ids."""
    _data: dict[str, pd.DataFrame] = field(default_factory=dict)
    _meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    _ids: Any = field(default_factory=lambda: itertools.count(1))

    def add(self, df: pd.DataFrame, source: str) -> str:
        ds_id = f"ds{next(self._ids)}"
        self._data[ds_id] = df
        self._meta[ds_id] = {"source": source}
        return ds_id

    def get(self, ds_id: str) -> pd.DataFrame:
        if ds_id not in self._data:
            raise KeyError(f"unknown dataset_id '{ds_id}'. Known: {list(self._data)}")
        return self._data[ds_id]

    def list(self) -> list[dict[str, Any]]:
        return [{"dataset_id": k, **self._meta[k], "n_samples": len(v),
                 "columns": list(v.columns)} for k, v in self._data.items()]
