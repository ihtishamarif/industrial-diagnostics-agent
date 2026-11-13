"""Shared result types for all detectors.

Every detector returns a Finding. A Finding always carries Evidence that names the
signal, the time window and the metric that triggered it, so any diagnosis built on
top of it can cite exactly where it came from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Evidence:
    signal: str
    t_start: float
    t_end: float
    metric: str
    value: float
    threshold: float | None = None
    unit: str = ""

    def cite(self) -> str:
        thr = "" if self.threshold is None else f" (threshold {self.threshold:g}{self.unit})"
        return (
            f"{self.signal} [{self.t_start:.2f}s to {self.t_end:.2f}s]: "
            f"{self.metric} = {self.value:.4g}{self.unit}{thr}"
        )


@dataclass(frozen=True)
class Finding:
    detector: str
    detected: bool
    severity: Severity
    summary: str
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["evidence"] = [dict(asdict(e), citation=e.cite()) for e in self.evidence]
        return d
