"""Device abstraction.

Phase 1 has a simulated PID controller. Phase 2 adds Modbus and Serial devices
behind the same interface, so the agent code does not change.

Writes are split in two on purpose: a device can only apply a WriteRequest, and
a WriteRequest can only be applied after it has been approved. The agent graph
creates the request, a human approves or rejects it, then the graph applies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd


class ApprovalRequired(PermissionError):
    pass


@dataclass
class WriteRequest:
    device_id: str
    params: dict[str, float]
    reason: str
    approved: bool = False
    approved_by: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Device(Protocol):
    device_id: str

    def read_params(self) -> dict[str, float]: ...

    def info(self) -> dict[str, Any]: ...

    def apply(self, req: WriteRequest) -> dict[str, Any]: ...

    def capture(self, **kwargs: Any) -> pd.DataFrame: ...


@dataclass
class AuditLog:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **kw: Any) -> None:
        self.entries.append({"time": datetime.now(timezone.utc).isoformat(), **kw})
