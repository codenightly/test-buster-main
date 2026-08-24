"""Compare two saved reports.

A single run says how a system behaves now. A pair of runs says whether a change
made it better or worse. This reads two JSON reports and lines up the metrics
that matter, then flags the ones that moved the wrong way past a threshold.

Latency lower is better. Success rate and throughput higher is better. Each
metric knows its own good direction, so a regression is unambiguous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from testbuster.errors import ExitCode, TestBusterError

Direction = Literal["lower_better", "higher_better"]


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One metric compared across two runs."""

    name: str
    unit: str
    base: float
    new: float
    direction: Direction

    @property
    def percent(self) -> float:
        if self.base == 0:
            return 0.0 if self.new == 0 else 100.0
        return (self.new - self.base) / abs(self.base) * 100.0

    @property
    def is_regression(self) -> bool:
        """True when the metric moved in its bad direction at all."""
        if self.direction == "lower_better":
            return self.new > self.base
        return self.new < self.base

    def regressed_beyond(self, percent_tolerance: float) -> bool:
        """True when a regression is larger than the allowed percentage."""
        if not self.is_regression:
            return False
        return abs(self.percent) > percent_tolerance


def load_report(path: Path) -> dict[str, Any]:
    """Read a JSON report written by --output."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TestBusterError(f"cannot read report {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise TestBusterError(f"{path} is not a valid report: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema") != "testbuster/report/1":
        raise TestBusterError(f"{path} is not a Test Buster! report")
    return data


def _get(report: dict[str, Any], *keys: str) -> float:
    """Read a nested numeric value, defaulting to zero when absent."""
    current: Any = report
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return 0.0
        current = current[key]
    return float(current) if isinstance(current, int | float) else 0.0


def compare(base: dict[str, Any], new: dict[str, Any]) -> list[MetricDelta]:
    """Return the metric deltas between two reports, in a fixed order."""
    deltas: list[MetricDelta] = [
        MetricDelta(
            "success rate",
            "%",
            _get(base, "summary", "success_rate_pct"),
            _get(new, "summary", "success_rate_pct"),
            "higher_better",
        ),
        MetricDelta(
            "throughput",
            "req/s",
            _get(base, "summary", "requests_per_second"),
            _get(new, "summary", "requests_per_second"),
            "higher_better",
        ),
    ]
    for pct in ("p50", "p95", "p99", "p99_9"):
        deltas.append(
            MetricDelta(
                f"latency {pct}",
                "ms",
                _get(base, "latency", pct),
                _get(new, "latency", pct),
                "lower_better",
            )
        )
    return deltas


def find_regressions(deltas: list[MetricDelta], tolerance_percent: float) -> list[MetricDelta]:
    """Return the deltas that regressed beyond the tolerance."""
    return [delta for delta in deltas if delta.regressed_beyond(tolerance_percent)]


def exit_code(regressions: list[MetricDelta]) -> ExitCode:
    """Return GATE_FAILED when anything regressed, else OK."""
    return ExitCode.GATE_FAILED if regressions else ExitCode.OK
