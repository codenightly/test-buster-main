"""Latency accumulation and the finished report.

Tally holds the running numbers while the engine works. It stores latencies in
array('d') rather than a list of Python floats, which costs 8 bytes per sample
instead of about 32. A ten million request run therefore fits in 80 MB of
samples, and the per-request records stay off the heap unless the caller asks
for them.

Every map that reaches the report gets sorted before rendering, so two runs
over the same data print the same lines in the same order.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from testbuster.config import REPORTED_PERCENTILES, RunPlan
from testbuster.histogram import LatencyHistogram

#: Status code recorded when the request never reached a response.
NO_RESPONSE = 0


@dataclass(frozen=True, slots=True)
class Attempt:
    """One request and what came back.

    status is NO_RESPONSE when the transport failed, in which case failure
    holds a short reason. dns and connect are None on a reused connection,
    because no lookup or handshake happened. validation_error holds the reason
    a response with a real status still failed a check.
    """

    status: int
    elapsed: float  # seconds from first byte sent to last byte read
    bytes_in: int
    failure: str | None
    ttfb: float | None
    dns: float | None
    connect: float | None
    retries: int
    label: str = "default"
    validation_error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the transport worked, the status is 2xx, and checks passed."""
        return self.failure is None and self.validation_error is None and 200 <= self.status < 300

    @property
    def reason(self) -> str | None:
        """The reason this request did not succeed, transport or validation."""
        return self.failure or self.validation_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "elapsed_ms": round(self.elapsed * 1000, 3),
            "bytes_in": self.bytes_in,
            "failure": self.failure,
            "validation_error": self.validation_error,
            "label": self.label,
            "ttfb_ms": None if self.ttfb is None else round(self.ttfb * 1000, 3),
            "dns_ms": None if self.dns is None else round(self.dns * 1000, 3),
            "connect_ms": None if self.connect is None else round(self.connect * 1000, 3),
            "retries": self.retries,
        }


def percentile(ordered: list[float], pct: float) -> float:
    """Return the pct-th percentile of an already sorted list.

    This interpolates between the two neighbouring samples, the same method
    NumPy uses by default. A ceiling-index method biases every percentile
    upward on small samples, so this avoids it.
    """
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


@dataclass(frozen=True, slots=True)
class Spread:
    """Min, mean, max, and percentiles for one timing series, in milliseconds."""

    count: int
    min_ms: float
    mean_ms: float
    max_ms: float
    stdev_ms: float
    percentiles_ms: dict[str, float]

    @classmethod
    def empty(cls) -> Spread:
        return cls(0, 0.0, 0.0, 0.0, 0.0, {pct_key(p): 0.0 for p in REPORTED_PERCENTILES})

    @classmethod
    def from_seconds(cls, samples: array[float]) -> Spread:
        """Build a Spread from a series of durations in seconds."""
        if not samples:
            return cls.empty()

        ordered = sorted(samples)
        count = len(ordered)
        total = math.fsum(ordered)
        mean = total / count

        # Welford is not needed here because the samples are already in hand.
        variance = math.fsum((value - mean) ** 2 for value in ordered) / count

        return cls(
            count=count,
            min_ms=ordered[0] * 1000,
            mean_ms=mean * 1000,
            max_ms=ordered[-1] * 1000,
            stdev_ms=math.sqrt(variance) * 1000,
            percentiles_ms={
                pct_key(p): percentile(ordered, p) * 1000 for p in REPORTED_PERCENTILES
            },
        )

    @classmethod
    def from_histogram(cls, hist: LatencyHistogram) -> Spread:
        """Build a Spread from a histogram, for compact-memory runs.

        The percentiles and the mean are the histogram's approximate values.
        The min and max are exact. Standard deviation is not tracked here, so it
        reads zero.
        """
        if hist.count == 0:
            return cls.empty()
        return cls(
            count=hist.count,
            min_ms=hist.min_s * 1000,
            mean_ms=hist.mean_s() * 1000,
            max_ms=hist.max_s * 1000,
            stdev_ms=0.0,
            percentiles_ms={pct_key(p): hist.percentile(p) * 1000 for p in REPORTED_PERCENTILES},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "stdev_ms": round(self.stdev_ms, 3),
            **{name: round(value, 3) for name, value in self.percentiles_ms.items()},
        }


def pct_key(pct: float) -> str:
    """Name a percentile for report keys: 50.0 -> p50, 99.9 -> p99_9."""
    text = f"{pct:g}".replace(".", "_")
    return f"p{text}"


@dataclass(slots=True)
class LabelStats:
    """Per-label totals for a scenario or a parameterized run.

    Each label keeps a histogram rather than a raw sample array, so a scenario
    with many steps stays bounded in memory. The percentiles here are the
    histogram's approximate values.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    bytes_in: int = 0
    hist: LatencyHistogram = field(default_factory=LatencyHistogram)

    def record(self, attempt: Attempt) -> None:
        self.total += 1
        if attempt.ok:
            self.succeeded += 1
        else:
            self.failed += 1
        self.bytes_in += attempt.bytes_in
        self.hist.record(attempt.elapsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate_pct": round(self.succeeded / self.total * 100, 4) if self.total else 0.0,
            "bytes_in": self.bytes_in,
            "p50_ms": round(self.hist.percentile(50) * 1000, 3),
            "p95_ms": round(self.hist.percentile(95) * 1000, 3),
            "p99_ms": round(self.hist.percentile(99) * 1000, 3),
        }


@dataclass(slots=True)
class Tally:
    """Running totals for one load test.

    The engine calls record() from a single event loop, so no lock is needed.
    Set keep_attempts to hold every Attempt, which the JSON and CSV writers
    need. Leave it off for long runs. Set compact_memory to skip the exact
    sample arrays and report percentiles from the histogram instead.
    """

    keep_attempts: bool = False
    compact_memory: bool = False

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    retried: int = 0
    bytes_in: int = 0

    status_counts: Counter[int] = field(default_factory=Counter)
    failure_counts: Counter[str] = field(default_factory=Counter)

    histogram: LatencyHistogram = field(default_factory=LatencyHistogram)
    timeline: dict[int, int] = field(default_factory=dict)
    by_label: dict[str, LabelStats] = field(default_factory=dict)

    _elapsed: array[float] = field(default_factory=lambda: array("d"))
    _ttfb: array[float] = field(default_factory=lambda: array("d"))
    _dns: array[float] = field(default_factory=lambda: array("d"))
    _connect: array[float] = field(default_factory=lambda: array("d"))
    _attempts: list[Attempt] = field(default_factory=list)
    _labels_on: bool = False

    def enable_labels(self, labels: tuple[str, ...]) -> None:
        """Turn on the per-label breakdown when a run uses more than one label."""
        if len(labels) > 1:
            self._labels_on = True
            for label in labels:
                self.by_label.setdefault(label, LabelStats())

    def record(self, attempt: Attempt, second: int | None = None) -> None:
        """Fold one finished request into the totals."""
        self.total += 1
        if attempt.ok:
            self.succeeded += 1
        else:
            self.failed += 1

        if attempt.retries:
            self.retried += 1

        self.bytes_in += attempt.bytes_in
        self.status_counts[attempt.status] += 1

        reason = attempt.reason
        if reason is not None:
            self.failure_counts[reason] += 1

        self.histogram.record(attempt.elapsed)
        if second is not None:
            self.timeline[second] = self.timeline.get(second, 0) + 1

        if not self.compact_memory:
            self._elapsed.append(attempt.elapsed)
            if attempt.ttfb is not None:
                self._ttfb.append(attempt.ttfb)
            if attempt.dns is not None:
                self._dns.append(attempt.dns)
            if attempt.connect is not None:
                self._connect.append(attempt.connect)

        if self._labels_on:
            # enable_labels pre-creates every known label, so a get that misses
            # only happens for a label a source did not declare. Reuse the
            # existing stats instead of building a throwaway one per request.
            label_stats = self.by_label.get(attempt.label)
            if label_stats is None:
                label_stats = LabelStats()
                self.by_label[attempt.label] = label_stats
            label_stats.record(attempt)

        if self.keep_attempts:
            self._attempts.append(attempt)

    def summarize(
        self,
        plan: RunPlan,
        wall_seconds: float,
        *,
        interrupted: bool,
        stop_reason: str = "completed",
    ) -> Report:
        """Freeze the totals into a Report."""
        throughput = self.total / wall_seconds if wall_seconds > 0 else 0.0
        success_rate = (self.succeeded / self.total * 100) if self.total else 0.0
        # A compact run keeps no exact samples, so the histogram answers instead.
        latency = (
            Spread.from_histogram(self.histogram)
            if self.compact_memory
            else Spread.from_seconds(self._elapsed)
        )

        return Report(
            plan=plan,
            wall_seconds=wall_seconds,
            interrupted=interrupted,
            stop_reason=stop_reason,
            total=self.total,
            succeeded=self.succeeded,
            failed=self.failed,
            retried=self.retried,
            bytes_in=self.bytes_in,
            requests_per_second=throughput,
            success_rate=success_rate,
            latency=latency,
            ttfb=Spread.from_seconds(self._ttfb),
            dns=Spread.from_seconds(self._dns),
            connect=Spread.from_seconds(self._connect),
            status_counts=dict(sorted(self.status_counts.items())),
            failure_counts=dict(
                sorted(self.failure_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            attempts=tuple(self._attempts),
            histogram=self.histogram,
            timeline=tuple(sorted(self.timeline.items())),
            by_label=dict(sorted(self.by_label.items())),
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    """One threshold check and its outcome."""

    name: str
    limit: float
    actual: float
    unit: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "actual": round(self.actual, 3),
            "unit": self.unit,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """The finished numbers for one load test."""

    plan: RunPlan
    wall_seconds: float
    interrupted: bool

    total: int
    succeeded: int
    failed: int
    retried: int
    bytes_in: int
    requests_per_second: float
    success_rate: float

    latency: Spread
    ttfb: Spread
    dns: Spread
    connect: Spread

    status_counts: dict[int, int]
    failure_counts: dict[str, int]
    attempts: tuple[Attempt, ...]

    histogram: LatencyHistogram = field(default_factory=LatencyHistogram)
    timeline: tuple[tuple[int, int], ...] = ()
    by_label: dict[str, LabelStats] = field(default_factory=dict)
    stop_reason: str = "completed"

    @property
    def failure_rate(self) -> float:
        return 100.0 - self.success_rate

    @property
    def bytes_per_request(self) -> float:
        return self.bytes_in / self.total if self.total else 0.0

    @property
    def throughput_bytes_per_second(self) -> float:
        return self.bytes_in / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def check_gates(self) -> list[GateResult]:
        """Run the configured thresholds and return one result per active gate."""
        gates = self.plan.gates
        results: list[GateResult] = []

        if gates.max_p95_ms is not None:
            actual = self.latency.percentiles_ms["p95"]
            results.append(
                GateResult(
                    "p95 latency", gates.max_p95_ms, actual, "ms", actual <= gates.max_p95_ms
                )
            )
        if gates.max_p99_ms is not None:
            actual = self.latency.percentiles_ms["p99"]
            results.append(
                GateResult(
                    "p99 latency", gates.max_p99_ms, actual, "ms", actual <= gates.max_p99_ms
                )
            )
        if gates.min_success_rate is not None:
            results.append(
                GateResult(
                    "success rate",
                    gates.min_success_rate,
                    self.success_rate,
                    "%",
                    self.success_rate >= gates.min_success_rate,
                )
            )
        return results

    def to_dict(self, *, tool_version: str) -> dict[str, Any]:
        """Build the JSON report.

        The shape is stable and versioned. Bump the schema string when a key
        changes meaning, so downstream parsers can tell the difference.
        """
        gates = self.check_gates()
        payload: dict[str, Any] = {
            "schema": "testbuster/report/1",
            "tool": {"name": "Test Buster!", "version": tool_version},
            "plan": self.plan.to_dict(),
            "summary": {
                "total_requests": self.total,
                "successful": self.succeeded,
                "failed": self.failed,
                "retried": self.retried,
                "success_rate_pct": round(self.success_rate, 4),
                "wall_seconds": round(self.wall_seconds, 6),
                "requests_per_second": round(self.requests_per_second, 4),
                "bytes_in": self.bytes_in,
                "bytes_per_request": round(self.bytes_per_request, 2),
                "bytes_per_second": round(self.throughput_bytes_per_second, 2),
                "interrupted": self.interrupted,
                "stop_reason": self.stop_reason,
            },
            "latency": self.latency.to_dict(),
            "phases": {
                "ttfb": self.ttfb.to_dict(),
                "dns": self.dns.to_dict(),
                "connect": self.connect.to_dict(),
            },
            "status_codes": {str(code): count for code, count in self.status_counts.items()},
            "failures": dict(self.failure_counts),
            "histogram": self.histogram.to_dict(),
            "gates": [gate.to_dict() for gate in gates],
        }
        if self.timeline:
            payload["timeline"] = [{"second": s, "count": c} for s, c in self.timeline]
        if self.by_label:
            payload["by_label"] = {name: stats.to_dict() for name, stats in self.by_label.items()}
        if self.plan.keep_attempts:
            payload["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return payload
