"""A bounded-memory latency histogram.

Storing every latency in an array is exact but grows with the request count. A
ten million request run holds ten million samples. This histogram instead keeps
a count per bucket, so memory stays flat no matter how many requests run.

The buckets are log-linear. Each power-of-two band of microseconds splits into a
fixed number of even sub-buckets, so the relative error of any reported value
stays near 1 / (2 * SUB_BUCKETS). Exact min and max are tracked on the side, so
the ends of the range never drift.

The histogram also merges, which is how a distributed run folds the numbers from
several workers into one report. to_dict carries the raw bucket counts and
from_dict reads them back, so the merge over the wire loses no resolution.
"""

from __future__ import annotations

import math
from typing import Any, Final

#: Sub-buckets per power-of-two band. 16 gives about 3 percent relative error.
SUB_BUCKETS: Final[int] = 16


def _bucket_index(micros: float) -> int:
    """Return the bucket a microsecond value falls in."""
    if micros < 1.0:
        return 0
    band = math.floor(math.log2(micros))
    band_low = 2.0**band
    step = band_low / SUB_BUCKETS
    sub = int((micros - band_low) / step)
    if sub >= SUB_BUCKETS:
        sub = SUB_BUCKETS - 1
    return band * SUB_BUCKETS + sub


def _bucket_bounds(index: int) -> tuple[float, float]:
    """Return the low and high microsecond edge of a bucket index."""
    band, sub = divmod(index, SUB_BUCKETS)
    band_low = 2.0**band
    step = band_low / SUB_BUCKETS
    low = band_low + sub * step
    return low, low + step


class LatencyHistogram:
    """A sparse count-per-bucket histogram over durations in seconds."""

    __slots__ = ("_counts", "_max_s", "_min_s", "_total")

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._total = 0
        self._min_s = math.inf
        self._max_s = 0.0

    def record(self, seconds: float) -> None:
        """Fold one latency into the histogram."""
        micros = seconds * 1_000_000.0
        index = _bucket_index(micros)
        self._counts[index] = self._counts.get(index, 0) + 1
        self._total += 1
        if seconds < self._min_s:
            self._min_s = seconds
        if seconds > self._max_s:
            self._max_s = seconds

    def merge(self, other: LatencyHistogram) -> None:
        """Add another histogram into this one, in place."""
        for index, count in other._counts.items():
            self._counts[index] = self._counts.get(index, 0) + count
        self._total += other._total
        if other._total:
            self._min_s = min(self._min_s, other._min_s)
            self._max_s = max(self._max_s, other._max_s)

    @property
    def count(self) -> int:
        return self._total

    @property
    def min_s(self) -> float:
        return 0.0 if self._total == 0 else self._min_s

    @property
    def max_s(self) -> float:
        return self._max_s

    def percentile(self, pct: float) -> float:
        """Return an approximate percentile in seconds.

        This walks the buckets in order and returns the high edge of the bucket
        that carries the target rank. The exact min and max clamp the ends.
        """
        if self._total == 0:
            return 0.0

        target = (pct / 100.0) * self._total
        seen = 0
        for index in sorted(self._counts):
            seen += self._counts[index]
            if seen >= target:
                _, high = _bucket_bounds(index)
                value = high / 1_000_000.0
                return min(max(value, self.min_s), self._max_s)
        return self._max_s

    def mean_s(self) -> float:
        """Return an approximate mean in seconds from the bucket midpoints."""
        if self._total == 0:
            return 0.0
        weighted = 0.0
        for index, count in self._counts.items():
            low, high = _bucket_bounds(index)
            weighted += ((low + high) / 2.0) * count
        return weighted / self._total / 1_000_000.0

    def bars(self, buckets: int = 40) -> list[tuple[float, float, int]]:
        """Return up to `buckets` merged bins as (low_s, high_s, count).

        The raw sub-buckets are too many to chart, so this groups them into a
        fixed number of even ranges across the observed span. An HTML or console
        chart draws these directly.
        """
        if self._total == 0:
            return []

        low = self.min_s
        high = self._max_s
        if high <= low:
            return [(low, high, self._total)]

        width = (high - low) / buckets
        merged = [0] * buckets
        for index, count in self._counts.items():
            b_low, b_high = _bucket_bounds(index)
            mid_s = ((b_low + b_high) / 2.0) / 1_000_000.0
            slot = int((mid_s - low) / width)
            if slot < 0:
                slot = 0
            if slot >= buckets:
                slot = buckets - 1
            merged[slot] += count

        return [(low + i * width, low + (i + 1) * width, merged[i]) for i in range(buckets)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the histogram for the JSON report.

        The bars draw a chart. The buckets are the lossless form, which lets
        another machine rebuild this histogram with from_dict. A JSON object
        needs string keys, so each bucket index becomes a string here.
        """
        return {
            "count": self._total,
            "min_ms": round(self.min_s * 1000, 4),
            "max_ms": round(self._max_s * 1000, 4),
            "buckets": {str(index): self._counts[index] for index in sorted(self._counts)},
            "bars": [
                {"low_ms": round(low * 1000, 4), "high_ms": round(high * 1000, 4), "count": count}
                for low, high, count in self.bars()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LatencyHistogram:
        """Rebuild a histogram from a to_dict payload.

        The 40 chart bars are too coarse to merge percentiles from, so this
        reads the buckets instead. The payload comes from parsed JSON, so the
        string bucket indices parse back to int here.
        """
        hist = cls()
        buckets: dict[str, Any] = payload.get("buckets") or {}
        for raw_index, raw_count in buckets.items():
            count = int(raw_count)
            if count <= 0:
                continue
            index = int(raw_index)
            hist._counts[index] = hist._counts.get(index, 0) + count
            hist._total += count

        if hist._total == 0:
            return hist

        # The payload holds the exact ends, so they survive the trip. Bucket
        # edges stand in when a payload leaves them out, because a zero max
        # would clamp every percentile to zero.
        low, _ = _bucket_bounds(min(hist._counts))
        _, high = _bucket_bounds(max(hist._counts))
        min_ms = payload.get("min_ms")
        max_ms = payload.get("max_ms")
        hist._min_s = low / 1_000_000.0 if min_ms is None else float(min_ms) / 1000.0
        hist._max_s = high / 1_000_000.0 if max_ms is None else float(max_ms) / 1000.0
        return hist
