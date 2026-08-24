"""Tests for percentile math, the histogram, the Tally, and the Report."""

from __future__ import annotations

from array import array
from collections.abc import Callable

import pytest

from testbuster.config import Gates, RunPlan
from testbuster.histogram import LatencyHistogram
from testbuster.metrics import Attempt, Report, Spread, Tally, pct_key, percentile


class TestPercentile:
    def test_returns_zero_on_no_samples(self) -> None:
        assert percentile([], 95) == 0.0

    def test_returns_the_only_sample(self) -> None:
        assert percentile([4.2], 50) == 4.2

    def test_p0_and_p100_are_the_ends(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 4.0

    def test_interpolates_between_neighbours(self) -> None:
        # Ten samples, p95 sits at rank 0.95 * 9 = 8.55, so between 9 and 10.
        values = [float(n) for n in range(1, 11)]
        assert percentile(values, 95) == pytest.approx(9.55)

    def test_median_of_an_even_count_sits_in_the_middle(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_does_not_round_up_like_the_old_go_build(self) -> None:
        # A ceiling-index method would report 10.0 here. Interpolation does not.
        values = [float(n) for n in range(1, 11)]
        assert percentile(values, 99) < 10.0


class TestPctKey:
    @pytest.mark.parametrize(
        ("pct", "expected"),
        [(50.0, "p50"), (95.0, "p95"), (99.9, "p99_9"), (75.0, "p75")],
    )
    def test_names_a_percentile(self, pct: float, expected: str) -> None:
        assert pct_key(pct) == expected


class TestSpread:
    def test_empty_series_reports_zeros(self) -> None:
        spread = Spread.from_seconds(array("d"))
        assert spread.count == 0
        assert spread.max_ms == 0.0
        assert spread.percentiles_ms["p95"] == 0.0

    def test_converts_seconds_to_milliseconds(self) -> None:
        spread = Spread.from_seconds(array("d", [0.010, 0.020, 0.030]))
        assert spread.min_ms == pytest.approx(10.0)
        assert spread.max_ms == pytest.approx(30.0)
        assert spread.mean_ms == pytest.approx(20.0)

    def test_computes_the_standard_deviation(self) -> None:
        spread = Spread.from_seconds(array("d", [0.001, 0.001, 0.001]))
        assert spread.stdev_ms == pytest.approx(0.0)

    def test_sorts_before_measuring(self) -> None:
        forward = Spread.from_seconds(array("d", [0.1, 0.2, 0.3]))
        backward = Spread.from_seconds(array("d", [0.3, 0.1, 0.2]))
        assert forward.percentiles_ms == backward.percentiles_ms

    def test_serializes_every_percentile(self) -> None:
        payload = Spread.from_seconds(array("d", [0.010] * 10)).to_dict()
        for key in ("p50", "p75", "p90", "p95", "p99", "p99_9"):
            assert key in payload


class TestAttempt:
    @pytest.mark.parametrize(
        ("status", "failure", "expected"),
        [
            (200, None, True),
            (204, None, True),
            (299, None, True),
            (301, None, False),
            (404, None, False),
            (500, None, False),
            (200, "timeout", False),
            (0, "timeout", False),
        ],
    )
    def test_only_2xx_without_a_failure_counts_as_ok(
        self, status: int, failure: str | None, expected: bool
    ) -> None:
        attempt = Attempt(
            status=status,
            elapsed=0.01,
            bytes_in=0,
            failure=failure,
            ttfb=None,
            dns=None,
            connect=None,
            retries=0,
        )
        assert attempt.ok is expected


class TestTally:
    def test_counts_successes_and_failures(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt(status=200))
        tally.record(make_attempt(status=500))
        tally.record(make_attempt(status=0, failure="timeout"))

        assert tally.total == 3
        assert tally.succeeded == 1
        assert tally.failed == 2
        assert tally.failure_counts["timeout"] == 1

    def test_adds_up_the_bytes(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt(bytes_in=100))
        tally.record(make_attempt(bytes_in=250))
        assert tally.bytes_in == 350

    def test_counts_requests_that_needed_a_retry(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        tally = Tally()
        tally.record(make_attempt(retries=0))
        tally.record(make_attempt(retries=2))
        assert tally.retried == 1

    def test_drops_per_request_records_by_default(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        tally = Tally(keep_attempts=False)
        tally.record(make_attempt())
        assert tally._attempts == []
        assert tally.total == 1

    def test_keeps_per_request_records_on_request(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        tally = Tally(keep_attempts=True)
        tally.record(make_attempt())
        assert len(tally._attempts) == 1

    def test_skips_absent_phase_samples(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt(dns=None, connect=0.001))
        report = tally.summarize(RunPlan(target="example.com"), 1.0, interrupted=False)
        assert report.dns.count == 0
        assert report.connect.count == 1


class TestReport:
    @staticmethod
    def build(
        make_attempt: Callable[..., Attempt],
        *,
        statuses: list[int],
        wall: float = 2.0,
        plan: RunPlan | None = None,
    ) -> Report:
        tally = Tally()
        for index, status in enumerate(statuses):
            tally.record(make_attempt(status=status, elapsed=0.01 * (index + 1)))
        return tally.summarize(plan or RunPlan(target="example.com"), wall, interrupted=False)

    def test_computes_throughput(self, make_attempt: Callable[..., Attempt]) -> None:
        report = self.build(make_attempt, statuses=[200] * 10, wall=2.0)
        assert report.requests_per_second == pytest.approx(5.0)

    def test_survives_a_zero_length_run(self, make_attempt: Callable[..., Attempt]) -> None:
        # A zero-length wall time must not divide by zero.
        report = self.build(make_attempt, statuses=[200], wall=0.0)
        assert report.requests_per_second == 0.0
        assert report.throughput_bytes_per_second == 0.0

    def test_survives_an_empty_run(self) -> None:
        report = Tally().summarize(RunPlan(target="example.com"), 1.0, interrupted=False)
        assert report.total == 0
        assert report.success_rate == 0.0
        assert report.bytes_per_request == 0.0

    def test_success_and_failure_rates_add_to_one_hundred(
        self, make_attempt: Callable[..., Attempt]
    ) -> None:
        report = self.build(make_attempt, statuses=[200, 200, 200, 500])
        assert report.success_rate == pytest.approx(75.0)
        assert report.failure_rate == pytest.approx(25.0)

    def test_sorts_status_codes(self, make_attempt: Callable[..., Attempt]) -> None:
        report = self.build(make_attempt, statuses=[500, 200, 404, 301, 200])
        assert list(report.status_counts) == [200, 301, 404, 500]

    def test_orders_failures_by_frequency(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        for _ in range(3):
            tally.record(make_attempt(status=0, failure="timeout"))
        tally.record(make_attempt(status=0, failure="connection refused"))
        report = tally.summarize(RunPlan(target="example.com"), 1.0, interrupted=False)
        assert list(report.failure_counts) == ["timeout", "connection refused"]


class TestGateChecks:
    @staticmethod
    def report_with(gates: Gates, latencies_s: list[float], oks: int) -> Report:
        tally = Tally()
        for index, elapsed in enumerate(latencies_s):
            tally.record(
                Attempt(
                    status=200 if index < oks else 500,
                    elapsed=elapsed,
                    bytes_in=10,
                    failure=None,
                    ttfb=elapsed / 2,
                    dns=None,
                    connect=None,
                    retries=0,
                )
            )
        plan = RunPlan(target="example.com", gates=gates)
        return tally.summarize(plan, 1.0, interrupted=False)

    def test_no_gates_means_no_results(self) -> None:
        report = self.report_with(Gates(), [0.01] * 5, oks=5)
        assert report.check_gates() == []

    def test_a_fast_run_passes_the_p95_gate(self) -> None:
        report = self.report_with(Gates(max_p95_ms=100), [0.010] * 20, oks=20)
        gate = report.check_gates()[0]
        assert gate.passed is True
        assert gate.name == "p95 latency"

    def test_a_slow_run_fails_the_p95_gate(self) -> None:
        report = self.report_with(Gates(max_p95_ms=5), [0.010] * 20, oks=20)
        assert report.check_gates()[0].passed is False

    def test_a_healthy_run_passes_the_success_gate(self) -> None:
        report = self.report_with(Gates(min_success_rate=95), [0.01] * 100, oks=100)
        assert report.check_gates()[0].passed is True

    def test_an_unhealthy_run_fails_the_success_gate(self) -> None:
        report = self.report_with(Gates(min_success_rate=95), [0.01] * 100, oks=90)
        gate = report.check_gates()[0]
        assert gate.passed is False
        assert gate.actual == pytest.approx(90.0)

    def test_every_active_gate_reports(self) -> None:
        gates = Gates(max_p95_ms=100, max_p99_ms=200, min_success_rate=99)
        report = self.report_with(gates, [0.01] * 10, oks=10)
        assert [gate.name for gate in report.check_gates()] == [
            "p95 latency",
            "p99 latency",
            "success rate",
        ]


class TestReportSerialization:
    def test_carries_a_schema_version(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt())
        payload = tally.summarize(RunPlan(target="example.com"), 1.0, interrupted=False).to_dict(
            tool_version="9.9.9"
        )

        assert payload["schema"] == "testbuster/report/1"
        assert payload["tool"] == {"name": "Test Buster!", "version": "9.9.9"}

    def test_uses_snake_case_keys_with_units(self, make_attempt: Callable[..., Attempt]) -> None:
        # The schema is snake_case with millisecond units, so a documented parser
        tally = Tally()
        tally.record(make_attempt())
        payload = tally.summarize(RunPlan(target="example.com"), 1.0, interrupted=False).to_dict(
            tool_version="1.0.0"
        )

        assert "total_requests" in payload["summary"]
        assert "requests_per_second" in payload["summary"]
        assert "p95" in payload["latency"]
        assert payload["latency"]["min_ms"] == pytest.approx(10.0)

    def test_omits_attempts_unless_asked(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally(keep_attempts=False)
        tally.record(make_attempt())
        payload = tally.summarize(RunPlan(target="example.com"), 1.0, interrupted=False).to_dict(
            tool_version="1.0.0"
        )
        assert "attempts" not in payload

    def test_includes_attempts_when_asked(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally(keep_attempts=True)
        tally.record(make_attempt())
        plan = RunPlan(target="example.com", keep_attempts=True)
        payload = tally.summarize(plan, 1.0, interrupted=False).to_dict(tool_version="1.0.0")
        assert len(payload["attempts"]) == 1

    def test_records_an_interrupted_run(self, make_attempt: Callable[..., Attempt]) -> None:
        tally = Tally()
        tally.record(make_attempt())
        payload = tally.summarize(RunPlan(target="example.com"), 1.0, interrupted=True).to_dict(
            tool_version="1.0.0"
        )
        assert payload["summary"]["interrupted"] is True


class TestRecording:
    def test_empty_reports_zeros(self) -> None:
        hist = LatencyHistogram()
        assert hist.count == 0
        assert hist.percentile(95) == 0.0
        assert hist.min_s == 0.0
        assert hist.max_s == 0.0
        assert hist.bars() == []

    def test_tracks_exact_min_and_max(self) -> None:
        hist = LatencyHistogram()
        for value in (0.010, 0.500, 0.002, 0.100):
            hist.record(value)
        assert hist.min_s == pytest.approx(0.002)
        assert hist.max_s == pytest.approx(0.500)

    def test_counts_every_sample(self) -> None:
        hist = LatencyHistogram()
        for _ in range(1000):
            hist.record(0.05)
        assert hist.count == 1000

    def test_memory_stays_bounded(self) -> None:
        # A million samples over a wide range must not grow without limit.
        hist = LatencyHistogram()
        for micros in range(1, 1_000_00):
            hist.record(micros / 1_000_000)
        # Buckets are log-linear, so the distinct count is small.
        assert len(hist._counts) < 400


class TestPercentiles:
    def test_uniform_data_lands_near_the_rank(self) -> None:
        hist = LatencyHistogram()
        for milli in range(1, 1001):  # 1ms .. 1000ms
            hist.record(milli / 1000)
        # Within the histogram's relative error of the true 500ms and 950ms.
        assert hist.percentile(50) == pytest.approx(0.500, rel=0.1)
        assert hist.percentile(95) == pytest.approx(0.950, rel=0.1)

    def test_percentiles_do_not_exceed_the_max(self) -> None:
        hist = LatencyHistogram()
        for value in (0.01, 0.02, 0.03):
            hist.record(value)
        assert hist.percentile(99.9) <= hist.max_s

    def test_mean_is_close(self) -> None:
        hist = LatencyHistogram()
        for _ in range(500):
            hist.record(0.020)
        assert hist.mean_s() == pytest.approx(0.020, rel=0.1)


class TestBars:
    def test_returns_the_requested_bucket_count(self) -> None:
        hist = LatencyHistogram()
        for milli in range(1, 201):
            hist.record(milli / 1000)
        bars = hist.bars(buckets=20)
        assert len(bars) == 20
        assert sum(count for _, _, count in bars) == 200

    def test_bars_span_min_to_max(self) -> None:
        hist = LatencyHistogram()
        hist.record(0.010)
        hist.record(0.100)
        bars = hist.bars(buckets=10)
        assert bars[0][0] == pytest.approx(hist.min_s)
        assert bars[-1][1] == pytest.approx(hist.max_s)


class TestMerge:
    def test_merge_sums_counts(self) -> None:
        a = LatencyHistogram()
        b = LatencyHistogram()
        for _ in range(100):
            a.record(0.01)
        for _ in range(50):
            b.record(0.02)
        a.merge(b)
        assert a.count == 150

    def test_merge_widens_the_range(self) -> None:
        a = LatencyHistogram()
        b = LatencyHistogram()
        a.record(0.05)
        b.record(0.001)
        b.record(0.500)
        a.merge(b)
        assert a.min_s == pytest.approx(0.001)
        assert a.max_s == pytest.approx(0.500)

    def test_merge_with_empty_is_a_noop(self) -> None:
        a = LatencyHistogram()
        a.record(0.03)
        a.merge(LatencyHistogram())
        assert a.count == 1


class TestSerialization:
    def test_to_dict_shape(self) -> None:
        hist = LatencyHistogram()
        for milli in range(1, 51):
            hist.record(milli / 1000)
        payload = hist.to_dict()
        assert payload["count"] == 50
        assert "min_ms" in payload and "max_ms" in payload
        assert isinstance(payload["bars"], list)
        assert all({"low_ms", "high_ms", "count"} <= set(bar) for bar in payload["bars"])
