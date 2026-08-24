"""Tests for distributed runs and the report diff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from testbuster.cluster import (
    NO_WORKERS,
    build_worker_app,
    combine_reports,
    dispatch,
    plan_from_wire,
    plan_to_wire,
)
from testbuster.config import RunPlan
from testbuster.diff import compare, find_regressions, load_report
from testbuster.errors import ExitCode, TestBusterError
from testbuster.histogram import LatencyHistogram


def _wire_histogram(samples_ms: list[float]) -> dict[str, Any]:
    """Build the histogram payload a worker sends, from latencies in ms."""
    hist = LatencyHistogram()
    for value in samples_ms:
        hist.record(value / 1000.0)
    return hist.to_dict()


def _latency_report(samples_ms: list[float]) -> dict[str, Any]:
    """A worker reply that carries only the given latencies, all successful."""
    count = len(samples_ms)
    return {
        "schema": "testbuster/report/1",
        "summary": {
            "total_requests": count,
            "successful": count,
            "failed": 0,
            "wall_seconds": 1.0,
            "bytes_in": 0,
            "interrupted": False,
        },
        "plan": {"target": "http://x/", "workers": 1},
        "histogram": _wire_histogram(samples_ms),
        "status_codes": {"200": count},
        "failures": {},
    }


def _worker_report(*, total: int, ok: int, p95_ms: float, wall: float = 1.0) -> dict[str, Any]:
    # Latencies spread evenly from 1ms up to p95_ms, so the buckets carry a real
    # spread rather than one spike.
    step = (p95_ms - 1.0) / total if total > 1 else 0.0
    return {
        "schema": "testbuster/report/1",
        "summary": {
            "total_requests": total,
            "successful": ok,
            "failed": total - ok,
            "retried": 0,
            "wall_seconds": wall,
            "bytes_in": total * 100,
            "interrupted": False,
        },
        "plan": {"target": "http://x/", "workers": 5},
        "histogram": _wire_histogram([1.0 + step * i for i in range(total)]),
        "status_codes": {"200": ok, "500": total - ok},
        "failures": {},
    }


class TestCombine:
    def test_sums_totals(self) -> None:
        merged = combine_reports(
            [
                _worker_report(total=100, ok=100, p95_ms=20),
                _worker_report(total=50, ok=40, p95_ms=30),
            ]
        )
        assert merged["summary"]["total_requests"] == 150
        assert merged["summary"]["successful"] == 140
        assert merged["summary"]["failed"] == 10

    def test_success_rate_recomputed(self) -> None:
        merged = combine_reports(
            [
                _worker_report(total=100, ok=90, p95_ms=20),
                _worker_report(total=100, ok=100, p95_ms=20),
            ]
        )
        assert merged["summary"]["success_rate_pct"] == pytest.approx(95.0)

    def test_wall_is_the_slowest_worker(self) -> None:
        merged = combine_reports(
            [
                _worker_report(total=10, ok=10, p95_ms=20, wall=1.0),
                _worker_report(total=10, ok=10, p95_ms=20, wall=3.0),
            ]
        )
        assert merged["summary"]["wall_seconds"] == pytest.approx(3.0)

    def test_merges_status_codes(self) -> None:
        merged = combine_reports(
            [_worker_report(total=10, ok=8, p95_ms=20), _worker_report(total=10, ok=9, p95_ms=20)]
        )
        assert merged["status_codes"] == {"200": 17, "500": 3}

    def test_latency_comes_from_the_merged_histogram(self) -> None:
        merged = combine_reports(
            [
                _worker_report(total=100, ok=100, p95_ms=10),
                _worker_report(total=100, ok=100, p95_ms=90),
            ]
        )
        # The merged p95 lands between the two workers' own spans.
        assert 10 <= merged["latency"]["p95"] <= 90

    def test_no_reports_raises(self) -> None:
        with pytest.raises(TestBusterError, match="usable report"):
            combine_reports([])

    def test_skips_a_reply_with_a_broken_summary(self) -> None:
        merged = combine_reports(
            [_worker_report(total=10, ok=10, p95_ms=20), {"summary": "not a dict"}]
        )
        assert merged["merged_from"] == 1
        assert merged["summary"]["total_requests"] == 10

    def test_a_summary_that_misses_fields_is_not_usable(self) -> None:
        # Only the top-level shape is right. The merge must not index blindly
        # into it and hand the user a traceback.
        with pytest.raises(TestBusterError, match="usable report") as caught:
            combine_reports([{"summary": {"total_requests": 5}}])
        assert caught.value.code is ExitCode.RUN_FAILED

    def test_records_the_worker_count(self) -> None:
        merged = combine_reports([_worker_report(total=1, ok=1, p95_ms=5)])
        assert merged["merged_from"] == 1


class TestWire:
    def test_round_trips_a_plan(self) -> None:
        plan = RunPlan(target="http://x/", workers=25, total_requests=500, timeout=12.0)
        rebuilt = plan_from_wire(plan_to_wire(plan, total_requests=250))
        assert rebuilt.target == "http://x/"
        assert rebuilt.workers == 25
        assert rebuilt.total_requests == 250
        assert rebuilt.compact_memory is True

    def test_does_not_send_compact_memory(self) -> None:
        # plan_from_wire forces the flag on, so sending it is dead weight.
        wire = plan_to_wire(RunPlan(target="http://x/"), total_requests=1)
        assert "compact_memory" not in wire


class TestLoopbackRoundTrip:
    async def test_worker_runs_a_share_and_coordinator_merges(self) -> None:
        # A tiny target the workers hit.
        async def ok(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        target_app = web.Application()
        target_app.router.add_get("/ok", ok)
        target = TestServer(target_app)
        await target.start_server()

        worker_a = TestServer(build_worker_app())
        worker_b = TestServer(build_worker_app())
        await worker_a.start_server()
        await worker_b.start_server()

        try:
            target_url = f"http://127.0.0.1:{target.port}/ok"
            workers = [
                f"http://127.0.0.1:{worker_a.port}",
                f"http://127.0.0.1:{worker_b.port}",
            ]
            plan = RunPlan(target=target_url, workers=4, total_requests=40)
            merged = await dispatch(plan, workers)

            assert merged["merged_from"] == 2
            assert merged["summary"]["total_requests"] == 40
            assert merged["summary"]["successful"] == 40
        finally:
            await target.close()
            await worker_a.close()
            await worker_b.close()

    async def test_an_unreachable_worker_raises(self) -> None:
        plan = RunPlan(target="http://127.0.0.1:1/", workers=1, total_requests=4)
        with pytest.raises(TestBusterError, match="no worker could be reached"):
            await dispatch(plan, ["http://127.0.0.1:2"])

    async def test_no_worker_url_raises_the_shared_message(self) -> None:
        # The CLI prints the same words when the flag is missing.
        plan = RunPlan(target="http://127.0.0.1:1/", workers=1, total_requests=4)
        with pytest.raises(TestBusterError) as caught:
            await dispatch(plan, [])
        assert str(caught.value) == NO_WORKERS


def _report(*, success: float, rps: float, p95: float, p99: float = 0.0) -> dict[str, Any]:
    return {
        "schema": "testbuster/report/1",
        "summary": {"success_rate_pct": success, "requests_per_second": rps},
        "latency": {"p50": p95 / 2, "p95": p95, "p99": p99 or p95, "p99_9": p99 or p95},
    }


class TestCompare:
    def test_lists_metrics_in_a_fixed_order(self) -> None:
        deltas = compare(_report(success=100, rps=10, p95=50), _report(success=100, rps=10, p95=50))
        names = [d.name for d in deltas]
        assert names == [
            "success rate",
            "throughput",
            "latency p50",
            "latency p95",
            "latency p99",
            "latency p99_9",
        ]

    def test_percent_change(self) -> None:
        deltas = compare(
            _report(success=100, rps=100, p95=50), _report(success=100, rps=150, p95=50)
        )
        throughput = next(d for d in deltas if d.name == "throughput")
        assert throughput.percent == pytest.approx(50.0)

    def test_latency_regression_direction(self) -> None:
        deltas = compare(_report(success=100, rps=10, p95=50), _report(success=100, rps=10, p95=80))
        p95 = next(d for d in deltas if d.name == "latency p95")
        assert p95.is_regression is True  # higher latency is worse

    def test_latency_improvement_is_not_a_regression(self) -> None:
        deltas = compare(_report(success=100, rps=10, p95=80), _report(success=100, rps=10, p95=50))
        p95 = next(d for d in deltas if d.name == "latency p95")
        assert p95.is_regression is False

    def test_success_drop_is_a_regression(self) -> None:
        deltas = compare(_report(success=100, rps=10, p95=50), _report(success=90, rps=10, p95=50))
        rate = next(d for d in deltas if d.name == "success rate")
        assert rate.is_regression is True  # lower success is worse


class TestRegressions:
    def test_tolerance_absorbs_small_moves(self) -> None:
        deltas = compare(
            _report(success=100, rps=10, p95=100), _report(success=100, rps=10, p95=105)
        )
        # A 5 percent latency rise is inside a 10 percent tolerance.
        assert find_regressions(deltas, 10.0) == []

    def test_a_big_move_trips_the_tolerance(self) -> None:
        deltas = compare(
            _report(success=100, rps=10, p95=100), _report(success=100, rps=10, p95=200)
        )
        flagged = find_regressions(deltas, 10.0)
        assert any(d.name == "latency p95" for d in flagged)


class TestLoadReport:
    def test_reads_a_report(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps(_report(success=100, rps=10, p95=50)), encoding="utf-8")
        assert load_report(path)["schema"] == "testbuster/report/1"

    def test_rejects_a_foreign_file(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        with pytest.raises(TestBusterError, match="not a Test Buster"):
            load_report(path)

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(TestBusterError, match="cannot read report"):
            load_report(tmp_path / "absent.json")


class TestMergedMeanIsAMean:
    """Finding 10: the merged mean_ms must be a mean, not the median."""

    def test_right_skew_mean_exceeds_median(self) -> None:
        # One report, heavily right-skewed: many fast, a few very slow. The mean
        # must sit above the p50, which it cannot if mean_ms just copies p50.
        merged = combine_reports([_latency_report([10.0] * 950 + [1000.0] * 50)])
        assert merged["latency"]["mean_ms"] > merged["latency"]["p50"]


class TestMergedPercentilesKeepResolution:
    """The wire form must carry the raw buckets, not only the 40 chart bars.

    A skewed run holds nearly every sample in a narrow fast band. A thin tail
    then stretches the span by a factor of hundreds. Forty even bars across that
    span drop the whole fast band into one wide bin. A merge that reads the bars
    therefore pins every low percentile to that bin midpoint, far above the truth.
    """

    def test_merged_percentiles_match_one_histogram(self) -> None:
        fast_a = [10.0] * 475
        fast_b = [12.0] * 475
        slow_a = [1000.0 + 160.0 * i for i in range(25)]
        slow_b = [1080.0 + 160.0 * i for i in range(25)]

        # What one machine reports over the very same 1000 samples.
        alone = LatencyHistogram()
        for value in fast_a + fast_b + slow_a + slow_b:
            alone.record(value / 1000.0)

        merged = combine_reports(
            [_latency_report(fast_a + slow_a), _latency_report(fast_b + slow_b)]
        )

        # The buckets fold with no loss, so only the rounding of the reported
        # ends can move a digit. That is why the tolerance is this tight.
        assert merged["latency"]["count"] == 1000
        assert merged["latency"]["p50"] == pytest.approx(alone.percentile(50) * 1000, rel=0.002)
        assert merged["latency"]["p99"] == pytest.approx(alone.percentile(99) * 1000, rel=0.002)
        assert merged["latency"]["max_ms"] == pytest.approx(alone.max_s * 1000, rel=1e-6)
