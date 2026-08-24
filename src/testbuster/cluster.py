"""Distributed runs: several workers, one merged report.

One machine caps out well before most production systems do. A cluster spreads
the load across several worker processes, each a small HTTP service that runs a
share of the plan and returns its report. The coordinator merges the reports
into one.

Merging exact percentiles across machines is not possible without shipping every
sample. The histogram solves that: each worker sends its bucket counts, and the
merged percentiles come from the combined buckets. The numbers are approximate,
the same way the compact-memory mode is, and the merge cost stays bounded.
"""

from __future__ import annotations

import json
from typing import Any, Final

from testbuster import __version__
from testbuster.config import REPORTED_PERCENTILES, RunPlan
from testbuster.errors import ExitCode, TestBusterError
from testbuster.histogram import LatencyHistogram
from testbuster.metrics import pct_key

#: What a cluster run says when the command names no worker URL. The CLI checks
#: the flag before it builds a plan, so both places raise the same words.
NO_WORKERS: Final[str] = "a cluster run needs at least one --worker URL"

#: Summary fields the merge reads with no fallback. A reply that misses one of
#: these cannot join the merge, so the filter drops it.
_REQUIRED_SUMMARY: Final[frozenset[str]] = frozenset(
    {"total_requests", "successful", "failed", "wall_seconds"}
)


def plan_to_wire(plan: RunPlan, *, total_requests: int | None) -> dict[str, Any]:
    """Serialize the parts of a plan a worker needs to run its share."""
    return {
        "target": plan.target,
        "method": plan.method,
        "headers": plan.headers,
        "payload": plan.payload,
        "workers": plan.workers,
        "total_requests": total_requests,
        "duration": plan.duration,
        "timeout": plan.timeout,
        "verify_tls": plan.verify_tls,
        "follow_redirects": plan.follow_redirects,
        "keepalive": plan.keepalive,
        "retries": plan.retries,
    }


def plan_from_wire(data: dict[str, Any]) -> RunPlan:
    """Rebuild a RunPlan a worker will run. Compact memory is forced on."""
    return RunPlan(
        target=str(data["target"]),
        method=str(data.get("method", "GET")),
        headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
        payload=data.get("payload"),
        workers=int(data.get("workers", 10)),
        total_requests=data.get("total_requests"),
        duration=data.get("duration"),
        timeout=float(data.get("timeout", 30.0)),
        verify_tls=bool(data.get("verify_tls", True)),
        follow_redirects=bool(data.get("follow_redirects", True)),
        keepalive=bool(data.get("keepalive", True)),
        retries=int(data.get("retries", 0)),
        compact_memory=True,
    )


def _mapping(value: Any) -> dict[str, Any]:
    """Return value when it is a dict, else an empty dict.

    Every nested part of a worker reply comes from parsed JSON, so any of them
    may hold the wrong type.
    """
    return value if isinstance(value, dict) else {}


def _usable(report: Any) -> bool:
    """True when a reply holds the summary fields the merge indexes directly."""
    summary = _mapping(report).get("summary")
    return isinstance(summary, dict) and summary.keys() >= _REQUIRED_SUMMARY


def _merge_counts(reports: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Sum a dict-of-counts field across reports."""
    merged: dict[str, int] = {}
    for report in reports:
        for name, count in _mapping(report.get(key)).items():
            merged[str(name)] = merged.get(str(name), 0) + int(count)
    return merged


def combine_reports(reports: list[Any]) -> dict[str, Any]:
    """Merge several worker reports into one report-shaped dict.

    The input comes from parsed JSON, so each item is checked before use. The
    latency numbers come from the workers' bucket counts folded into one
    histogram, so they carry the same error a single machine reports. Counts and
    totals are exact.
    """
    valid = [r for r in reports if _usable(r)]
    if not valid:
        raise TestBusterError("no worker returned a usable report", ExitCode.RUN_FAILED)

    total = sum(int(r["summary"]["total_requests"]) for r in valid)
    succeeded = sum(int(r["summary"]["successful"]) for r in valid)
    failed = sum(int(r["summary"]["failed"]) for r in valid)
    retried = sum(int(r["summary"].get("retried", 0)) for r in valid)
    bytes_in = sum(int(r["summary"].get("bytes_in", 0)) for r in valid)
    wall = max(float(r["summary"]["wall_seconds"]) for r in valid)

    merged_hist = LatencyHistogram()
    for report in valid:
        merged_hist.merge(LatencyHistogram.from_dict(_mapping(report.get("histogram"))))

    latency: dict[str, Any] = {
        "count": merged_hist.count,
        "min_ms": round(merged_hist.min_s * 1000, 3),
        "mean_ms": round(merged_hist.mean_s() * 1000, 3),
        "max_ms": round(merged_hist.max_s * 1000, 3),
        "stdev_ms": 0.0,
        **{pct_key(p): round(merged_hist.percentile(p) * 1000, 3) for p in REPORTED_PERCENTILES},
    }

    first_plan = _mapping(valid[0].get("plan"))
    return {
        "schema": "testbuster/report/1",
        "tool": {"name": "Test Buster!", "version": __version__},
        "merged_from": len(valid),
        "plan": {
            "target": first_plan.get("target"),
            "method": first_plan.get("method"),
            "workers": sum(int(_mapping(r.get("plan")).get("workers", 0)) for r in valid),
        },
        "summary": {
            "total_requests": total,
            "successful": succeeded,
            "failed": failed,
            "retried": retried,
            "success_rate_pct": round(succeeded / total * 100, 4) if total else 0.0,
            "wall_seconds": round(wall, 6),
            "requests_per_second": round(total / wall, 4) if wall > 0 else 0.0,
            "bytes_in": bytes_in,
            "bytes_per_request": round(bytes_in / total, 2) if total else 0.0,
            "bytes_per_second": round(bytes_in / wall, 2) if wall > 0 else 0.0,
            "interrupted": any(r["summary"].get("interrupted") for r in valid),
            "stop_reason": "completed",
        },
        "latency": latency,
        "status_codes": dict(sorted(_merge_counts(valid, "status_codes").items())),
        "failures": _merge_counts(valid, "failures"),
        "gates": [],
    }


def build_worker_app() -> Any:
    """Return an aiohttp web app that runs a plan and returns its report.

    POST /run takes a wire plan and returns the report JSON. GET /health lets
    the coordinator check a worker is up before it farms out work.
    """
    from aiohttp import web

    from testbuster.engine import LoadEngine

    async def health(_request: Any) -> Any:
        return web.json_response({"status": "ok", "version": __version__})

    async def run(request: Any) -> Any:
        try:
            data = await request.json()
            plan = plan_from_wire(data)
        except (json.JSONDecodeError, TestBusterError, KeyError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        report = await LoadEngine(plan).run()
        return web.json_response(report.to_dict(tool_version=__version__))

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/run", run)
    return app


def run_worker(host: str, port: int) -> None:
    """Start a worker service. This blocks until the process is stopped."""
    from aiohttp import web

    web.run_app(build_worker_app(), host=host, port=port, print=None)


async def dispatch(plan: RunPlan, worker_urls: list[str]) -> dict[str, Any]:
    """Send a share of the plan to each worker and merge the reports."""
    import aiohttp

    if not worker_urls:
        raise TestBusterError(NO_WORKERS)

    count = len(worker_urls)
    shares: list[int | None]
    if plan.total_requests is not None:
        base, extra = divmod(plan.total_requests, count)
        shares = [base + (1 if i < extra else 0) for i in range(count)]
    else:
        # A duration run gives every worker the same length.
        shares = [None] * count

    async def one(url: str, share: int | None) -> dict[str, Any] | None:
        wire = plan_to_wire(plan, total_requests=share)
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"{url.rstrip('/')}/run", json=wire) as response,
            ):
                if response.status != 200:
                    return None
                result: dict[str, Any] = await response.json()
                return result
        except aiohttp.ClientError:
            return None

    import asyncio

    results = await asyncio.gather(
        *(one(url, share) for url, share in zip(worker_urls, shares, strict=True))
    )
    reports = [report for report in results if report is not None]
    if not reports:
        raise TestBusterError("no worker could be reached", ExitCode.RUN_FAILED)
    return combine_reports(reports)
