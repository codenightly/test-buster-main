"""Engine tests against a real loopback server."""

from __future__ import annotations

import asyncio
import importlib.util
import time

import aiohttp
import pytest
from tests.conftest import StubServer

from testbuster.config import Gates, LoadProfile, RunPlan
from testbuster.engine import LoadEngine, RateLimiter, classify_failure, execute
from testbuster.errors import TestBusterError
from testbuster.metrics import NO_RESPONSE, Attempt, Report
from testbuster.sources import RequestSource, RequestSpec, Step, from_rows, from_steps
from testbuster.validation import NO_EXPECTATIONS, build_expectations

_HAS_HTTPX = importlib.util.find_spec("httpx") is not None


async def _run(plan: RunPlan, source: RequestSource | None = None) -> Report:
    """Run a plan with no progress hook and return the report."""
    return await LoadEngine(plan, source=source).run()


class TestBasicRun:
    async def test_sends_the_requested_count(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=25, workers=5)
        report = await _run(plan)

        assert report.total == 25
        assert report.succeeded == 25
        assert report.failed == 0
        assert server.hits["ok"] == 25

    async def test_never_overshoots_the_request_cap(self, server: StubServer) -> None:
        # 7 requests over 5 workers does not divide evenly. The shared counter
        # has to stop the pool at exactly 7.
        plan = RunPlan(target=server.url("/ok"), total_requests=7, workers=5)
        report = await _run(plan)

        assert report.total == 7
        assert server.hits["ok"] == 7

    async def test_a_single_worker_still_works(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=3, workers=1)
        assert (await _run(plan)).total == 3

    async def test_more_workers_than_requests_is_safe(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=2, workers=20)
        report = await _run(plan)
        assert report.total == 2
        assert server.hits["ok"] == 2

    async def test_counts_the_body_bytes(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/payload"), total_requests=4, workers=2)
        report = await _run(plan)
        assert report.bytes_in == 4 * 4096
        assert report.bytes_per_request == pytest.approx(4096.0)

    async def test_measures_time_to_first_byte(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=5, workers=1)
        report = await _run(plan)

        assert report.ttfb.count == 5
        # Headers always arrive before the body finishes.
        assert report.ttfb.mean_ms <= report.latency.mean_ms + 0.001


class TestStatusHandling:
    async def test_counts_a_500_as_a_failure(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/boom"), total_requests=5, workers=1)
        report = await _run(plan)

        assert report.succeeded == 0
        assert report.failed == 5
        assert report.status_counts == {500: 5}
        # A 500 is a real response, so it carries no transport failure.
        assert report.failure_counts == {}

    async def test_records_the_error_body_bytes(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/boom"), total_requests=2, workers=1)
        report = await _run(plan)
        assert report.bytes_in > 0


class TestMethodsAndPayload:
    async def test_sends_the_chosen_method(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/echo"), method="POST", total_requests=1, workers=1)
        report = await _run(plan)
        assert report.succeeded == 1

    async def test_sends_the_body(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/echo"),
            method="POST",
            payload='{"hello": "world"}',
            total_requests=1,
            workers=1,
        )
        report = await _run(plan)
        assert report.succeeded == 1
        assert report.bytes_in > 0

    async def test_sets_a_default_user_agent(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/headers"), total_requests=1, workers=1)
        assert (await _run(plan)).succeeded == 1

    async def test_keeps_a_caller_supplied_user_agent(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/headers"),
            headers={"User-Agent": "mine/1.0"},
            total_requests=1,
            workers=1,
        )
        assert (await _run(plan)).succeeded == 1


class TestStopConditions:
    async def test_duration_stops_the_run(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=None, duration=0.5, workers=4)
        started = time.perf_counter()
        report = await _run(plan)
        spent = time.perf_counter() - started

        assert report.total > 0
        # Allow one in-flight request to finish past the deadline.
        assert spent < 2.0
        assert report.wall_seconds >= 0.4

    async def test_the_request_cap_can_win_the_race(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=5, duration=60.0, workers=2)
        report = await _run(plan)
        assert report.total == 5

    async def test_the_deadline_can_win_the_race(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/slow"), total_requests=10_000, duration=0.5, workers=2)
        report = await _run(plan)
        assert report.total < 10_000


class TestRetries:
    async def test_retries_a_503_and_recovers(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/flaky"),
            total_requests=1,
            workers=1,
            retries=3,
            retry_backoff=0.001,
        )
        report = await _run(plan)

        assert report.succeeded == 1
        assert report.retried == 1
        assert server.hits["flaky"] == 2

    async def test_no_retry_leaves_the_503(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/flaky"), total_requests=1, workers=1, retries=0)
        report = await _run(plan)

        assert report.failed == 1
        assert report.status_counts == {503: 1}
        assert server.hits["flaky"] == 1

    async def test_skips_a_status_outside_the_retry_set(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/boom"),
            total_requests=1,
            workers=1,
            retries=3,
            retry_backoff=0.001,
            retry_statuses=frozenset({429}),
        )
        await _run(plan)
        assert server.hits["boom"] == 1


class TestTimeouts:
    async def test_a_short_timeout_produces_a_timeout_failure(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/slow"), total_requests=2, workers=1, timeout=0.05)
        report = await _run(plan)

        assert report.succeeded == 0
        assert report.failure_counts == {"timeout": 2}
        assert report.status_counts == {NO_RESPONSE: 2}

    async def test_a_generous_timeout_lets_the_slow_route_finish(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/slow"), total_requests=1, workers=1, timeout=5.0)
        assert (await _run(plan)).succeeded == 1


class TestUnreachableTarget:
    async def test_a_closed_port_reports_a_connection_failure(self) -> None:
        # Port 1 on loopback refuses connections on every supported platform.
        plan = RunPlan(target="http://127.0.0.1:1/", total_requests=3, workers=1, timeout=2.0)
        report = await _run(plan)

        assert report.total == 3
        assert report.succeeded == 0
        assert report.status_counts == {NO_RESPONSE: 3}
        assert sum(report.failure_counts.values()) == 3

    async def test_one_dead_host_does_not_kill_the_pool(self) -> None:
        # Every request must be accounted for even when all of them fail.
        plan = RunPlan(target="http://127.0.0.1:1/", total_requests=8, workers=4, timeout=2.0)
        assert (await _run(plan)).total == 8


class TestRateLimit:
    async def test_holds_the_run_to_the_requested_rate(self, server: StubServer) -> None:
        # 20 requests at 20/s: 2 come free from the burst, the other 18 pace out
        # over about 0.9 s. Loopback would otherwise finish this in milliseconds,
        # so a fast result here means the limiter did nothing.
        plan = RunPlan(target=server.url("/ok"), total_requests=20, workers=8, rate_limit=20.0)
        started = time.perf_counter()
        report = await _run(plan)
        spent = time.perf_counter() - started

        assert report.total == 20
        assert spent >= 0.6
        assert report.requests_per_second <= 40.0

    async def test_no_rate_limit_runs_flat_out(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=20, workers=8)
        started = time.perf_counter()
        await _run(plan)
        assert time.perf_counter() - started < 0.6


class TestWarmup:
    async def test_warmup_requests_stay_out_of_the_report(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=5, workers=2, warmup=4)
        report = await _run(plan)

        assert report.total == 5
        assert server.hits["ok"] == 9  # 4 warmup plus 5 measured


class TestKeepAlive:
    async def test_reuse_shows_up_as_zero_connect_time(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=10, workers=1, keepalive=True)
        report = await _run(plan)
        assert report.connect.count == 10
        assert report.connect.min_ms == pytest.approx(0.0)

    async def test_no_keepalive_still_completes(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=6, workers=2, keepalive=False)
        report = await _run(plan)
        assert report.total == 6
        assert report.succeeded == 6


class TestAttemptRetention:
    async def test_keeps_one_record_per_request(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=6, workers=2, keep_attempts=True)
        report = await _run(plan)
        assert len(report.attempts) == 6

    async def test_drops_records_by_default(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=6, workers=2)
        assert (await _run(plan)).attempts == ()


class TestProgressHook:
    async def test_the_hook_sees_the_run_finish(self, server: StubServer) -> None:
        seen: list[int] = []
        plan = RunPlan(target=server.url("/ok"), total_requests=20, workers=4)

        report = await LoadEngine(plan).run(lambda snapshot: seen.append(snapshot.finished))

        assert report.total == 20
        # The engine pushes a final snapshot after the workers stop.
        assert seen[-1] == 20


class TestSyncEntryPoint:
    def test_execute_runs_a_plan_from_sync_code(self, unused_tcp_port: int) -> None:
        # No server here. A refused connection is enough to prove the wiring.
        plan = RunPlan(
            target=f"http://127.0.0.1:{unused_tcp_port}/",
            total_requests=2,
            workers=1,
            timeout=2.0,
        )
        report = execute(plan)
        assert report.total == 2


class TestSocksGuard:
    async def test_a_socks_proxy_without_the_extra_fails_clearly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "aiohttp_socks":
                raise ImportError("blocked for the test")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)

        plan = RunPlan(
            target="http://127.0.0.1:1/", proxy="socks5://127.0.0.1:1080", total_requests=1
        )
        with pytest.raises(TestBusterError, match="socks extra"):
            await _run(plan)


class TestRateLimiterUnit:
    async def test_the_first_token_is_immediate(self) -> None:
        limiter = RateLimiter(10.0)
        started = time.perf_counter()
        await limiter.take()
        assert time.perf_counter() - started < 0.05

    async def test_a_tiny_rate_still_allows_one_request(self) -> None:
        # Without the floor on the bucket, --rate 0.5 would hold half a token
        # and the very first request would wait a whole second.
        limiter = RateLimiter(0.5)
        started = time.perf_counter()
        assert await limiter.take() is True
        assert time.perf_counter() - started < 0.05

    async def test_it_spaces_tokens_once_the_burst_is_spent(self) -> None:
        # A full second of burst would let a short run ignore --rate entirely.
        # At 100/s the bucket holds a tenth of a second, so ten tokens are free.
        limiter = RateLimiter(100.0)
        started = time.perf_counter()
        for _ in range(10):
            await limiter.take()
        assert time.perf_counter() - started < 0.05

        started = time.perf_counter()
        await limiter.take()
        # The eleventh token needs about 1/100 s. Allow a generous floor.
        assert time.perf_counter() - started >= 0.005

    async def test_a_rate_rise_does_not_bank_the_idle_time(self) -> None:
        # The idle wait belongs to the old rate. Crediting it at the new rate
        # would fill the bucket and fire a burst the run never earned.
        limiter = RateLimiter(1.0)
        await limiter.take()  # spend the one starting token
        await asyncio.sleep(0.1)
        limiter.set_rate(1000.0)

        started = time.perf_counter()
        for _ in range(10):
            await limiter.take()
        # Ten tokens at 1000/s need about 10 ms. A banked burst would be free.
        assert time.perf_counter() - started >= 0.005


class TestFailureClassifier:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (TimeoutError(), "timeout"),
            (aiohttp.ServerDisconnectedError(), "server disconnected"),
            (aiohttp.ClientPayloadError(), "truncated response"),
            (aiohttp.InvalidURL("http://["), "invalid url"),
        ],
    )
    def test_names_known_failures(self, exc: BaseException, expected: str) -> None:
        assert classify_failure(exc) == expected

    def test_falls_back_to_the_exception_name(self) -> None:
        assert classify_failure(RuntimeError("odd")) == "RuntimeError"

    def test_reports_the_errno_for_a_bare_os_error(self) -> None:
        assert "os error" in classify_failure(OSError(9, "bad fd"))


@pytest.fixture
def unused_tcp_port() -> int:
    """Return a port with nothing listening on it."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


class TestMaxErrors:
    async def test_stops_near_the_failure_cap(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/boom"), total_requests=10_000, workers=4, max_errors=5)
        report = await _run(plan)

        # Every /boom is a failure, so the cap trips quickly. In-flight requests
        # may push the count a little past the cap, bounded by the worker count.
        assert report.failed >= 5
        assert report.total < 10_000
        assert report.stop_reason == "max errors reached"

    async def test_a_healthy_run_never_trips_the_cap(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=50, workers=5, max_errors=5)
        report = await _run(plan)
        assert report.total == 50
        assert report.stop_reason == "completed"


class TestValidation:
    async def test_a_wrong_status_expectation_fails_a_2xx(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/ok"),
            total_requests=10,
            workers=2,
            expectations=build_expectations(
                status=["500"], body_regex=None, json_specs=None, max_latency_s=None
            ),
        )
        report = await _run(plan)
        # The transport worked and the status was 200, but the check failed.
        assert report.succeeded == 0
        assert report.status_counts == {200: 10}
        assert any("status" in reason for reason in report.failure_counts)

    async def test_a_matching_expectation_passes(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/ok"),
            total_requests=10,
            workers=2,
            expectations=build_expectations(
                status=["2xx"], body_regex="hello", json_specs=None, max_latency_s=None
            ),
        )
        report = await _run(plan)
        assert report.succeeded == 10

    async def test_a_latency_budget_can_fail_a_slow_response(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/slow"),
            total_requests=2,
            workers=1,
            timeout=5.0,
            expectations=build_expectations(
                status=None, body_regex=None, json_specs=None, max_latency_s=0.05
            ),
        )
        report = await _run(plan)
        assert report.succeeded == 0
        assert any("budget" in reason for reason in report.failure_counts)


class TestDataFileSource:
    async def test_each_row_hits_its_own_path(self, server: StubServer) -> None:
        rows = [{"path": "ok"}, {"path": "boom"}]
        source = from_rows(
            rows,
            method="GET",
            url_template=server.url("/{{path}}"),
            header_templates={},
            body_template=None,
            expectations=NO_EXPECTATIONS,
        )
        plan = RunPlan(target=server.url("/ok"), total_requests=10, workers=2)
        report = await _run(plan, source=source)

        assert report.total == 10
        # Half hit /ok (200), half hit /boom (500).
        assert server.hits["ok"] == 5
        assert server.hits["boom"] == 5


class TestScenario:
    async def test_runs_weighted_steps_with_a_breakdown(self, server: StubServer) -> None:
        source = from_steps(
            [
                Step(RequestSpec("GET", server.url("/ok"), label="read"), 3),
                Step(RequestSpec("GET", server.url("/payload"), label="download"), 1),
            ]
        )
        plan = RunPlan(target=server.url("/ok"), total_requests=40, workers=4)
        report = await _run(plan, source=source)

        assert report.total == 40
        assert set(report.by_label) == {"read", "download"}
        assert report.by_label["read"].total == 30
        assert report.by_label["download"].total == 10


class TestProfile:
    async def test_a_ramp_paces_the_run(self, server: StubServer) -> None:
        profile = LoadProfile("ramp", duration=1.0, start_rate=5, peak_rate=50)
        plan = RunPlan(
            target=server.url("/ok"), total_requests=None, duration=1.0, workers=10, profile=profile
        )
        report = await _run(plan)
        # A ramp from 5 to 50 over 1s averages far below flat-out loopback speed.
        assert 0 < report.total < 2000


class TestCompactMemory:
    async def test_keeps_no_sample_array_but_still_reports(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=200, workers=8, compact_memory=True)
        report = await _run(plan)

        assert report.total == 200
        assert report.latency.count == 200  # percentiles come from the histogram
        assert report.latency.percentiles_ms["p95"] > 0


class TestNdjsonHook:
    async def test_on_attempt_sees_every_request(self, server: StubServer) -> None:
        seen: list[Attempt] = []
        plan = RunPlan(target=server.url("/ok"), total_requests=15, workers=3)
        await LoadEngine(plan).run(on_attempt=seen.append)
        assert len(seen) == 15
        assert all(a.status == 200 for a in seen)


class TestCookies:
    async def test_a_cookie_run_still_completes(self, server: StubServer) -> None:
        plan = RunPlan(target=server.url("/ok"), total_requests=10, workers=2, cookies=True)
        assert (await _run(plan)).succeeded == 10


class TestGatesWithValidation:
    async def test_a_failing_check_can_trip_the_success_gate(self, server: StubServer) -> None:
        plan = RunPlan(
            target=server.url("/ok"),
            total_requests=20,
            workers=2,
            expectations=build_expectations(
                status=["404"], body_regex=None, json_specs=None, max_latency_s=None
            ),
            gates=Gates(min_success_rate=99),
        )
        report = await _run(plan)
        gate = next(g for g in report.check_gates() if g.name == "success rate")
        assert gate.passed is False


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx is not installed")
class TestHttp2Backend:
    async def test_runs_over_httpx(self, server: StubServer) -> None:
        # The loopback server is HTTP/1.1, but httpx speaks that too. This proves
        # the backend switch works and feeds the same tally.
        plan = RunPlan(target=server.url("/ok"), total_requests=10, workers=2, http2=True)
        report = await _run(plan)
        assert report.total == 10
        assert report.succeeded == 10

    async def test_httpx_reports_a_failure_cleanly(self) -> None:
        plan = RunPlan(
            target="http://127.0.0.1:1/", total_requests=3, workers=1, http2=True, timeout=2.0
        )
        report = await _run(plan)
        assert report.total == 3
        assert report.succeeded == 0


class TestRateLimiterDoesNotWedge:
    """Finding 1: a very low rate must not commit a worker to an hours-long sleep."""

    async def test_take_bails_when_asked_to_stop(self) -> None:
        limiter = RateLimiter(1e-6)  # capacity 1, then effectively no refill
        await limiter.take()  # consume the one token

        started = time.perf_counter()
        took = await limiter.take(should_stop=lambda: True)
        assert took is False
        # It returned promptly instead of sleeping for ~11 days.
        assert time.perf_counter() - started < 1.0

    async def test_take_returns_true_when_a_token_is_free(self) -> None:
        limiter = RateLimiter(100.0)
        assert await limiter.take(should_stop=lambda: False) is True

    async def test_a_zero_start_ramp_still_runs_and_ends(self, server: StubServer) -> None:
        # A ramp starting at zero used to wedge the run. It must finish near its
        # duration and deliver more than the single burst token.
        from testbuster.engine import LoadEngine

        profile = LoadProfile("ramp", duration=1.0, start_rate=0.0, peak_rate=200.0)
        plan = RunPlan(
            target=server.url("/ok"),
            total_requests=None,
            duration=1.0,
            workers=8,
            profile=profile,
        )
        started = time.perf_counter()
        report = await LoadEngine(plan).run()
        spent = time.perf_counter() - started

        assert spent < 3.0  # not wedged
        assert report.total > 1  # the ramp opened up past the first token


class TestRetriesRespectTheDeadline:
    """Finding 2: retries must stop at a duration deadline, not only at Ctrl+C."""

    async def test_retries_stop_when_the_deadline_passes(self, server: StubServer) -> None:
        from testbuster.engine import LoadEngine

        # /boom always 500, which is retriable. A tight deadline with slow
        # backoff must not let retries run far past it.
        plan = RunPlan(
            target=server.url("/boom"),
            total_requests=None,
            duration=0.5,
            workers=2,
            retries=10,
            retry_backoff=1.0,
        )
        started = time.perf_counter()
        await LoadEngine(plan).run()
        spent = time.perf_counter() - started

        # Without the deadline check, ten 1s-plus backoffs would run for many
        # seconds past the 0.5s deadline. One in-flight backoff is tolerated.
        # The bound carries slack: a loaded machine has pushed a run past 4s.
        # A run without the deadline check still clears 10s, so 6s keeps the
        # test able to tell the two apart.
        assert spent < 6.0
