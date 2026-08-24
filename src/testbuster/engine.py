"""The load engine.

A task per request would allocate one task object for every request in the
plan. That wastes memory on large runs and adds scheduler churn.

Test Buster! instead starts exactly --concurrency worker tasks. Each worker
claims the next request from a shared counter and loops until a stop condition
fires. Memory stays flat whether the run is a hundred requests or ten million.

Two transports live here. aiohttp is the default and carries HTTP/1.1 with full
phase timing. httpx carries HTTP/2 when --http2 is set. Both feed the same
worker loop, the same retries, and the same tally.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import functools
import random
import signal
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import aiohttp

from testbuster import APP_NAME, __version__
from testbuster.config import RunPlan, require_http2_support, require_socks_support
from testbuster.errors import ExitCode, TestBusterError
from testbuster.metrics import NO_RESPONSE, Attempt, Report, Tally
from testbuster.sources import RequestSource, RequestSpec, single

#: Body chunk size while draining a response. 64 KiB keeps syscalls low
#: without holding a large buffer per worker.
DRAIN_CHUNK = 64 * 1024

#: Most bytes of a response body kept for a validation check. Everything past
#: this is counted and dropped, so one huge response cannot exhaust memory.
BODY_CAPTURE_LIMIT = 2 * 1024 * 1024

#: How often the progress callback runs, in seconds.
PROGRESS_INTERVAL = 0.1

#: How often a load profile updates the target rate, in seconds.
PROFILE_INTERVAL = 0.1

#: Longest single wait inside the rate limiter. Capping it means a rate change
#: from a load profile and a stop request are both seen within this window, even
#: when the current rate is very low.
MAX_LIMITER_SLEEP = 0.1

#: Longest single retry backoff, whatever the attempt number.
BACKOFF_CEILING = 5.0

#: Keeps the event loop turning so a Ctrl+C on Windows lands promptly.
WATCHDOG_INTERVAL = 0.2

#: How much burst the rate limiter allows, in seconds of traffic.
#: A textbook token bucket banks a full second, which would let `--rate 20 -n 20`
#: fire all twenty at once and finish at an observed 700 req/s. A tenth of a
#: second absorbs scheduler jitter and still paces the run honestly.
BURST_WINDOW = 0.1


class Progress(SimpleNamespace):
    """A snapshot handed to the progress callback while the run is live."""

    finished: int
    elapsed: float
    succeeded: int
    failed: int


ProgressHook = Callable[[Progress], None]
AttemptHook = Callable[[Attempt], None]


class RateLimiter:
    """A token bucket that caps requests per second across all workers.

    The bucket refills continuously. A worker that finds it empty sleeps for
    exactly the time the next token needs, so the workers do not spin. A load
    profile changes the rate over the run through set_rate.
    """

    __slots__ = ("_capacity", "_rate", "_tokens", "_updated")

    def __init__(self, rate: float) -> None:
        self._rate = max(1e-6, rate)
        self._capacity = max(1.0, self._rate * BURST_WINDOW)
        self._tokens = self._capacity
        self._updated = time.monotonic()

    def _refill(self, now: float) -> None:
        """Credit the tokens earned between the last look and now."""
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now

    def set_rate(self, rate: float) -> None:
        """Change the target rate. A load profile calls this over time.

        The idle time so far belongs to the old rate, so bank it first.
        Otherwise a rate rise pays that whole idle wait at the new rate. That
        hands out a burst the run never earned.
        """
        self._refill(time.monotonic())
        self._rate = max(1e-6, rate)
        self._capacity = max(1.0, self._rate * BURST_WINDOW)
        if self._tokens > self._capacity:
            self._tokens = self._capacity

    async def take(self, should_stop: Callable[[], bool] | None = None) -> bool:
        """Wait for a token. Return True when one was taken, False when stopping.

        Each wait is capped, so a rate change from a load profile and a stop
        request are both seen quickly. Without the cap, a very low rate would
        commit a worker to a sleep of hours that no later rate rise could break.
        """
        while True:
            self._refill(time.monotonic())

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            if should_stop is not None and should_stop():
                return False

            needed = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(min(needed, MAX_LIMITER_SLEEP))


def classify_failure(exc: BaseException) -> str:
    """Name a transport failure in a few words.

    Grouping errors keeps the failure table short. A run against a dead host
    should print one line with a count, not ten thousand identical tracebacks.
    """
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return "server disconnected"
    if isinstance(exc, aiohttp.ClientPayloadError):
        return "truncated response"
    if isinstance(exc, aiohttp.TooManyRedirects):
        return "too many redirects"
    if isinstance(exc, aiohttp.ClientConnectorCertificateError):
        return "tls certificate rejected"
    if isinstance(exc, aiohttp.ClientConnectorSSLError):
        return "tls handshake failed"
    if isinstance(exc, aiohttp.ClientConnectorDNSError):
        return "dns lookup failed"
    if isinstance(exc, aiohttp.ClientConnectorError):
        code = getattr(exc.os_error, "errno", None)
        if code == errno.ECONNREFUSED:
            return "connection refused"
        if code in {errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH}:
            return "host unreachable"
        return "connection failed"
    if isinstance(exc, aiohttp.ClientResponseError):
        return f"http error {exc.status}"
    if isinstance(exc, aiohttp.InvalidURL):
        return "invalid url"
    if isinstance(exc, aiohttp.ClientError):
        return type(exc).__name__
    if isinstance(exc, OSError):
        return f"os error {exc.errno}"
    return type(exc).__name__


def classify_httpx_failure(exc: BaseException) -> str:
    """Name an httpx transport failure. Mirrors classify_failure for HTTP/2."""
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        text = str(exc).lower()
        if "refused" in text:
            return "connection refused"
        if "name or service" in text or "resolve" in text or "getaddrinfo" in text:
            return "dns lookup failed"
        return "connection failed"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "server disconnected"
    if isinstance(exc, httpx.TooManyRedirects):
        return "too many redirects"
    if isinstance(exc, httpx.HTTPError):
        return type(exc).__name__
    if isinstance(exc, OSError):
        return f"os error {exc.errno}"
    return type(exc).__name__


def _build_tracer() -> aiohttp.TraceConfig:
    """Wire up per-request DNS and connect timing.

    aiohttp offers no hook for "headers received", so the engine measures time
    to first byte itself: session.request() returns once the status line and
    headers are parsed. DNS and connect need these hooks because they happen
    inside the connector, out of the caller's sight.
    """
    tracer = aiohttp.TraceConfig()

    def phase_hooks(key: str) -> tuple[Any, Any]:
        """Build the start and end hooks that time one phase into key.

        The closure holds the mark, so no hook call builds a string.
        """
        mark = f"{key}_start"

        async def start(_session: Any, ctx: Any, _params: Any) -> None:
            ctx.trace_request_ctx[mark] = time.perf_counter()

        async def end(_session: Any, ctx: Any, _params: Any) -> None:
            started = ctx.trace_request_ctx.get(mark)
            if started is not None:
                ctx.trace_request_ctx[key] = time.perf_counter() - started

        return start, end

    dns_start, dns_end = phase_hooks("dns")
    connect_start, connect_end = phase_hooks("connect")

    async def connection_reused(_session: Any, ctx: Any, _params: Any) -> None:
        # A reused socket skips the lookup and the handshake. Record zero so the
        # connect series shows how often the pool paid off.
        ctx.trace_request_ctx["connect"] = 0.0

    tracer.on_dns_resolvehost_start.append(dns_start)
    tracer.on_dns_resolvehost_end.append(dns_end)
    tracer.on_connection_create_start.append(connect_start)
    tracer.on_connection_create_end.append(connect_end)
    tracer.on_connection_reuseconn.append(connection_reused)
    return tracer


class LoadEngine:
    """Runs one RunPlan and returns a Report."""

    def __init__(self, plan: RunPlan, source: RequestSource | None = None) -> None:
        self._plan = plan
        self._source: RequestSource = source or single(_default_spec(plan))
        self._tally = Tally(keep_attempts=plan.keep_attempts, compact_memory=plan.compact_memory)

        self._profile = plan.profile
        if self._profile is not None:
            self._limiter: RateLimiter | None = RateLimiter(self._profile.rate_at(0.0))
        elif plan.rate_limit:
            self._limiter = RateLimiter(plan.rate_limit)
        else:
            self._limiter = None

        self._stop = asyncio.Event()
        self._on_attempt: AttemptHook | None = None
        self._signal_restores: list[Callable[[], object]] = []

        self._claimed = 0
        self._deadline: float | None = None
        self._started_at = 0.0
        self._quota: int | None = plan.total_requests
        self._stop_reason = "completed"

        # The transport and its client, set once run() picks a backend.
        self._backend = "aiohttp"
        self._client: Any = None

    # ------------------------------------------------------------------ public

    async def run(
        self,
        on_progress: ProgressHook | None = None,
        *,
        on_attempt: AttemptHook | None = None,
    ) -> Report:
        """Execute the plan. Return the Report even after a Ctrl+C."""
        plan = self._plan
        self._on_attempt = on_attempt
        self._tally.enable_labels(self._source.labels)

        if plan.uses_socks_proxy:
            require_socks_support()
        if plan.http2:
            require_http2_support()
            return await self._run_with_httpx(on_progress)
        return await self._run_with_aiohttp(on_progress)

    @property
    def interrupted(self) -> bool:
        return self._stop_reason == "interrupted"

    # ------------------------------------------------------------- aiohttp path

    async def _run_with_aiohttp(self, on_progress: ProgressHook | None) -> Report:
        plan = self._plan
        connector = self._build_connector()
        timeout = aiohttp.ClientTimeout(total=plan.timeout)
        # A load test is stateless by default, so responses do not set cookies.
        # --cookies turns on a jar shared across the run, for a login flow.
        jar = aiohttp.CookieJar() if plan.cookies else aiohttp.DummyCookieJar()

        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._session_headers(),
                trace_configs=[_build_tracer()],
                auto_decompress=True,
                cookie_jar=jar,
            ) as session:
                self._backend = "aiohttp"
                self._client = session
                return await self._drive(on_progress)
        except aiohttp.ClientError as exc:
            # A failure out here means the session itself could not be built,
            # not that a request failed. Requests never escape _single_shot.
            raise TestBusterError(
                f"cannot start the run: {classify_failure(exc)}", ExitCode.RUN_FAILED
            ) from exc

    def _build_connector(self) -> aiohttp.BaseConnector:
        plan = self._plan
        shared: dict[str, Any] = {
            "limit": plan.workers,
            "limit_per_host": plan.workers,
            "ttl_dns_cache": 300,
            "use_dns_cache": True,
            # ssl=False turns off verification. None keeps the default context.
            "ssl": None if plan.verify_tls else False,
        }
        # aiohttp rejects keepalive_timeout together with force_close, so the
        # two settings have to stay mutually exclusive.
        if plan.keepalive:
            shared["keepalive_timeout"] = 30.0
        else:
            shared["force_close"] = True

        if plan.proxy is not None and plan.uses_socks_proxy:
            # The proxy check reads plan.proxy directly, so the type stays narrow
            # for from_url, which takes a string rather than an optional one.
            from aiohttp_socks import ProxyConnector

            connector: aiohttp.BaseConnector = ProxyConnector.from_url(plan.proxy, **shared)
            return connector

        return aiohttp.TCPConnector(**shared)

    async def _send_aiohttp(self, spec: RequestSpec) -> Attempt:
        """Send one request over aiohttp and measure every phase."""
        plan = self._plan
        phases: dict[str, float] = {}
        clock = time.perf_counter()

        kwargs: dict[str, Any] = {"allow_redirects": plan.follow_redirects}
        if spec.body is not None:
            kwargs["data"] = spec.body_bytes
        if spec.headers:
            kwargs["headers"] = spec.headers
        if plan.proxy is not None and not plan.uses_socks_proxy:
            kwargs["proxy"] = plan.proxy

        capture = spec.expectations.needs_body

        try:
            async with self._client.request(
                spec.method, spec.url, trace_request_ctx=phases, **kwargs
            ) as response:
                ttfb = time.perf_counter() - clock
                received, body = await self._drain(
                    response.content.iter_chunked(DRAIN_CHUNK), capture
                )
                elapsed = time.perf_counter() - clock
                return self._finish(spec, response.status, elapsed, received, ttfb, phases, body)
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, OSError, TimeoutError, UnicodeError, ValueError) as exc:
            return _failed_attempt(spec, time.perf_counter() - clock, classify_failure(exc), phases)

    # --------------------------------------------------------------- httpx path

    async def _run_with_httpx(self, on_progress: ProgressHook | None) -> Report:
        import httpx

        plan = self._plan
        limits = httpx.Limits(max_connections=plan.workers, max_keepalive_connections=plan.workers)
        try:
            async with httpx.AsyncClient(
                http2=True,
                verify=plan.verify_tls,
                timeout=plan.timeout,
                follow_redirects=plan.follow_redirects,
                headers=self._session_headers(),
                limits=limits,
            ) as client:
                self._backend = "httpx"
                self._client = client
                return await self._drive(on_progress)
        except httpx.HTTPError as exc:
            raise TestBusterError(
                f"cannot start the run: {classify_httpx_failure(exc)}", ExitCode.RUN_FAILED
            ) from exc

    async def _send_httpx(self, spec: RequestSpec) -> Attempt:
        """Send one request over httpx. DNS and connect timing are not split."""
        import httpx

        clock = time.perf_counter()
        capture = spec.expectations.needs_body
        try:
            # build_request stays inside the try: httpx.InvalidURL and a bad
            # header value raise here, and one bad request must not tear down
            # the worker pool.
            request = self._client.build_request(
                spec.method, spec.url, content=spec.body_bytes, headers=spec.headers or None
            )
            response = await self._client.send(request, stream=True)
            ttfb = time.perf_counter() - clock
            try:
                received, body = await self._drain(response.aiter_bytes(DRAIN_CHUNK), capture)
            finally:
                await response.aclose()
            elapsed = time.perf_counter() - clock
            return self._finish(spec, response.status_code, elapsed, received, ttfb, {}, body)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, httpx.InvalidURL, OSError, TimeoutError, ValueError) as exc:
            return _failed_attempt(
                spec, time.perf_counter() - clock, classify_httpx_failure(exc), {}
            )

    # ------------------------------------------------------------ shared engine

    @staticmethod
    async def _drain(chunks: Any, capture: bool) -> tuple[int, bytes | None]:
        """Count the response body, keeping a bounded copy only when needed."""
        received = 0
        buffer = bytearray() if capture else None
        async for chunk in chunks:
            received += len(chunk)
            if buffer is not None and len(buffer) < BODY_CAPTURE_LIMIT:
                buffer.extend(chunk[: BODY_CAPTURE_LIMIT - len(buffer)])
        return received, (bytes(buffer) if buffer is not None else None)

    def _finish(
        self,
        spec: RequestSpec,
        status: int,
        elapsed: float,
        received: int,
        ttfb: float,
        phases: dict[str, float],
        body: bytes | None,
    ) -> Attempt:
        """Turn a completed response into an Attempt, running any check."""
        verror = spec.expectations.evaluate(status, body, elapsed)
        return Attempt(
            status=status,
            elapsed=elapsed,
            bytes_in=received,
            failure=None,
            ttfb=ttfb,
            dns=phases.get("dns"),
            connect=phases.get("connect"),
            retries=0,
            label=spec.label,
            validation_error=verror,
        )

    async def _single_shot(self, spec: RequestSpec) -> Attempt:
        """Send one request through the active backend."""
        if self._backend == "httpx":
            return await self._send_httpx(spec)
        return await self._send_aiohttp(spec)

    def _session_headers(self) -> dict[str, str]:
        headers = dict(self._plan.headers)
        if not any(name.lower() == "user-agent" for name in headers):
            headers["User-Agent"] = f"TestBuster/{__version__}"
        if self._plan.payload is not None and not any(
            name.lower() == "content-type" for name in headers
        ):
            headers["Content-Type"] = "application/json"
        return headers

    # -------------------------------------------------------------- run phases

    async def _drive(self, on_progress: ProgressHook | None) -> Report:
        """Run the warmup, then the measured run, on the chosen backend."""
        with self._stop_on_signals():
            if self._plan.warmup:
                await self._run_warmup()
            return await self._run_main(on_progress)

    async def _run_warmup(self) -> None:
        """Fire throwaway requests to fill the connection pool and any caches.

        Warmup results never reach the report, because this path sends through
        _single_shot and never records into the tally. Their only job is to keep
        the first real measurements from carrying handshake cost.
        """
        remaining = self._plan.warmup
        spec = self._source.spec_for(0)

        async def warm() -> None:
            nonlocal remaining
            while remaining > 0 and not self._stop.is_set():
                remaining -= 1
                await self._single_shot(spec)

        async with asyncio.TaskGroup() as group:
            for _ in range(min(self._plan.workers, self._plan.warmup)):
                group.create_task(warm())

    async def _run_main(self, on_progress: ProgressHook | None) -> Report:
        plan = self._plan
        self._started_at = time.perf_counter()
        self._deadline = time.monotonic() + plan.duration if plan.duration else None

        helpers = [asyncio.create_task(self._watchdog(), name="watchdog")]
        if on_progress is not None:
            helpers.append(asyncio.create_task(self._pump(on_progress), name="progress"))
        if self._profile is not None and self._limiter is not None:
            helpers.append(asyncio.create_task(self._drive_profile(), name="profile"))

        try:
            async with asyncio.TaskGroup() as group:
                for index in range(plan.workers):
                    group.create_task(self._worker(), name=f"worker-{index}")
        except asyncio.CancelledError:
            self._stop_reason = "interrupted"
            raise
        finally:
            for task in helpers:
                task.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)

        wall = time.perf_counter() - self._started_at

        if on_progress is not None:
            on_progress(self._snapshot())

        return self._tally.summarize(
            plan, wall, interrupted=self.interrupted, stop_reason=self._stop_reason
        )

    async def _worker(self) -> None:
        """Claim and send requests until a stop condition fires."""
        while True:
            index = self._claim()
            if index is None:
                return

            if self._limiter is not None:
                took = await self._limiter.take(self._should_stop)
                # The limiter may have held this worker past the deadline, or
                # bailed because a stop fired while it waited.
                if not took or self._should_stop():
                    return

            spec = self._source.spec_for(index)
            attempt = await self._send_with_retries(spec)

            second = int(time.perf_counter() - self._started_at)
            self._tally.record(attempt, second=second)

            if self._on_attempt is not None:
                self._on_attempt(attempt)

            if self._plan.max_errors is not None and self._tally.failed >= self._plan.max_errors:
                self._stop_reason = "max errors reached"
                self._stop.set()
                return

    def _claim(self) -> int | None:
        """Reserve one request slot and return its index, or None to stop.

        This runs start to finish with no await, so the single-threaded event
        loop cannot interleave two workers inside it. That is what makes the
        bare counter safe and lets the engine skip a lock.
        """
        if self._should_stop():
            return None
        if self._quota is not None and self._claimed >= self._quota:
            return None
        index = self._claimed
        self._claimed += 1
        return index

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        return self._deadline is not None and time.monotonic() >= self._deadline

    # ------------------------------------------------------------------ request

    async def _send_with_retries(self, spec: RequestSpec) -> Attempt:
        """Send one request, retrying transport errors and retriable statuses.

        The returned Attempt describes the final try. retries counts how many
        earlier tries it took, so the report can separate a slow-but-fine
        endpoint from one that only answers on the second ask.
        """
        plan = self._plan
        used = 0

        while True:
            attempt = await self._single_shot(spec)
            worth_retrying = attempt.failure is not None or attempt.status in plan.retry_statuses

            if not worth_retrying or used >= plan.retries or self._should_stop():
                if used == 0:
                    return attempt
                return replace(attempt, retries=used)

            used += 1
            await asyncio.sleep(self._backoff(used))

    def _backoff(self, attempt_number: int) -> float:
        """Return the wait before retry number attempt_number.

        The delay doubles each time and carries full jitter, so a wave of
        workers that all failed together does not retry in lockstep.
        """
        window = min(BACKOFF_CEILING, self._plan.retry_backoff * (2 ** (attempt_number - 1)))
        return random.uniform(0.0, window)

    # ------------------------------------------------------------- housekeeping

    def _snapshot(self) -> Progress:
        return Progress(
            finished=self._tally.total,
            elapsed=time.perf_counter() - self._started_at,
            succeeded=self._tally.succeeded,
            failed=self._tally.failed,
        )

    async def _pump(self, on_progress: ProgressHook) -> None:
        """Report progress on a fixed timer.

        One task owns the console. Printing from every worker at once would
        let progress lines interleave and shred the output.
        """
        while True:
            on_progress(self._snapshot())
            await asyncio.sleep(PROGRESS_INTERVAL)

    async def _drive_profile(self) -> None:
        """Update the limiter rate from the load profile as time passes."""
        assert self._profile is not None and self._limiter is not None
        while True:
            elapsed = time.perf_counter() - self._started_at
            self._limiter.set_rate(self._profile.rate_at(elapsed))
            await asyncio.sleep(PROFILE_INTERVAL)

    async def _watchdog(self) -> None:
        """Wake the loop regularly.

        On Windows a signal handler only runs between bytecodes in the main
        thread. Without a task that wakes often, a Ctrl+C during a long
        network wait can sit unnoticed. This costs one short sleep per tick.
        """
        while True:  # noqa: ASYNC110 - the periodic wake is the whole purpose
            await asyncio.sleep(WATCHDOG_INTERVAL)

    def _note_interrupt(self) -> None:
        """Record a Ctrl+C and hand the next one back to Python.

        The first press sets the stop flag and restores the previous handlers,
        so a second press raises KeyboardInterrupt and ends the process even if
        the run is somehow wedged.
        """
        self._stop_reason = "interrupted"
        self._stop.set()
        self._restore_signals()

    def _restore_signals(self) -> None:
        """Undo every signal handler this run installed. Safe to call twice."""
        while self._signal_restores:
            restore = self._signal_restores.pop()
            with contextlib.suppress(NotImplementedError, RuntimeError, OSError, ValueError):
                restore()

    @contextlib.contextmanager
    def _stop_on_signals(self) -> Iterator[None]:
        """Turn the first Ctrl+C into a clean stop.

        Workers finish the request in flight, then exit, and the report still
        prints with whatever was measured. The first press also restores the
        previous handlers, so a second press is left to Python, which raises
        KeyboardInterrupt and ends the process.
        """
        loop = asyncio.get_running_loop()
        self._signal_restores = []

        def via_loop_c_handler() -> None:
            loop.call_soon_threadsafe(self._note_interrupt)

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self._note_interrupt)
                self._signal_restores.append(functools.partial(loop.remove_signal_handler, sig))
            except (NotImplementedError, RuntimeError):
                # Windows has no add_signal_handler. Fall back to the C handler.
                # getsignal reads the current handler, so it is restored exactly.
                try:
                    previous = signal.getsignal(sig)
                    signal.signal(sig, lambda *_: via_loop_c_handler())
                    restore_to = previous if previous is not None else signal.SIG_DFL
                    self._signal_restores.append(functools.partial(signal.signal, sig, restore_to))
                except (OSError, ValueError):
                    # Not the main thread. Ctrl+C handling is the caller's job.
                    continue

        try:
            yield
        finally:
            self._restore_signals()


def _default_spec(plan: RunPlan) -> RequestSpec:
    """Build the single request a plain run sends every time.

    The base headers live on the session, so this spec carries only the body
    and the plan's expectations.
    """
    return RequestSpec(
        method=plan.method,
        url=plan.target,
        headers={},
        body=plan.payload,
        label="default",
        expectations=plan.expectations,
    )


def _failed_attempt(
    spec: RequestSpec, elapsed: float, reason: str, phases: dict[str, float]
) -> Attempt:
    """Build the Attempt for a request that never got a response."""
    return Attempt(
        status=NO_RESPONSE,
        elapsed=elapsed,
        bytes_in=0,
        failure=reason,
        ttfb=None,
        dns=phases.get("dns"),
        connect=phases.get("connect"),
        retries=0,
        label=spec.label,
    )


def _loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """Return uvloop's loop factory when uvloop is installed, else None.

    uvloop raises the ceiling on requests per second by a wide margin. It does
    not build on Windows, and the plain asyncio loop works fine there, so a
    missing uvloop is never an error. None tells asyncio.Runner to use the
    default loop.
    """
    try:
        import uvloop
    except ImportError:
        return None
    return uvloop.new_event_loop  # type: ignore[no-any-return]


def loop_backend() -> str:
    """Name the event loop the next run will use. The header prints this."""
    return "asyncio" if _loop_factory() is None else "uvloop"


def execute(
    plan: RunPlan,
    on_progress: ProgressHook | None = None,
    *,
    source: RequestSource | None = None,
    on_attempt: AttemptHook | None = None,
) -> Report:
    """Run a plan from synchronous code. This is what the CLI calls."""
    engine = LoadEngine(plan, source=source)
    try:
        with asyncio.Runner(loop_factory=_loop_factory()) as runner:
            return runner.run(engine.run(on_progress, on_attempt=on_attempt))
    except KeyboardInterrupt as exc:
        # The graceful path sets an event and still reports. Reaching here means
        # the signal handler could not be installed, so no numbers survived.
        raise TestBusterError(
            f"{APP_NAME} stopped before it could report", ExitCode.INTERRUPTED
        ) from exc
