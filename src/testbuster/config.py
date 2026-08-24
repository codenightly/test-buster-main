"""The run plan: what to hit, how hard, and when to stop.

RunPlan validates itself on construction, so an invalid plan never reaches the
engine. Every parse helper here raises TestBusterError with a message aimed at
the person who typed the command.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

from testbuster.errors import ExitCode, TestBusterError
from testbuster.validation import NO_EXPECTATIONS, Expectations
from testbuster.validation import to_dict as _expectations_to_dict

#: Methods that a load test may send. Test Buster! refuses the rest so a typo
#: like "PSOT" fails before the first packet leaves the machine.
KNOWN_METHODS: Final[frozenset[str]] = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
)

#: Status codes that get another attempt when --retries is above zero.
DEFAULT_RETRY_STATUSES: Final[tuple[int, ...]] = (408, 425, 429, 500, 502, 503, 504)

#: Percentiles that every report carries. Sorted, so output order never moves.
REPORTED_PERCENTILES: Final[tuple[float, ...]] = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.IGNORECASE)
_UNIT_SECONDS: Final[dict[str, float]] = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_SOCKS_SCHEMES: Final[frozenset[str]] = frozenset({"socks4", "socks5", "socks5h"})
_HTTP_PROXY_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})


def parse_duration(text: str, *, label: str = "duration") -> float:
    """Turn "1h30m", "500ms", or "45" into seconds.

    A bare number counts as seconds. Units may repeat and add up, so "1m30s"
    gives 90.0. Raise TestBusterError when nothing parses.
    """
    raw = text.strip().lower()
    if not raw:
        raise TestBusterError(f"{label} is empty")

    # Drop inner spaces up front, so "1m 30s" and "500 ms" parse the same as the
    # compact forms and the consumed-length check below stays honest.
    compact = raw.replace(" ", "")

    try:
        # A bare number is the common case. Accept it before the unit parser.
        return _positive(float(compact), label)
    except ValueError:
        pass

    total = 0.0
    consumed = 0
    for match in _DURATION_PART.finditer(compact):
        total += float(match.group(1)) * _UNIT_SECONDS[match.group(2)]
        consumed += len(match.group(0))

    if consumed != len(compact):
        raise TestBusterError(
            f"cannot read {label} {text!r}. Use a number of seconds, "
            "or units: 500ms, 30s, 5m, 1h30m"
        )
    return _positive(total, label)


def _positive(value: float, label: str) -> float:
    """Check one parsed number.

    Test the finite case first. float() reads "nan", "inf", and "infinity".
    Every comparison against nan is False, so a test for "above zero" alone
    lets a non-finite value reach the engine.
    """
    if not math.isfinite(value):
        raise TestBusterError(f"{label} must be a finite number, got {value}")
    if value <= 0:
        raise TestBusterError(f"{label} must be above zero, got {value}")
    return value


def parse_header_pairs(pairs: list[str]) -> dict[str, str]:
    """Read curl-style "Name: value" strings into a dict.

    A later value replaces an earlier one with the same name.
    """
    parsed: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition(":")
        if not separator or not name.strip():
            raise TestBusterError(f"cannot read header {pair!r}. Expected 'Name: value'")
        parsed[name.strip()] = value.strip()
    return parsed


def parse_header_json(blob: str) -> dict[str, str]:
    """Read a JSON object of headers.

    This keeps the -H '{"A": "b"}' form that earlier releases accepted.
    """
    try:
        decoded = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise TestBusterError(
            f"--headers is not valid JSON: {exc.msg} at position {exc.pos}"
        ) from exc
    if not isinstance(decoded, dict):
        raise TestBusterError('--headers must be a JSON object, for example \'{"Accept": "*/*"}\'')
    return {str(name): str(value) for name, value in decoded.items()}


def resolve_payload(raw: str | None) -> str | None:
    """Return the request body.

    A value that starts with "@" names a file to read, the same way curl does.
    Use "@-" for nothing, and a literal "@" needs no escape because only the
    first character counts.
    """
    if raw is None or raw == "":
        return None
    if raw == "@-":
        # The documented sentinel for an empty body.
        return None
    if not raw.startswith("@"):
        return raw

    source = Path(raw[1:]).expanduser()
    try:
        return source.read_text(encoding="utf-8")
    except OSError as exc:
        raise TestBusterError(f"cannot read body file {source}: {exc.strerror}") from exc
    except UnicodeDecodeError as exc:
        raise TestBusterError(f"body file {source} is not UTF-8 text: {exc.reason}") from exc


def normalize_target(raw: str) -> str:
    """Check the target URL and add a scheme when the user left it out."""
    candidate = raw.strip()
    if not candidate:
        raise TestBusterError("the target URL is empty")

    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parts = urlparse(candidate)
    if parts.scheme not in {"http", "https"}:
        raise TestBusterError(
            f"unsupported URL scheme {parts.scheme!r}. Test Buster! speaks http and https"
        )
    if not parts.hostname:
        raise TestBusterError(f"the target URL {raw!r} has no host")
    return candidate


# --------------------------------------------------------------- profiles
# A flat rate never finds the point where a system tips over. A profile
# varies the target rate across the run, so one command can ramp traffic up,
# hold plateaus, or fire a spike. It answers one question: at t seconds in,
# what rate should the engine aim for?

#: Profile names the CLI accepts.
PROFILE_NAMES: Final[tuple[str, ...]] = ("constant", "ramp", "step", "spike")


@dataclass(frozen=True, slots=True)
class LoadProfile:
    """A rate schedule over the length of a run.

    `start_rate` and `peak_rate` are requests per second. `steps` only matters
    for the step profile. `spike_at` is the fraction of the run where the spike
    peaks, from 0 to 1.
    """

    kind: str
    duration: float
    start_rate: float
    peak_rate: float
    steps: int = 4
    spike_at: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in PROFILE_NAMES:
            allowed = ", ".join(PROFILE_NAMES)
            raise TestBusterError(f"unknown profile {self.kind!r}. Use one of: {allowed}")
        if self.duration <= 0:
            raise TestBusterError("a load profile needs --duration above zero")
        if self.start_rate < 0 or self.peak_rate <= 0:
            raise TestBusterError("profile rates must be above zero")
        if self.steps < 1:
            raise TestBusterError("--profile-steps must be at least 1")
        if not 0.0 <= self.spike_at <= 1.0:
            raise TestBusterError("--spike-at must be between 0 and 1")

    def rate_at(self, elapsed: float) -> float:
        """Return the target rate, in requests per second, at time `elapsed`."""
        fraction = 0.0 if self.duration <= 0 else min(1.0, max(0.0, elapsed / self.duration))

        if self.kind == "constant":
            return self.peak_rate

        if self.kind == "ramp":
            return self.start_rate + (self.peak_rate - self.start_rate) * fraction

        if self.kind == "step":
            # Climb from start to peak in `steps` even plateaus.
            level = min(self.steps - 1, int(fraction * self.steps))
            if self.steps == 1:
                return self.peak_rate
            share = level / (self.steps - 1)
            return self.start_rate + (self.peak_rate - self.start_rate) * share

        # spike: rise to the peak at spike_at, then fall back to the start rate.
        if fraction <= self.spike_at:
            climb = 0.0 if self.spike_at == 0 else fraction / self.spike_at
            return self.start_rate + (self.peak_rate - self.start_rate) * climb
        tail = 1.0 - self.spike_at
        fall = 1.0 if tail == 0 else (fraction - self.spike_at) / tail
        return self.peak_rate - (self.peak_rate - self.start_rate) * fall

    def describe(self) -> str:
        """Return a one-line summary for the run header."""
        if self.kind == "constant":
            return f"constant {self.peak_rate:g} req/s for {self.duration:g}s"
        if self.kind == "ramp":
            return f"ramp {self.start_rate:g} -> {self.peak_rate:g} req/s over {self.duration:g}s"
        if self.kind == "step":
            return (
                f"step {self.start_rate:g} -> {self.peak_rate:g} req/s "
                f"in {self.steps} levels over {self.duration:g}s"
            )
        return (
            f"spike {self.start_rate:g} -> {self.peak_rate:g} req/s "
            f"peaking at {self.spike_at:g} of {self.duration:g}s"
        )


@dataclass(frozen=True, slots=True)
class Gates:
    """Thresholds that decide the exit code.

    A gate stays off while its field is None. Latency gates use milliseconds.
    """

    max_p95_ms: float | None = None
    max_p99_ms: float | None = None
    min_success_rate: float | None = None

    @property
    def any_set(self) -> bool:
        return any(
            value is not None for value in (self.max_p95_ms, self.max_p99_ms, self.min_success_rate)
        )


@dataclass(frozen=True, slots=True)
class RunPlan:
    """One load test, fully described.

    Either total_requests or duration stops the run. Set both and whichever
    limit arrives first wins, which is how you cap an open-ended soak test.
    """

    target: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    payload: str | None = None

    workers: int = 10
    total_requests: int | None = 100
    duration: float | None = None
    warmup: int = 0
    rate_limit: float | None = None

    timeout: float = 30.0
    verify_tls: bool = True
    follow_redirects: bool = True
    keepalive: bool = True
    proxy: str | None = None

    retries: int = 0
    retry_backoff: float = 0.25
    retry_statuses: frozenset[int] = frozenset(DEFAULT_RETRY_STATUSES)

    max_errors: int | None = None
    cookies: bool = False
    http2: bool = False
    compact_memory: bool = False
    profile: LoadProfile | None = None
    expectations: Expectations = NO_EXPECTATIONS

    keep_attempts: bool = False
    gates: Gates = field(default_factory=Gates)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", normalize_target(self.target))
        object.__setattr__(self, "method", self.method.strip().upper())

        if self.method not in KNOWN_METHODS:
            allowed = ", ".join(sorted(KNOWN_METHODS))
            raise TestBusterError(f"unknown HTTP method {self.method!r}. Use one of: {allowed}")

        if self.workers < 1:
            raise TestBusterError(f"--concurrency must be at least 1, got {self.workers}")
        if self.total_requests is not None and self.total_requests < 1:
            raise TestBusterError(f"--requests must be at least 1, got {self.total_requests}")
        if self.duration is not None and self.duration <= 0:
            raise TestBusterError(f"--duration must be above zero, got {self.duration}")
        if self.total_requests is None and self.duration is None:
            raise TestBusterError(
                "set --requests, --duration, or both. Nothing tells the run to stop"
            )
        if self.warmup < 0:
            raise TestBusterError(f"--warmup cannot be negative, got {self.warmup}")
        if self.retries < 0:
            raise TestBusterError(f"--retries cannot be negative, got {self.retries}")
        if self.retry_backoff < 0:
            raise TestBusterError(f"--retry-backoff cannot be negative, got {self.retry_backoff}")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise TestBusterError(f"--rate must be above zero, got {self.rate_limit}")
        if self.timeout <= 0:
            raise TestBusterError(f"--timeout must be above zero, got {self.timeout}")

        if self.max_errors is not None and self.max_errors < 1:
            raise TestBusterError(f"--max-errors must be at least 1, got {self.max_errors}")

        if self.profile is not None:
            if self.duration is None:
                raise TestBusterError("a --profile needs --duration to set its length")
            if self.rate_limit is not None:
                raise TestBusterError("--profile drives the rate, so it cannot go with --rate")

        if self.proxy is not None:
            object.__setattr__(self, "proxy", self._checked_proxy(self.proxy))

        if self.http2 and self.proxy is not None:
            raise TestBusterError("--http2 and --proxy cannot combine yet")

        if self.gates.min_success_rate is not None and not 0 <= self.gates.min_success_rate <= 100:
            raise TestBusterError("--fail-under-success takes a percentage between 0 and 100")

    @staticmethod
    def _checked_proxy(raw: str) -> str:
        candidate = raw.strip()
        parts = urlparse(candidate)
        if not parts.scheme or not parts.hostname:
            raise TestBusterError(
                f"cannot read proxy {raw!r}. Expected http://host:port or socks5://host:port"
            )
        if parts.scheme in _SOCKS_SCHEMES:
            return candidate
        if parts.scheme in _HTTP_PROXY_SCHEMES:
            return candidate
        raise TestBusterError(
            f"unsupported proxy scheme {parts.scheme!r}. Use http, https, socks4, or socks5"
        )

    @property
    def uses_socks_proxy(self) -> bool:
        if self.proxy is None:
            return False
        return urlparse(self.proxy).scheme in _SOCKS_SCHEMES

    def advisories(self) -> list[str]:
        """Return notes worth showing before the run starts.

        These describe choices that work but rarely do what the user meant.
        """
        notes: list[str] = []
        if self.payload is not None and self.method in {"GET", "HEAD"}:
            notes.append(f"a body was given with {self.method}. Most servers ignore it")
        if not self.verify_tls:
            notes.append("TLS verification is off. Any certificate will pass")
        if self.rate_limit is not None and self.rate_limit < self.workers:
            notes.append(
                f"--rate {self.rate_limit:g} is below --concurrency {self.workers}, "
                "so most workers will idle"
            )
        return notes

    @property
    def stop_label(self) -> str:
        """Describe the stop condition for the console header."""
        limit = ""
        if self.total_requests is not None and self.duration is not None:
            limit = f"{self.total_requests} requests or {self.duration:g}s, whichever comes first"
        elif self.duration is not None:
            limit = f"{self.duration:g}s"
        else:
            limit = f"{self.total_requests} requests"
        if self.max_errors is not None:
            limit += f", or {self.max_errors} failures"
        return limit

    def redacted_headers(self) -> dict[str, str]:
        """Return the headers with credentials masked, safe to print or save."""
        sensitive = {"authorization", "proxy-authorization", "cookie", "x-api-key", "api-key"}
        return {
            name: ("<redacted>" if name.lower() in sensitive else value)
            for name, value in self.headers.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan for the JSON report. Secrets stay masked."""
        return {
            "target": self.target,
            "method": self.method,
            "headers": self.redacted_headers(),
            "has_payload": self.payload is not None,
            "payload_bytes": len(self.payload.encode("utf-8")) if self.payload else 0,
            "workers": self.workers,
            "total_requests": self.total_requests,
            "duration_s": self.duration,
            "warmup": self.warmup,
            "rate_limit_rps": self.rate_limit,
            "timeout_s": self.timeout,
            "verify_tls": self.verify_tls,
            "follow_redirects": self.follow_redirects,
            "keepalive": self.keepalive,
            "proxy": self.proxy,
            "retries": self.retries,
            "retry_backoff_s": self.retry_backoff,
            "retry_statuses": sorted(self.retry_statuses),
            "max_errors": self.max_errors,
            "cookies": self.cookies,
            "http2": self.http2,
            "compact_memory": self.compact_memory,
            "profile": None if self.profile is None else self.profile.describe(),
            "expectations": None
            if self.expectations.is_empty
            else _expectations_to_dict(self.expectations),
        }


def require_socks_support() -> None:
    """Fail early when a SOCKS proxy is asked for but the extra is missing."""
    try:
        import aiohttp_socks  # noqa: F401
    except ImportError as exc:
        raise TestBusterError(
            "a SOCKS proxy needs the socks extra. Install it with: "
            "pip install 'test-buster[socks]'",
            ExitCode.RUN_FAILED,
        ) from exc


def require_http2_support() -> None:
    """Fail early when --http2 is asked for but httpx is missing."""
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise TestBusterError(
            "--http2 needs the http2 extra. Install it with: pip install 'test-buster[http2]'",
            ExitCode.RUN_FAILED,
        ) from exc
