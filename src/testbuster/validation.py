"""Response validation.

A status of 200 does not mean the response was correct. The body can carry an
error, or the wrong shape, or arrive too late. Expectations let a run assert on
those, so a wrong-but-200 response counts as a failure instead of a success.

Every check is optional. An Expectations with nothing set passes everything, so
a plain run behaves as before.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from testbuster.errors import TestBusterError


def _parse_status_set(specs: list[str]) -> frozenset[int]:
    """Read status codes and ranges: 200, 201, 2xx, 200-204."""
    codes: set[int] = set()
    for spec in specs:
        token = spec.strip().lower()
        if token.endswith("xx") and len(token) == 3 and token[0].isdigit():
            base = int(token[0]) * 100
            codes.update(range(base, base + 100))
        elif "-" in token:
            low_text, _, high_text = token.partition("-")
            try:
                low, high = int(low_text), int(high_text)
            except ValueError as exc:
                raise TestBusterError(f"cannot read status range {spec!r}") from exc
            if low > high:
                raise TestBusterError(f"status range {spec!r} runs backwards")
            codes.update(range(low, high + 1))
        else:
            try:
                codes.add(int(token))
            except ValueError as exc:
                raise TestBusterError(f"cannot read status code {spec!r}") from exc
    return frozenset(codes)


def _dig(data: Any, path: str) -> Any:
    """Follow a dotted path through nested dicts and lists.

    "data.items.0.name" reads key data, key items, index 0, then key name. A
    missing key or index raises KeyError, which the caller turns into a failure.
    """
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


@dataclass(frozen=True, slots=True)
class JsonCheck:
    """One JSON-path equality check."""

    path: str
    expected: str


@dataclass(frozen=True, slots=True)
class Expectations:
    """The checks a response must pass to count as a success."""

    status: frozenset[int] | None = None
    body_regex: str | None = None
    json_checks: tuple[JsonCheck, ...] = ()
    max_latency_s: float | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.status is None
            and self.body_regex is None
            and not self.json_checks
            and self.max_latency_s is None
        )

    @property
    def needs_body(self) -> bool:
        """True when a check reads the response body."""
        return self.body_regex is not None or bool(self.json_checks)

    def evaluate(self, status: int, body: bytes | None, elapsed_s: float) -> str | None:
        """Return None when every check passes, else a short failure reason."""
        if self.status is not None and status not in self.status:
            return f"status {status} not expected"

        if self.max_latency_s is not None and elapsed_s > self.max_latency_s:
            return f"latency {elapsed_s * 1000:.0f}ms over budget"

        if self.needs_body:
            text = "" if body is None else body.decode("utf-8", errors="replace")

            if self.body_regex is not None and re.search(self.body_regex, text) is None:
                return "body did not match the pattern"

            if self.json_checks:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return "response body is not JSON"
                for check in self.json_checks:
                    try:
                        actual = _dig(parsed, check.path)
                    except (KeyError, IndexError, ValueError):
                        return f"json path {check.path} is missing"
                    if str(actual) != check.expected:
                        return f"json {check.path} was {actual!r}, expected {check.expected!r}"

        return None


def build_expectations(
    *,
    status: list[str] | None,
    body_regex: str | None,
    json_specs: list[str] | None,
    max_latency_s: float | None,
) -> Expectations:
    """Turn raw CLI values into an Expectations.

    A json spec is "path=value", for example "data.id=42".
    """
    status_set = _parse_status_set(status) if status else None

    if body_regex is not None:
        try:
            re.compile(body_regex)
        except re.error as exc:
            raise TestBusterError(f"--expect-regex is not a valid pattern: {exc}") from exc

    checks: list[JsonCheck] = []
    for spec in json_specs or []:
        path, sep, value = spec.partition("=")
        if not sep or not path.strip():
            raise TestBusterError(f"cannot read --expect-json {spec!r}. Use path=value")
        checks.append(JsonCheck(path.strip(), value))

    return Expectations(
        status=status_set,
        body_regex=body_regex,
        json_checks=tuple(checks),
        max_latency_s=max_latency_s,
    )


def to_dict(expectations: Expectations) -> dict[str, Any]:
    """Serialize expectations for the JSON report."""
    return {
        "status": sorted(expectations.status) if expectations.status else None,
        "body_regex": expectations.body_regex,
        "json_checks": [{"path": c.path, "expected": c.expected} for c in expectations.json_checks],
        "max_latency_ms": (
            None if expectations.max_latency_s is None else expectations.max_latency_s * 1000
        ),
    }


# A shared empty instance, so callers avoid building one per request.
NO_EXPECTATIONS: Expectations = Expectations()
