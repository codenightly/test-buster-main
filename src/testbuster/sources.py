"""What a request should send, and where the engine gets the next one.

The engine asks a source for a spec before each send. Three flavors exist: one
fixed request, one request per row of a CSV, and a weighted set of scenario
steps. All three reduce to the same shape, a list of specs plus a wheel of
positions to walk, so RequestCycle implements all of them.

Specs are built once, up front. A CSV of a hundred rows fills its placeholders a
hundred times at load, not once per request, so a million-request run does no
repeated string work.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import urljoin

from testbuster.config import normalize_target
from testbuster.errors import TestBusterError
from testbuster.validation import NO_EXPECTATIONS, Expectations, build_expectations

_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """One request the engine can send.

    `label` groups the request in a per-label breakdown. A scenario names each
    step, so its label is the step name. A plain run leaves it as the default.
    """

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    label: str = "default"
    expectations: Expectations = NO_EXPECTATIONS
    # Encoded here, once, because the engine reads it on every send.
    body_bytes: bytes | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        encoded = None if self.body is None else self.body.encode("utf-8")
        # A frozen dataclass blocks plain assignment, so this call sets the field.
        object.__setattr__(self, "body_bytes", encoded)


class RequestSource(Protocol):
    """Something that hands the engine the next request to send.

    RequestCycle covers every source that ships today. This protocol is the seam
    a future source implements, for example one that reads from a queue.
    """

    @property
    def labels(self) -> tuple[str, ...]:
        """The distinct labels this source produces, for the per-label tallies."""
        ...

    def spec_for(self, index: int) -> RequestSpec:
        """Return the request for the claimed request number `index`."""
        ...


class RequestCycle:
    """A fixed list of specs walked through a repeating wheel of positions.

    The wheel is what makes the weighting exact and the run repeatable: no
    random draw, so two runs over the same input send the same sequence.
    """

    __slots__ = ("_labels", "_specs", "_wheel")

    def __init__(self, specs: list[RequestSpec], wheel: list[int]) -> None:
        if not specs or not wheel:
            raise TestBusterError("a request source needs at least one request")
        self._specs = specs
        self._wheel = wheel

        # Distinct labels, in first-seen order. Deduplication matters: a CSV of
        # a hundred rows shares one label, and reporting a hundred copies would
        # switch on the per-label breakdown for a run that has only one group.
        seen: dict[str, None] = {}
        for spec in specs:
            seen.setdefault(spec.label, None)
        self._labels = tuple(seen)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def spec_for(self, index: int) -> RequestSpec:
        return self._specs[self._wheel[index % len(self._wheel)]]


def single(spec: RequestSpec) -> RequestCycle:
    """Return a source that sends the same request every time."""
    return RequestCycle([spec], [0])


# --------------------------------------------------------------------- CSV rows


def _fill(template: str, row: dict[str, str]) -> str:
    """Replace every {{column}} in template with the row value."""

    def swap(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in row:
            raise TestBusterError(
                f"data file has no column {name!r} for placeholder {match.group(0)}"
            )
        return row[name]

    return _PLACEHOLDER.sub(swap, template)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV with a header row into a list of dicts."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TestBusterError(f"cannot read data file {path}: {exc.strerror}") from exc

    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise TestBusterError(f"data file {path} is empty. It needs a header row")

    rows: list[dict[str, str]] = []
    for row in reader:
        # A row with too few values leaves the last columns empty. That would
        # blank a placeholder and send the request to the wrong URL, so name the
        # line and stop.
        empty = [str(name) for name, value in row.items() if value is None]
        if empty:
            names = ", ".join(empty)
            raise TestBusterError(
                f"data file {path} line {reader.line_num} is short. It has no value for {names}"
            )
        rows.append(dict(row))

    if not rows:
        raise TestBusterError(f"data file {path} has a header but no data rows")
    return rows


def from_rows(
    rows: list[dict[str, str]],
    *,
    method: str,
    url_template: str,
    header_templates: dict[str, str],
    body_template: str | None,
    expectations: Expectations,
) -> RequestCycle:
    """Build one spec per CSV row, filling placeholders once at load."""
    specs = [
        RequestSpec(
            method=method,
            url=_fill(url_template, row),
            headers={name: _fill(value, row) for name, value in header_templates.items()},
            body=None if body_template is None else _fill(body_template, row),
            expectations=expectations,
        )
        for row in rows
    ]
    return RequestCycle(specs, list(range(len(specs))))


# --------------------------------------------------------------------- scenarios


@dataclass(frozen=True, slots=True)
class Step:
    """One named request in a scenario, with its weight."""

    spec: RequestSpec
    weight: int


def from_steps(steps: list[Step]) -> RequestCycle:
    """Expand weighted steps into a cycle whose mix matches the weights.

    The weights are reduced by their greatest common divisor first, so 2:2
    builds a wheel of two rather than four.
    """
    if not steps:
        raise TestBusterError("a scenario needs at least one step")

    for index, step in enumerate(steps):
        # A weight below 1 wins no place on the wheel, but the step still adds a
        # label. The per-label table would then hold a row stuck at zero.
        if step.weight < 1:
            raise TestBusterError(f"step {index + 1} weight must be at least 1")

    weights = [step.weight for step in steps]
    divisor = gcd(*weights) or 1

    wheel: list[int] = []
    for position, weight in enumerate(weights):
        wheel.extend([position] * (weight // divisor))

    return RequestCycle([step.spec for step in steps], wheel)


def _step_int(value: Any, index: int, name: str) -> int:
    """Read a whole number from a step, so a bad value fails as a clean error."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TestBusterError(f"step {index + 1} {name} must be a whole number") from exc


def _step_float(value: Any, index: int, name: str) -> float:
    """Read a number from a step, so a bad value fails as a clean error."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TestBusterError(f"step {index + 1} {name} must be a number") from exc


def _status_list(raw: Any) -> list[str] | None:
    """Read an expected status into a list of codes.

    A step writes one code, a list of codes, or nothing at all.
    """
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)] if isinstance(raw, int | str) else None


def _step_expectations(
    raw: dict[str, Any], index: int, default_expectations: Expectations = NO_EXPECTATIONS
) -> Expectations:
    """Read a step's expect block, or fall back to the checks the run carries.

    A step that states nothing of its own inherits the --expect-* flags, so one
    flag covers every quiet step.
    """
    expect = raw.get("expect")
    if expect is None:
        return default_expectations
    if not isinstance(expect, dict):
        raise TestBusterError(f"step {index + 1} 'expect' must be an object")

    json_specs = expect.get("json")
    json_list = [str(item) for item in json_specs] if isinstance(json_specs, list) else None

    latency_ms = expect.get("max_latency_ms")
    max_latency_s = (
        None if latency_ms is None else _step_float(latency_ms, index, "max_latency_ms") / 1000
    )

    return build_expectations(
        status=_status_list(expect.get("status")),
        body_regex=expect.get("regex"),
        json_specs=json_list,
        max_latency_s=max_latency_s,
    )


def _build_step(
    raw: dict[str, Any],
    base_url: Any,
    index: int,
    *,
    default_method: str | None = None,
    default_body: str | None = None,
    default_expectations: Expectations = NO_EXPECTATIONS,
) -> Step:
    """Turn one raw step dict into a Step.

    The defaults hold what the command line asked for. Anything the step states
    itself wins over them.
    """
    if "url" not in raw:
        raise TestBusterError(f"step {index + 1} has no 'url'")

    url = str(raw["url"])
    if base_url is not None:
        if not isinstance(base_url, str):
            raise TestBusterError(f"step {index + 1} needs 'base_url' to be a string")
        if "://" not in url:
            url = urljoin(base_url, url)
    url = normalize_target(url)

    headers = raw.get("headers", {})
    if not isinstance(headers, dict):
        raise TestBusterError(f"step {index + 1} headers must be an object")
    step_headers = {str(name): str(value) for name, value in headers.items()}

    raw_body = raw.get("body")
    body_is_json = isinstance(raw_body, dict | list)
    body: str | None
    if body_is_json:
        body = json.dumps(raw_body)
    elif raw_body is not None:
        body = str(raw_body)
    else:
        body = default_body

    # A step that wrote its body as an object or a list meant JSON, and a server
    # can reject a JSON body that arrives without the type. A step that set the
    # header keeps it, whatever the spelling.
    if body_is_json and not any(name.lower() == "content-type" for name in step_headers):
        step_headers["Content-Type"] = "application/json"

    return Step(
        spec=RequestSpec(
            method=str(raw.get("method") or default_method or "GET").upper(),
            url=url,
            headers=step_headers,
            body=body,
            label=str(raw.get("name", f"step{index + 1}")),
            expectations=_step_expectations(raw, index, default_expectations),
        ),
        weight=_step_int(raw.get("weight", 1), index, "weight"),
    )


def _read_scenario_file(path: Path) -> dict[str, Any]:
    """Read a scenario file as JSON, or YAML when the extra is installed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TestBusterError(f"cannot read scenario {path}: {exc.strerror}") from exc

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise TestBusterError(
                "a YAML scenario needs the yaml extra. Install it with: "
                "pip install 'test-buster[yaml]'. A JSON scenario needs no extra."
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TestBusterError(f"{path} is not valid YAML: {exc}") from exc
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TestBusterError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(loaded, list):
        loaded = {"steps": loaded}
    if not isinstance(loaded, dict) or "steps" not in loaded:
        raise TestBusterError(f"{path} needs a 'steps' list, or be a list of steps")
    return loaded


def load_scenario(
    path: Path,
    *,
    default_method: str | None = None,
    default_body: str | None = None,
    default_expectations: Expectations = NO_EXPECTATIONS,
) -> RequestCycle:
    """Read a scenario file into a request source.

    The three defaults carry the command line into every step. A step that
    states its own method, body, or expect block wins over them.
    """
    document = _read_scenario_file(path)
    raw_steps = document["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise TestBusterError(f"{path} 'steps' must be a non-empty list")

    base_url = document.get("base_url")
    steps: list[Step] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise TestBusterError(f"step {index + 1} must be an object, not a bare value")
        step = _build_step(
            raw,
            base_url,
            index,
            default_method=default_method,
            default_body=default_body,
            default_expectations=default_expectations,
        )
        steps.append(step)
    return from_steps(steps)
