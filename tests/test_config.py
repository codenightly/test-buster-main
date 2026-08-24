"""Tests for plan building, parsing, validation, and load profiles."""

from __future__ import annotations

import pytest

from testbuster.config import (
    Gates,
    LoadProfile,
    RunPlan,
    normalize_target,
    parse_duration,
    parse_header_json,
    parse_header_pairs,
    resolve_payload,
)
from testbuster.errors import TestBusterError
from testbuster.validation import Expectations, build_expectations


class TestParseDuration:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("30", 30.0),
            ("30s", 30.0),
            ("500ms", 0.5),
            ("2m", 120.0),
            ("1h", 3600.0),
            ("1m30s", 90.0),
            ("1h30m", 5400.0),
            ("1.5s", 1.5),
            ("  10s  ", 10.0),
            ("30S", 30.0),
        ],
    )
    def test_accepts_known_forms(self, text: str, expected: float) -> None:
        assert parse_duration(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "abc", "30x", "s30", "10 seconds", "-5s"])
    def test_rejects_junk(self, text: str) -> None:
        with pytest.raises(TestBusterError):
            parse_duration(text)

    def test_rejects_zero(self) -> None:
        with pytest.raises(TestBusterError, match="above zero"):
            parse_duration("0s")

    def test_message_names_the_flag(self) -> None:
        with pytest.raises(TestBusterError, match="--timeout"):
            parse_duration("nope", label="--timeout")


class TestHeaders:
    def test_reads_colon_pairs(self) -> None:
        parsed = parse_header_pairs(["Accept: application/json", "X-Trace: abc123"])
        assert parsed == {"Accept": "application/json", "X-Trace": "abc123"}

    def test_keeps_colons_inside_the_value(self) -> None:
        parsed = parse_header_pairs(["Referer: https://example.com:8443/x"])
        assert parsed["Referer"] == "https://example.com:8443/x"

    def test_last_value_wins(self) -> None:
        assert parse_header_pairs(["A: 1", "A: 2"]) == {"A": "2"}

    @pytest.mark.parametrize("bad", ["nocolon", ": empty-name"])
    def test_rejects_malformed_pairs(self, bad: str) -> None:
        with pytest.raises(TestBusterError, match="header"):
            parse_header_pairs([bad])

    def test_reads_json_object(self) -> None:
        assert parse_header_json('{"Accept": "*/*"}') == {"Accept": "*/*"}

    def test_rejects_json_array(self) -> None:
        with pytest.raises(TestBusterError, match="JSON object"):
            parse_header_json('["Accept"]')

    def test_rejects_broken_json(self) -> None:
        with pytest.raises(TestBusterError, match="not valid JSON"):
            parse_header_json("{oops")


class TestPayload:
    def test_passes_a_literal_through(self) -> None:
        assert resolve_payload('{"a": 1}') == '{"a": 1}'

    def test_treats_empty_as_absent(self) -> None:
        assert resolve_payload("") is None
        assert resolve_payload(None) is None

    def test_reads_an_at_prefixed_file(self, tmp_path) -> None:
        source = tmp_path / "body.json"
        source.write_text('{"from": "disk"}', encoding="utf-8")
        assert resolve_payload(f"@{source}") == '{"from": "disk"}'

    def test_reports_a_missing_file(self, tmp_path) -> None:
        with pytest.raises(TestBusterError, match="cannot read body file"):
            resolve_payload(f"@{tmp_path / 'absent.json'}")


class TestNormalizeTarget:
    def test_adds_https_to_a_bare_host(self) -> None:
        assert normalize_target("example.com") == "https://example.com"

    def test_keeps_an_explicit_scheme(self) -> None:
        assert normalize_target("http://example.com/x") == "http://example.com/x"

    @pytest.mark.parametrize("bad", ["", "   ", "ftp://example.com", "https://"])
    def test_rejects_unusable_targets(self, bad: str) -> None:
        with pytest.raises(TestBusterError):
            normalize_target(bad)


class TestRunPlan:
    def test_defaults_to_one_hundred_requests(self) -> None:
        plan = RunPlan(target="example.com")
        assert plan.total_requests == 100
        assert plan.workers == 10
        assert plan.method == "GET"
        assert plan.verify_tls is True

    def test_uppercases_the_method(self) -> None:
        assert RunPlan(target="example.com", method="post").method == "POST"

    def test_rejects_an_unknown_method(self) -> None:
        with pytest.raises(TestBusterError, match="unknown HTTP method"):
            RunPlan(target="example.com", method="PSOT")

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"workers": 0}, "--concurrency"),
            ({"total_requests": 0}, "--requests"),
            ({"warmup": -1}, "--warmup"),
            ({"retries": -1}, "--retries"),
            ({"rate_limit": 0}, "--rate"),
            ({"timeout": 0}, "--timeout"),
        ],
    )
    def test_rejects_out_of_range_numbers(self, kwargs: dict[str, object], fragment: str) -> None:
        with pytest.raises(TestBusterError, match=fragment):
            RunPlan(target="example.com", **kwargs)  # type: ignore[arg-type]

    def test_needs_a_stop_condition(self) -> None:
        with pytest.raises(TestBusterError, match="Nothing tells the run to stop"):
            RunPlan(target="example.com", total_requests=None, duration=None)

    def test_duration_alone_is_a_valid_stop_condition(self) -> None:
        plan = RunPlan(target="example.com", total_requests=None, duration=5.0)
        assert plan.stop_label == "5s"

    def test_both_limits_report_a_race(self) -> None:
        plan = RunPlan(target="example.com", total_requests=50, duration=5.0)
        assert "whichever comes first" in plan.stop_label

    @pytest.mark.parametrize(
        "proxy", ["http://p:8080", "https://p:8443", "socks5://p:1080", "socks4://p:1080"]
    )
    def test_accepts_known_proxy_schemes(self, proxy: str) -> None:
        assert RunPlan(target="example.com", proxy=proxy).proxy == proxy

    def test_rejects_an_unknown_proxy_scheme(self) -> None:
        with pytest.raises(TestBusterError, match="unsupported proxy scheme"):
            RunPlan(target="example.com", proxy="gopher://p:70")

    def test_rejects_a_proxy_with_no_host(self) -> None:
        with pytest.raises(TestBusterError, match="cannot read proxy"):
            RunPlan(target="example.com", proxy="http://")

    @pytest.mark.parametrize(
        ("proxy", "expected"),
        [("socks5://p:1080", True), ("http://p:8080", False), (None, False)],
    )
    def test_detects_a_socks_proxy(self, proxy: str | None, expected: bool) -> None:
        assert RunPlan(target="example.com", proxy=proxy).uses_socks_proxy is expected

    def test_rejects_a_success_rate_outside_zero_to_one_hundred(self) -> None:
        with pytest.raises(TestBusterError, match="between 0 and 100"):
            RunPlan(target="example.com", gates=Gates(min_success_rate=120))

    def test_masks_credentials_in_headers(self) -> None:
        plan = RunPlan(
            target="example.com",
            headers={"Authorization": "Bearer secret", "Accept": "*/*"},
        )
        redacted = plan.redacted_headers()
        assert redacted["Authorization"] == "<redacted>"
        assert redacted["Accept"] == "*/*"

    def test_keeps_secrets_out_of_the_serialized_plan(self) -> None:
        plan = RunPlan(target="example.com", headers={"X-API-Key": "hunter2"})
        assert "hunter2" not in str(plan.to_dict())

    def test_reports_the_payload_size_not_the_payload(self) -> None:
        plan = RunPlan(target="example.com", method="POST", payload='{"a":1}')
        as_dict = plan.to_dict()
        assert as_dict["has_payload"] is True
        assert as_dict["payload_bytes"] == 7
        assert "a" not in str(as_dict.get("payload", ""))


class TestAdvisories:
    def test_warns_about_a_body_on_a_get(self) -> None:
        plan = RunPlan(target="example.com", payload="{}")
        assert any("ignore it" in note for note in plan.advisories())

    def test_warns_when_tls_checks_are_off(self) -> None:
        plan = RunPlan(target="example.com", verify_tls=False)
        assert any("TLS verification is off" in note for note in plan.advisories())

    def test_warns_when_the_rate_cap_starves_the_workers(self) -> None:
        plan = RunPlan(target="example.com", workers=50, rate_limit=5)
        assert any("idle" in note for note in plan.advisories())

    def test_stays_quiet_on_a_plain_plan(self) -> None:
        assert RunPlan(target="example.com").advisories() == []


class TestGates:
    def test_reports_nothing_set(self) -> None:
        assert Gates().any_set is False

    @pytest.mark.parametrize(
        "kwargs",
        [{"max_p95_ms": 100}, {"max_p99_ms": 100}, {"min_success_rate": 99}],
    )
    def test_detects_a_single_active_gate(self, kwargs: dict[str, float]) -> None:
        assert Gates(**kwargs).any_set is True  # type: ignore[arg-type]


class TestBuild:
    def test_empty_when_nothing_set(self) -> None:
        exp = build_expectations(status=None, body_regex=None, json_specs=None, max_latency_s=None)
        assert exp.is_empty
        assert not exp.needs_body

    def test_status_ranges_and_classes(self) -> None:
        exp = build_expectations(
            status=["200", "2xx", "301-303"], body_regex=None, json_specs=None, max_latency_s=None
        )
        assert exp.status is not None
        assert 200 in exp.status
        assert 250 in exp.status  # from 2xx
        assert 302 in exp.status  # from the range

    def test_rejects_a_bad_regex(self) -> None:
        with pytest.raises(TestBusterError, match="not a valid pattern"):
            build_expectations(status=None, body_regex="(", json_specs=None, max_latency_s=None)

    def test_rejects_a_bad_json_spec(self) -> None:
        with pytest.raises(TestBusterError, match="path=value"):
            build_expectations(
                status=None, body_regex=None, json_specs=["nope"], max_latency_s=None
            )

    def test_needs_body_only_for_body_checks(self) -> None:
        status_only = build_expectations(
            status=["200"], body_regex=None, json_specs=None, max_latency_s=None
        )
        assert not status_only.needs_body

        with_regex = build_expectations(
            status=None, body_regex="ok", json_specs=None, max_latency_s=None
        )
        assert with_regex.needs_body


class TestEvaluate:
    def test_status_pass_and_fail(self) -> None:
        exp = Expectations(status=frozenset({200, 201}))
        assert exp.evaluate(200, None, 0.1) is None
        assert exp.evaluate(404, None, 0.1) is not None

    def test_latency_budget(self) -> None:
        exp = Expectations(max_latency_s=0.5)
        assert exp.evaluate(200, None, 0.4) is None
        assert "over budget" in (exp.evaluate(200, None, 0.6) or "")

    def test_regex_over_the_body(self) -> None:
        exp = build_expectations(
            status=None, body_regex="hello", json_specs=None, max_latency_s=None
        )
        assert exp.evaluate(200, b"well hello there", 0.1) is None
        assert exp.evaluate(200, b"goodbye", 0.1) is not None

    def test_json_path_equality(self) -> None:
        exp = build_expectations(
            status=None, body_regex=None, json_specs=["data.id=42"], max_latency_s=None
        )
        assert exp.evaluate(200, b'{"data": {"id": 42}}', 0.1) is None
        assert exp.evaluate(200, b'{"data": {"id": 7}}', 0.1) is not None

    def test_json_path_into_a_list(self) -> None:
        exp = build_expectations(
            status=None, body_regex=None, json_specs=["items.0.name=ada"], max_latency_s=None
        )
        assert exp.evaluate(200, b'{"items": [{"name": "ada"}]}', 0.1) is None

    def test_missing_json_path_fails(self) -> None:
        exp = build_expectations(
            status=None, body_regex=None, json_specs=["a.b=1"], max_latency_s=None
        )
        assert "missing" in (exp.evaluate(200, b'{"a": {}}', 0.1) or "")

    def test_non_json_body_fails_a_json_check(self) -> None:
        exp = build_expectations(
            status=None, body_regex=None, json_specs=["a=1"], max_latency_s=None
        )
        assert "not JSON" in (exp.evaluate(200, b"plain text", 0.1) or "")

    def test_checks_run_in_a_fixed_order(self) -> None:
        # Status is checked before the body, so a bad status reports first.
        exp = build_expectations(
            status=["200"], body_regex="x", json_specs=None, max_latency_s=None
        )
        assert "status" in (exp.evaluate(500, b"x", 0.1) or "")


class TestValidation:
    def test_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(TestBusterError, match="unknown profile"):
            LoadProfile("wobble", 10, 1, 10)

    def test_rejects_a_nonpositive_duration(self) -> None:
        with pytest.raises(TestBusterError, match="duration"):
            LoadProfile("ramp", 0, 1, 10)

    def test_rejects_a_bad_spike_point(self) -> None:
        with pytest.raises(TestBusterError, match="spike-at"):
            LoadProfile("spike", 10, 1, 10, spike_at=2.0)

    def test_rejects_too_few_steps(self) -> None:
        with pytest.raises(TestBusterError, match="profile-steps"):
            LoadProfile("step", 10, 1, 10, steps=0)


class TestConstant:
    def test_holds_the_peak(self) -> None:
        profile = LoadProfile("constant", 10, 1, 50)
        assert profile.rate_at(0) == 50
        assert profile.rate_at(5) == 50
        assert profile.rate_at(10) == 50


class TestRamp:
    def test_climbs_linearly(self) -> None:
        profile = LoadProfile("ramp", 10, 0, 100)
        assert profile.rate_at(0) == pytest.approx(0)
        assert profile.rate_at(5) == pytest.approx(50)
        assert profile.rate_at(10) == pytest.approx(100)

    def test_clamps_past_the_end(self) -> None:
        profile = LoadProfile("ramp", 10, 0, 100)
        assert profile.rate_at(20) == pytest.approx(100)


class TestStep:
    def test_climbs_in_levels(self) -> None:
        profile = LoadProfile("step", 10, 0, 90, steps=4)
        # Four levels across 0..90 land on 0, 30, 60, 90.
        assert profile.rate_at(0.0) == pytest.approx(0)
        assert profile.rate_at(9.9) == pytest.approx(90)

    def test_a_single_step_is_the_peak(self) -> None:
        profile = LoadProfile("step", 10, 0, 90, steps=1)
        assert profile.rate_at(5) == pytest.approx(90)


class TestSpike:
    def test_rises_then_falls(self) -> None:
        profile = LoadProfile("spike", 10, 10, 100, spike_at=0.5)
        assert profile.rate_at(0) == pytest.approx(10)
        assert profile.rate_at(5) == pytest.approx(100)
        assert profile.rate_at(10) == pytest.approx(10)

    def test_peak_sits_at_the_spike_point(self) -> None:
        profile = LoadProfile("spike", 10, 0, 100, spike_at=0.2)
        assert profile.rate_at(2.0) == pytest.approx(100)


class TestDescribe:
    @pytest.mark.parametrize("kind", ["constant", "ramp", "step", "spike"])
    def test_describes_every_kind(self, kind: str) -> None:
        text = LoadProfile(kind, 10, 1, 100).describe()
        assert kind in text


class TestParseDurationWhitespace:
    """Finding 6: a space between the number and the unit must parse."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("500 ms", 0.5), ("1m 30s", 90.0), ("2 s", 2.0), ("1h 30m", 5400.0)],
    )
    def test_accepts_inner_spaces(self, text: str, expected: float) -> None:
        assert parse_duration(text) == pytest.approx(expected)

    def test_still_rejects_words(self) -> None:
        with pytest.raises(TestBusterError):
            parse_duration("10 seconds")


class TestDurationValidated:
    """Finding 7: a nonpositive duration must be rejected."""

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_duration(self, bad: float) -> None:
        with pytest.raises(TestBusterError, match="--duration"):
            RunPlan(target="example.com", total_requests=None, duration=bad)


class TestEmptyBodySentinel:
    """Finding 8: '@-' means an empty body, not a file named '-'."""

    def test_at_dash_is_empty(self) -> None:
        assert resolve_payload("@-") is None


class TestNonFiniteDuration:
    """A non-finite number must not pass as a limit.

    float() reads "nan", "inf", and "infinity". A nan compares False against
    zero, so a test for "above zero" alone lets a non-finite value through.
    """

    @pytest.mark.parametrize("text", ["nan", "inf", "infinity", "-inf", "+inf", "NaN"])
    def test_rejects_non_finite_numbers(self, text: str) -> None:
        with pytest.raises(TestBusterError, match="finite"):
            parse_duration(text)

    def test_the_message_names_the_flag(self) -> None:
        with pytest.raises(TestBusterError, match="--timeout"):
            parse_duration("inf", label="--timeout")

    def test_a_plain_number_still_parses(self) -> None:
        assert parse_duration("2.5") == pytest.approx(2.5)


class TestReversedStatusRange:
    """Finding 9: a backwards status range must be rejected, not silently empty."""

    def test_backwards_range_raises(self) -> None:
        with pytest.raises(TestBusterError, match="backwards"):
            build_expectations(
                status=["500-200"], body_regex=None, json_specs=None, max_latency_s=None
            )

    def test_forward_range_still_works(self) -> None:
        exp = build_expectations(
            status=["200-204"], body_regex=None, json_specs=None, max_latency_s=None
        )
        assert exp.status is not None and 202 in exp.status
