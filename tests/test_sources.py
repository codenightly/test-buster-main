"""Tests for request sources: CSV rows and weighted scenario steps."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from testbuster.errors import TestBusterError
from testbuster.sources import (
    RequestCycle,
    RequestSpec,
    Step,
    _fill,
    from_rows,
    from_steps,
    load_rows,
    load_scenario,
)
from testbuster.validation import NO_EXPECTATIONS, Expectations, build_expectations


def _write_scenario(tmp_path: Path, document: Any) -> Path:
    """Write a scenario document as JSON and return its path."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _one_step(tmp_path: Path, step: dict[str, Any]) -> Path:
    """Write a one-step scenario, for every case that varies a single field."""
    return _write_scenario(tmp_path, {"steps": [step]})


def _status_only(code: str) -> Expectations:
    """Build expectations that check one status code and nothing else."""
    return build_expectations(status=[code], body_regex=None, json_specs=None, max_latency_s=None)


class TestRequestSpec:
    def test_the_body_encodes_once(self) -> None:
        spec = RequestSpec("POST", "http://x/", body="hi")
        assert spec.body_bytes == b"hi"
        # One object across two reads proves the spec does not re-encode per send.
        assert spec.body_bytes is spec.body_bytes

    def test_no_body_encodes_to_none(self) -> None:
        assert RequestSpec("GET", "http://x/").body_bytes is None


class TestFill:
    def test_replaces_placeholders(self) -> None:
        assert _fill("http://x/{{id}}", {"id": "7"}) == "http://x/7"

    def test_allows_spaces_in_braces(self) -> None:
        assert _fill("{{ id }}", {"id": "7"}) == "7"

    def test_missing_column_raises(self) -> None:
        with pytest.raises(TestBusterError, match="no column"):
            _fill("{{missing}}", {"id": "7"})


class TestLoadRows:
    def test_reads_a_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_text("id,name\n1,ada\n2,alan\n", encoding="utf-8")
        rows = load_rows(path)
        assert rows == [{"id": "1", "name": "ada"}, {"id": "2", "name": "alan"}]

    def test_a_header_only_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_text("id,name\n", encoding="utf-8")
        with pytest.raises(TestBusterError, match="no data rows"):
            load_rows(path)

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TestBusterError, match="cannot read data file"):
            load_rows(tmp_path / "absent.csv")

    def test_a_short_row_raises_and_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_text("id,name\n1,ada\n2\n", encoding="utf-8")
        # Line 3 holds the short row: the header is line 1.
        with pytest.raises(TestBusterError, match=r"line 3 is short\. It has no value for name"):
            load_rows(path)


class TestDataFileSource:
    def _source(self) -> RequestCycle:
        rows = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        return from_rows(
            rows,
            method="GET",
            url_template="http://x/{{id}}",
            header_templates={"X-Id": "{{id}}"},
            body_template=None,
            expectations=NO_EXPECTATIONS,
        )

    def test_fills_per_row(self) -> None:
        source = self._source()
        assert source.spec_for(0).url == "http://x/1"
        assert source.spec_for(1).url == "http://x/2"
        assert source.spec_for(0).headers == {"X-Id": "1"}

    def test_cycles_past_the_last_row(self) -> None:
        source = self._source()
        assert source.spec_for(3).url == "http://x/1"
        assert source.spec_for(4).url == "http://x/2"

    def test_fills_the_body(self) -> None:
        source = from_rows(
            [{"name": "ada"}],
            method="POST",
            url_template="http://x/",
            header_templates={},
            body_template='{"who": "{{name}}"}',
            expectations=NO_EXPECTATIONS,
        )
        assert source.spec_for(0).body == '{"who": "ada"}'

    def test_reports_one_label(self) -> None:
        assert self._source().labels == ("default",)


class TestScenarioSource:
    def test_weights_set_the_mix(self) -> None:
        source = from_steps(
            [
                Step(RequestSpec("GET", "http://x/a", label="a"), 3),
                Step(RequestSpec("GET", "http://x/b", label="b"), 1),
            ]
        )
        picks = Counter(source.spec_for(i).label for i in range(80))
        # 3:1 over a wheel of length 4 gives exactly 60 and 20.
        assert picks == {"a": 60, "b": 20}

    def test_reduces_weights_by_their_gcd(self) -> None:
        source = from_steps(
            [
                Step(RequestSpec("GET", "http://x/a", label="a"), 2),
                Step(RequestSpec("GET", "http://x/b", label="b"), 2),
            ]
        )
        # The wheel reduces 2:2 to 1:1, so it is length 2.
        assert len(source._wheel) == 2

    def test_reports_every_label(self) -> None:
        source = from_steps(
            [
                Step(RequestSpec("GET", "http://x/a", label="a"), 1),
                Step(RequestSpec("GET", "http://x/b", label="b"), 1),
            ]
        )
        assert set(source.labels) == {"a", "b"}

    def test_empty_scenario_raises(self) -> None:
        with pytest.raises(TestBusterError, match="at least one step"):
            from_steps([])

    def test_a_zero_weight_step_raises(self) -> None:
        # A zero weight wins no place on the wheel, but it still adds a label.
        with pytest.raises(TestBusterError, match="step 1 weight must be at least 1"):
            from_steps([Step(RequestSpec("GET", "http://x/a", label="a"), 0)])

    def test_the_weight_message_names_the_step(self) -> None:
        steps = [
            Step(RequestSpec("GET", "http://x/a", label="a"), 1),
            Step(RequestSpec("GET", "http://x/b", label="b"), 0),
        ]
        with pytest.raises(TestBusterError, match="step 2 weight"):
            from_steps(steps)


class TestLoadScenario:
    def test_reads_json_with_base_url(self, tmp_path: Path) -> None:
        path = _write_scenario(
            tmp_path,
            {
                "base_url": "http://api.example.com",
                "steps": [
                    {"name": "read", "url": "/items", "weight": 2},
                    {"name": "write", "method": "POST", "url": "/items", "body": {"a": 1}},
                ],
            },
        )
        source = load_scenario(path)
        assert set(source.labels) == {"read", "write"}
        read = next(source.spec_for(i) for i in range(4) if source.spec_for(i).label == "read")
        assert read.url == "http://api.example.com/items"
        write = next(source.spec_for(i) for i in range(4) if source.spec_for(i).label == "write")
        assert write.method == "POST"
        assert write.body == '{"a": 1}'

    def test_a_bare_list_is_accepted(self, tmp_path: Path) -> None:
        path = _write_scenario(tmp_path, [{"url": "http://x/a"}])
        assert load_scenario(path).spec_for(0).url == "http://x/a"

    def test_step_expectations_load(self, tmp_path: Path) -> None:
        expect = {"status": ["2xx"], "max_latency_ms": 250}
        path = _one_step(tmp_path, {"url": "http://x/a", "expect": expect})
        exp = load_scenario(path).spec_for(0).expectations
        assert exp.status is not None and 200 in exp.status
        assert exp.max_latency_s == pytest.approx(0.25)

    def test_a_single_status_expands(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "expect": {"status": "404"}})
        assert load_scenario(path).spec_for(0).expectations.status == frozenset({404})

    def test_an_unnamed_step_gets_a_number(self, tmp_path: Path) -> None:
        path = _write_scenario(tmp_path, [{"url": "http://x/a"}, {"url": "http://x/b"}])
        assert set(load_scenario(path).labels) == {"step1", "step2"}

    def test_a_json_body_gets_a_content_type(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "method": "POST", "body": {"a": 1}})
        spec = load_scenario(path).spec_for(0)
        assert spec.body == '{"a": 1}'
        assert spec.headers["Content-Type"] == "application/json"

    def test_a_list_body_gets_a_content_type(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "method": "POST", "body": [1, 2]})
        headers = load_scenario(path).spec_for(0).headers
        assert headers == {"Content-Type": "application/json"}

    def test_a_step_content_type_survives_any_spelling(self, tmp_path: Path) -> None:
        step = {
            "url": "http://x/a",
            "method": "POST",
            "headers": {"content-type": "application/vnd.api+json"},
            "body": {"a": 1},
        }
        headers = load_scenario(_one_step(tmp_path, step)).spec_for(0).headers
        assert headers == {"content-type": "application/vnd.api+json"}

    def test_a_text_body_gets_no_content_type(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "method": "POST", "body": "raw"})
        assert load_scenario(path).spec_for(0).headers == {}

    def test_broken_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(TestBusterError, match="not valid JSON"):
            load_scenario(path)

    def test_broken_yaml_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "s.yaml"
        path.write_text("steps: [oops\n", encoding="utf-8")
        with pytest.raises(TestBusterError, match="not valid YAML"):
            load_scenario(path)

    def test_yaml_without_the_extra_explains_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)
        path = tmp_path / "s.yaml"
        path.write_text("steps:\n  - url: http://x/a\n", encoding="utf-8")
        with pytest.raises(TestBusterError, match="yaml extra"):
            load_scenario(path)


class TestScenarioDefaults:
    """A step inherits what the command line set. A step that states a value wins."""

    def test_the_default_method_fills_a_quiet_step(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a"})
        assert load_scenario(path, default_method="POST").spec_for(0).method == "POST"

    def test_a_step_method_beats_the_default(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "method": "put"})
        assert load_scenario(path, default_method="POST").spec_for(0).method == "PUT"

    def test_no_method_anywhere_is_get(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a"})
        assert load_scenario(path).spec_for(0).method == "GET"

    def test_the_default_body_fills_a_quiet_step(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a"})
        assert load_scenario(path, default_body="hello").spec_for(0).body == "hello"

    def test_a_step_body_beats_the_default(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "body": "mine"})
        assert load_scenario(path, default_body="hello").spec_for(0).body == "mine"

    def test_no_body_anywhere_stays_none(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a"})
        assert load_scenario(path).spec_for(0).body is None

    def test_a_default_body_gets_no_content_type(self, tmp_path: Path) -> None:
        # The --body flag carries text of an unknown type, so -H sets the header.
        path = _one_step(tmp_path, {"url": "http://x/a"})
        assert load_scenario(path, default_body="hello").spec_for(0).headers == {}

    def test_the_default_expectations_fill_a_quiet_step(self, tmp_path: Path) -> None:
        expectations = _status_only("201")
        path = _one_step(tmp_path, {"url": "http://x/a"})
        spec = load_scenario(path, default_expectations=expectations).spec_for(0)
        assert spec.expectations is expectations

    def test_a_step_expect_block_beats_the_default(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "expect": {"status": 404}})
        spec = load_scenario(path, default_expectations=_status_only("201")).spec_for(0)
        assert spec.expectations.status == frozenset({404})

    def test_a_quiet_step_keeps_no_expectations_by_default(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a"})
        assert load_scenario(path).spec_for(0).expectations.is_empty


class TestScenarioErrors:
    def test_a_step_without_a_url_raises(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"name": "x"})
        with pytest.raises(TestBusterError, match="no 'url'"):
            load_scenario(path)

    def test_a_non_dict_step_raises(self, tmp_path: Path) -> None:
        path = _write_scenario(tmp_path, {"steps": ["http://x/a"]})
        with pytest.raises(TestBusterError, match="step 1 must be an object"):
            load_scenario(path)

    def test_a_weight_that_is_not_a_number_raises(self, tmp_path: Path) -> None:
        path = _write_scenario(
            tmp_path,
            {"steps": [{"url": "http://x/a"}, {"url": "http://x/b", "weight": "many"}]},
        )
        with pytest.raises(TestBusterError, match="step 2 weight must be a whole number"):
            load_scenario(path)

    def test_a_max_latency_that_is_not_a_number_raises(self, tmp_path: Path) -> None:
        expect = {"max_latency_ms": "soon"}
        path = _one_step(tmp_path, {"url": "http://x/a", "expect": expect})
        with pytest.raises(TestBusterError, match="step 1 max_latency_ms must be a number"):
            load_scenario(path)

    def test_a_base_url_that_is_not_a_string_raises(self, tmp_path: Path) -> None:
        path = _write_scenario(tmp_path, {"base_url": 7, "steps": [{"url": "/items"}]})
        with pytest.raises(TestBusterError, match="step 1 needs 'base_url' to be a string"):
            load_scenario(path)

    def test_a_zero_weight_step_in_a_file_raises(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "weight": 0})
        with pytest.raises(TestBusterError, match="weight must be at least 1"):
            load_scenario(path)

    def test_a_non_dict_expect_block_raises(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "expect": "2xx"})
        with pytest.raises(TestBusterError, match="'expect' must be an object"):
            load_scenario(path)

    def test_non_dict_headers_raise(self, tmp_path: Path) -> None:
        path = _one_step(tmp_path, {"url": "http://x/a", "headers": ["a: b"]})
        with pytest.raises(TestBusterError, match="headers must be an object"):
            load_scenario(path)

    def test_an_empty_steps_list_raises(self, tmp_path: Path) -> None:
        path = _write_scenario(tmp_path, {"steps": []})
        with pytest.raises(TestBusterError, match="non-empty list"):
            load_scenario(path)
