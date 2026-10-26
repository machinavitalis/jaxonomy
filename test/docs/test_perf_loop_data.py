# SPDX-License-Identifier: MIT

"""The numbers of ``docs/perf_loop.md`` follow from the recorded measurements.

The measurements in ``benchmarks/perf_loop/`` were taken once and are not
re-run here. What is checked is that the summaries, the figures and the
figures quoted on the page are the ones those measurements give.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "benchmarks" / "perf_loop"
_PAGE = (_ROOT / "docs" / "perf_loop.md").read_text()


def _load(name):
    return json.loads((_DATA / name).read_text())


def _geomean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def test_cumulative_speedup_is_the_product_of_the_kept_changes():
    data = _load("accepted_changes.json")
    kept = [c for c in data["changes"] if c["kept"]]
    product = math.prod(c["speedup"] for c in kept)
    assert kept[-1]["cumulative"] == pytest.approx(product, rel=1e-3)
    assert f"{len(kept)} kept changes" in _PAGE
    assert f"&#215;{product:.2f}" in _PAGE
    assert data["verdicts"]["accepted"] == len(data["changes"])
    assert sum(data["verdicts"].values()) == data["iterations"]


def test_end_to_end_summaries_follow_from_the_measurements():
    data = _load("end_to_end.json")
    pairs = data["measurements"]
    assert len(pairs) == data["pairs"]
    for p in pairs:
        assert p["speedup"] == pytest.approx(p["before_s"] / p["after_s"], rel=2e-3)
    assert data["speedup"] == pytest.approx(_geomean(p["speedup"] for p in pairs), rel=1e-3)
    for row in data["by_workload"]:
        mine = [p["speedup"] for p in pairs if p["workload"] == row["workload"]]
        assert len(mine) == row["pairs"]
        assert row["speedup"] == pytest.approx(_geomean(mine), rel=1e-3)
    assert f"&#215;{data['speedup']:.2f}" in _PAGE
    assert f"{data['pairs']} timed situations" in _PAGE


def test_figures_are_the_ones_the_data_draws(tmp_path):
    spec = importlib.util.spec_from_file_location("make_figures", _DATA / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    img = _ROOT / "docs" / "img"
    assert module.cumulative(_load("accepted_changes.json")) == (
        img / "perf_loop_cumulative.svg"
    ).read_text()
    assert module.end_to_end(_load("end_to_end.json")) == (
        img / "perf_loop_end_to_end.svg"
    ).read_text()
