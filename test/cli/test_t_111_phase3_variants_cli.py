# SPDX-License-Identifier: MIT

"""T-111 phase 3 — variant-config CLI helper (``jaxonomy variants``).

Exercises ``jaxonomy variants list / dump / apply`` against a small
diagram builder defined in this test module. The CLI resolves the
builder via ``--builder package.module:function``; we use this very
module's own qualified name so the test runs without installing extra
fixtures.
"""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

import jaxonomy
from jaxonomy.cli.run_variants import jaxonomy_variants
from jaxonomy.framework import Variant, select_variant
from jaxonomy.library import Constant, Gain, Integrator


# ---------------------------------------------------------------------------
# Fixture builder (resolved by the CLI via --builder module:fn).
# ---------------------------------------------------------------------------


def _build_p_controller():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="P_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="p_controller")


def _build_pi_controller():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="P_gain"))
    i_gain = b.add(Gain(0.5, name="I_gain"))
    integ = b.add(Integrator(0.0, name="I_state"))
    b.connect(g.output_ports[0], i_gain.input_ports[0])
    b.connect(i_gain.output_ports[0], integ.input_ports[0])
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="pi_controller")


def _build_passthrough_plant():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(1.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="passthrough_plant")


def _build_double_plant():
    b = jaxonomy.DiagramBuilder()
    g = b.add(Gain(2.0, name="plant_gain"))
    b.export_input(g.input_ports[0], name="u")
    b.export_output(g.output_ports[0], name="y")
    return b.build(name="double_plant")


def build_fixture_diagram():
    """Top-level diagram with two named variants; CLI builder target."""
    controller = select_variant(
        Variant(
            choices={"p": _build_p_controller, "pi": _build_pi_controller},
            default="p",
            name="controller",
        )
    )
    plant = select_variant(
        Variant(
            choices={"lti": _build_passthrough_plant, "doubler": _build_double_plant},
            default="lti",
            name="plant",
        )
    )
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(3.0, name="src"))
    b.add(controller)
    b.add(plant)
    b.connect(src.output_ports[0], controller.input_ports[0])
    b.connect(controller.output_ports[0], plant.input_ports[0])
    b.export_output(plant.output_ports[0], name="y")
    return b.build(name="root")


# The CLI imports this module dynamically via --builder; the spec is
# this file's qualified name + the builder function.
_BUILDER_SPEC = f"{__name__}:build_fixture_diagram"


# ---------------------------------------------------------------------------
# Builder-loader error paths.
# ---------------------------------------------------------------------------


def test_missing_colon_in_builder_spec_raises():
    runner = CliRunner()
    res = runner.invoke(jaxonomy_variants, ["list", "--builder", "no_colon_here"])
    assert res.exit_code != 0
    assert "package.module:function" in res.output


def test_unknown_module_in_builder_spec_raises():
    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["list", "--builder", "no.such.module.xyz:fn"],
    )
    assert res.exit_code != 0
    assert "failed to import module" in res.output


def test_unknown_attr_in_builder_spec_raises():
    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["list", "--builder", f"{__name__}:no_such_function"],
    )
    assert res.exit_code != 0
    assert "no attribute" in res.output


# ---------------------------------------------------------------------------
# `list` subcommand.
# ---------------------------------------------------------------------------


def test_list_text_format_prints_named_variants():
    runner = CliRunner()
    res = runner.invoke(jaxonomy_variants, ["list", "--builder", _BUILDER_SPEC])
    assert res.exit_code == 0, res.output
    # Each named variant appears on its own line with active + choices.
    assert "controller" in res.output
    assert "plant" in res.output
    assert "active='p'" in res.output
    assert "active='lti'" in res.output
    assert "'pi'" in res.output  # in the choices list


def test_list_json_format_is_parseable_array_of_records():
    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants, ["list", "--builder", _BUILDER_SPEC, "--format", "json"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert isinstance(payload, list)
    by_name = {row["name"]: row for row in payload}
    assert by_name["controller"]["active"] == "p"
    assert set(by_name["controller"]["choices"]) == {"p", "pi"}
    assert by_name["plant"]["active"] == "lti"


# ---------------------------------------------------------------------------
# `dump` subcommand.
# ---------------------------------------------------------------------------


def test_dump_prints_active_choices_as_json_object():
    runner = CliRunner()
    res = runner.invoke(jaxonomy_variants, ["dump", "--builder", _BUILDER_SPEC])
    assert res.exit_code == 0, res.output
    parsed = json.loads(res.output)
    assert parsed == {"controller": "p", "plant": "lti"}


def test_dump_writes_output_file_when_output_given(tmp_path):
    out_path = tmp_path / "config.json"
    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["dump", "--builder", _BUILDER_SPEC, "--output", str(out_path)],
    )
    assert res.exit_code == 0, res.output
    assert out_path.exists()
    text = out_path.read_text()
    assert text.endswith("\n")  # CLI appends trailing newline when writing files
    assert json.loads(text) == {"controller": "p", "plant": "lti"}


def test_dump_compact_form_when_indent_zero():
    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["dump", "--builder", _BUILDER_SPEC, "--indent", "0"],
    )
    assert res.exit_code == 0
    # Trailing newline is from click.echo, not from the JSON itself —
    # the JSON payload must be on a single line.
    body = res.output.rstrip("\n")
    assert "\n" not in body
    assert json.loads(body) == {"controller": "p", "plant": "lti"}


# ---------------------------------------------------------------------------
# `apply` subcommand.
# ---------------------------------------------------------------------------


def test_apply_swaps_choices_and_dumps_result(tmp_path):
    cfg_path = tmp_path / "swap.json"
    cfg_path.write_text(json.dumps({"controller": "pi", "plant": "doubler"}))

    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["apply", "--builder", _BUILDER_SPEC, "--config", str(cfg_path)],
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.output) == {"controller": "pi", "plant": "doubler"}


def test_apply_rejects_unknown_variant_with_clean_error(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps({"nonexistent": "p"}))

    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["apply", "--builder", _BUILDER_SPEC, "--config", str(cfg_path)],
    )
    assert res.exit_code != 0
    assert "no Variant with name" in res.output


def test_apply_rejects_non_object_json_with_clean_error(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps([1, 2, 3]))

    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        ["apply", "--builder", _BUILDER_SPEC, "--config", str(cfg_path)],
    )
    assert res.exit_code != 0
    assert "expected a JSON object" in res.output


def test_apply_writes_canonicalized_output_file(tmp_path):
    cfg_path = tmp_path / "swap.json"
    cfg_path.write_text(json.dumps({"controller": "pi"}))  # only one swap
    out_path = tmp_path / "applied.json"

    runner = CliRunner()
    res = runner.invoke(
        jaxonomy_variants,
        [
            "apply",
            "--builder", _BUILDER_SPEC,
            "--config", str(cfg_path),
            "--output", str(out_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_path.exists()
    # Output is canonicalized: includes BOTH variants (the unchanged
    # one keeps its default), sorted-keys JSON.
    assert json.loads(out_path.read_text()) == {
        "controller": "pi",
        "plant": "lti",
    }
