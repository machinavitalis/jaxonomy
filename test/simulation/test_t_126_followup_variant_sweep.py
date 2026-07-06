# SPDX-License-Identifier: MIT

"""Regression test for T-126-followup-variant-sweep-vmap.

``simulate_variant_sweep`` runs ``simulate_batch`` (or ``simulate`` when no
batch is supplied) once per variant configuration and returns a dict keyed
by configuration. Variant-axis vmap is not possible — the pytree shape of
the simulator state is not stable across variant choices — so the per-
variant axis stays a Python loop. This helper just packages the loop so
tutorial authors don't reinvent it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import Variant, select_variant
from jaxonomy.library import Gain, Integrator
from jaxonomy.simulation import simulate_variant_sweep


pytestmark = pytest.mark.minimal


def _build_p_controller() -> jaxonomy.Diagram:
    builder = jaxonomy.DiagramBuilder()
    p = builder.add(Gain(2.0, name="P_gain"))
    builder.export_input(p.input_ports[0], name="u")
    builder.export_output(p.output_ports[0], name="y")
    return builder.build(name="p_controller")


def _build_double_p_controller() -> jaxonomy.Diagram:
    """Two P gains in series — structurally distinct from the single-gain
    variant so the variant choice is observable in the output."""
    builder = jaxonomy.DiagramBuilder()
    p1 = builder.add(Gain(2.0, name="P_gain1"))
    p2 = builder.add(Gain(3.0, name="P_gain2"))
    builder.connect(p1.output_ports[0], p2.input_ports[0])
    builder.export_input(p1.input_ports[0], name="u")
    builder.export_output(p2.output_ports[0], name="y")
    return builder.build(name="double_p_controller")


def _make_controller_variant() -> Variant:
    return Variant(
        choices={
            "single": _build_p_controller,
            "double": _build_double_p_controller,
        },
        default="single",
        name="controller",
    )


def _build_top_diagram():
    """Build a top-level diagram with one variant + a constant input."""
    from jaxonomy.library import Constant

    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(1.0, name="src"))
    ctrl = select_variant(_make_controller_variant())
    ctrl_block = builder.add(ctrl)
    builder.connect(src.output_ports[0], ctrl_block.input_ports[0])
    builder.export_output(ctrl_block.output_ports[0], name="y")
    return builder.build(name="top"), ctrl_block


def test_simulate_variant_sweep_without_param_batches():
    """Without ``param_batches``, the helper runs one ``simulate`` per
    variant configuration and returns ``SimulationResults`` per key."""
    diag, _ = _build_top_diagram()

    results = simulate_variant_sweep(
        diag,
        t_span=(0.0, 0.1),
        recorded_signals=lambda d: {"y": d.output_ports[0]},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10),
    )

    keys = list(results.keys())
    assert len(keys) == 2, f"expected 2 variant configs, got {len(keys)}: {keys}"

    cfg_to_value = {}
    for cfg, res in results.items():
        cfg_dict = dict(cfg)
        # Last sample of "y": constant input 1.0 propagated through the
        # variant's gain stage.
        y_last = float(np.asarray(res.outputs["y"])[-1])
        cfg_to_value[cfg_dict["controller"]] = y_last

    # Single P-gain (2.0) at unit input → 2.0.
    # Double P-gain (2.0 * 3.0) at unit input → 6.0.
    assert cfg_to_value["single"] == pytest.approx(2.0, abs=1e-6)
    assert cfg_to_value["double"] == pytest.approx(6.0, abs=1e-6)


def test_simulate_variant_sweep_rejects_non_diagram():
    with pytest.raises(TypeError, match="Diagram"):
        simulate_variant_sweep("not a diagram", t_span=(0.0, 1.0))  # type: ignore[arg-type]


def test_simulate_variant_sweep_no_variants_returns_one_config():
    """A diagram with no variant nodes yields a single configuration
    (the empty dict / empty tuple key)."""
    from jaxonomy.library import Constant

    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(1.0, name="src"))
    gain = builder.add(Gain(5.0, name="gain"))
    builder.connect(src.output_ports[0], gain.input_ports[0])
    builder.export_output(gain.output_ports[0], name="y")
    diag = builder.build(name="no_variants")

    results = simulate_variant_sweep(
        diag,
        t_span=(0.0, 0.1),
        recorded_signals={"y": diag.output_ports[0]},
        options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10),
    )
    assert len(results) == 1
    only_key = next(iter(results.keys()))
    assert only_key == ()  # no variant overrides
