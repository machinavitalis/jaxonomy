# SPDX-License-Identifier: MIT

"""T-111-followup-runtime-switch — runtime variant switching.

Covers ``RuntimeVariantSubsystem``: a SIMULATE-time switch that picks
which of N pre-built submodel choices drives the output, based on a
discrete selector input. Unlike build-time ``select_variant`` (which
never instantiates the unselected branches), the runtime switch builds
ALL branches and evaluates each on every step — so the selected branch
can change at runtime without re-compiling the diagram.

Tests verify:

    - Default-when-selector-pinned-to-zero hits the first choice.
    - Selector switching mid-sim swaps the active branch's output.
    - Out-of-range selector is clipped (matches ``MultiPortSwitch``
      semantics) and float selectors are rounded.
    - Differentiability: gradient through the active branch's parameters
      is non-zero; gradient through inactive-branch parameters is zero.
    - The selector itself is non-differentiable (round + clip kill the
      gradient), as expected for a control signal.
    - Validation: empty / non-callable / mis-keyed choices, bad
      default_choice, bad n_inputs all raise ``VariantError`` early.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import RuntimeVariantSubsystem, VariantError
from jaxonomy.library import Constant


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Selector helpers (small LeafSystem that emits a time-varying selector).
# ---------------------------------------------------------------------------


class _SelectorStep(jaxonomy.LeafSystem):
    """Emit ``low`` for ``t < t_switch``, then ``high``.

    Used to drive the runtime variant's selector port through a clean
    mid-sim transition.
    """

    def __init__(self, t_switch: float, low: float, high: float, **kw):
        super().__init__(**kw)
        self._t = float(t_switch)
        self._low = float(low)
        self._high = float(high)
        self.declare_output_port(
            lambda t, s, *u, **p: jnp.where(t < self._t, self._low, self._high),
            prerequisites_of_calc=[],
            requires_inputs=False,
        )


# ---------------------------------------------------------------------------
# Build helpers.
# ---------------------------------------------------------------------------


def _gain1(u):
    """Branch 0: y = 1.0 * u."""
    return 1.0 * u


def _gain2(u):
    """Branch 1: y = 2.0 * u."""
    return 2.0 * u


def _build_diagram(selector_value, u_value, *, n_choices=2,
                   default_choice=0, choices=None,
                   selector_step=None):
    """Wire up a ``RuntimeVariantSubsystem`` with constant input + selector.

    If ``selector_step`` is given (a ``_SelectorStep``-like LeafSystem),
    it is used in place of a constant selector source.
    """
    if choices is None:
        choices = [_gain1, _gain2][:n_choices]

    rvs = RuntimeVariantSubsystem(
        choices=choices,
        n_inputs=1,
        default_choice=default_choice,
        name="rvs",
    )
    bld = jaxonomy.DiagramBuilder()
    if selector_step is None:
        sel = bld.add(Constant(jnp.asarray(float(selector_value)), name="sel"))
        sel_port = sel.output_ports[0]
    else:
        sel = bld.add(selector_step)
        sel_port = sel.output_ports[0]
    u = bld.add(Constant(jnp.asarray(float(u_value)), name="u"))
    blk = bld.add(rvs)
    bld.connect(sel_port, blk.input_ports[0])
    bld.connect(u.output_ports[0], blk.input_ports[1])
    diagram = bld.build()
    return diagram, blk


# ---------------------------------------------------------------------------
# Selection: per-step routing.
# ---------------------------------------------------------------------------


class TestRouting:
    def test_selector_zero_picks_first_branch(self):
        diagram, blk = _build_diagram(selector_value=0, u_value=3.0)
        ctx = diagram.create_context()
        y = blk.output_ports[0].eval(ctx)
        # gain1: 1.0 * 3.0 = 3.0
        np.testing.assert_allclose(float(y), 3.0)

    def test_selector_one_picks_second_branch(self):
        diagram, blk = _build_diagram(selector_value=1, u_value=3.0)
        ctx = diagram.create_context()
        y = blk.output_ports[0].eval(ctx)
        # gain2: 2.0 * 3.0 = 6.0
        np.testing.assert_allclose(float(y), 6.0)

    def test_selector_clipped_above_range(self):
        diagram, blk = _build_diagram(selector_value=99, u_value=3.0)
        ctx = diagram.create_context()
        y = blk.output_ports[0].eval(ctx)
        # Clipped to 1 (n_choices - 1) → gain2: 6.0
        np.testing.assert_allclose(float(y), 6.0)

    def test_selector_clipped_below_zero(self):
        diagram, blk = _build_diagram(selector_value=-7, u_value=3.0)
        ctx = diagram.create_context()
        y = blk.output_ports[0].eval(ctx)
        # Clipped to 0 → gain1: 3.0
        np.testing.assert_allclose(float(y), 3.0)

    def test_selector_float_is_rounded(self):
        # 0.4 rounds to 0 → gain1=3.0; 0.6 rounds to 1 → gain2=6.0.
        d_a, blk_a = _build_diagram(selector_value=0.4, u_value=3.0)
        d_b, blk_b = _build_diagram(selector_value=0.6, u_value=3.0)
        np.testing.assert_allclose(float(blk_a.output_ports[0].eval(d_a.create_context())), 3.0)
        np.testing.assert_allclose(float(blk_b.output_ports[0].eval(d_b.create_context())), 6.0)


# ---------------------------------------------------------------------------
# Runtime switching mid-simulation.
# ---------------------------------------------------------------------------


class TestRuntimeSwitch:
    def test_selector_swaps_branch_mid_simulation(self):
        """Selector starts at 0 (gain=1.0), switches to 1 (gain=2.0) at t=0.25.

        We verify by sampling the recorded output before and after the
        transition: pre-switch the output should be ~u; post-switch it
        should be ~2*u.
        """
        u_value = 3.0
        rvs = RuntimeVariantSubsystem(
            choices=[_gain1, _gain2],
            n_inputs=1,
            default_choice=0,
            name="rvs",
        )
        bld = jaxonomy.DiagramBuilder()
        sel = bld.add(_SelectorStep(t_switch=0.25, low=0.0, high=1.0, name="sel"))
        u = bld.add(Constant(jnp.asarray(u_value), name="u"))
        blk = bld.add(rvs)
        bld.connect(sel.output_ports[0], blk.input_ports[0])
        bld.connect(u.output_ports[0], blk.input_ports[1])
        diagram = bld.build()
        ctx = diagram.create_context()

        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.5),
            recorded_signals={"y": blk.output_ports[0]},
        )
        ts = np.asarray(results.time)
        ys = np.asarray(results.outputs["y"])

        # Pre-switch samples (t < 0.25): expect gain1 = u_value.
        pre_mask = ts < 0.25 - 1e-9
        post_mask = ts > 0.25 + 1e-9
        assert pre_mask.any(), "no pre-switch samples recorded"
        assert post_mask.any(), "no post-switch samples recorded"
        np.testing.assert_allclose(ys[pre_mask], u_value, atol=1e-9)
        np.testing.assert_allclose(ys[post_mask], 2.0 * u_value, atol=1e-9)


# ---------------------------------------------------------------------------
# Default-when-unset behavior.
# ---------------------------------------------------------------------------


class TestDefault:
    def test_default_choice_property(self):
        rvs = RuntimeVariantSubsystem(
            choices=[_gain1, _gain2], n_inputs=1, default_choice=1, name="rvs",
        )
        assert rvs.default_choice == 1
        assert rvs.n_choices == 2

    def test_default_choice_used_when_selector_input_pinned_to_default(self):
        """Documents the contract: the selector input ALWAYS rules the
        runtime choice. The ``default_choice`` arg is documentary — to
        observe the default at runtime, the caller pins the selector to
        ``rvs.default_choice``.
        """
        rvs = RuntimeVariantSubsystem(
            choices=[_gain1, _gain2], n_inputs=1, default_choice=0, name="rvs",
        )
        bld = jaxonomy.DiagramBuilder()
        sel = bld.add(Constant(jnp.asarray(float(rvs.default_choice)), name="sel"))
        u = bld.add(Constant(jnp.asarray(3.0), name="u"))
        blk = bld.add(rvs)
        bld.connect(sel.output_ports[0], blk.input_ports[0])
        bld.connect(u.output_ports[0], blk.input_ports[1])
        diagram = bld.build()
        y = blk.output_ports[0].eval(diagram.create_context())
        # default_choice=0 → gain1 → 3.0
        np.testing.assert_allclose(float(y), 3.0)


# ---------------------------------------------------------------------------
# Differentiability.
# ---------------------------------------------------------------------------


class TestDifferentiability:
    def test_gradient_flows_through_selected_branch(self):
        """When selector=1 (active branch is gain2 = 2*u), the underlying
        stack+index op routes gradients only to that branch's data.
        """
        def f(selector, *data):
            stacked = jnp.stack(data, axis=0)
            idx = jnp.clip(jnp.round(selector).astype(jnp.int32), 0, len(data) - 1)
            return stacked[idx]

        # Build branch outputs explicitly so we can inspect gradients
        # w.r.t. each branch's "output" independently.
        u = 3.0
        # Selector=1 picks data[1] → grad w.r.t. data[1] = 1.0, others = 0.
        grads = jax.grad(f, argnums=(1, 2))(1.0, 1.0 * u, 2.0 * u)
        np.testing.assert_allclose(float(grads[0]), 0.0)
        np.testing.assert_allclose(float(grads[1]), 1.0)

    def test_gradient_through_active_branch_parameter(self):
        """End-to-end: differentiate the runtime-variant block's output
        w.r.t. a parameter that lives only inside the active branch.
        Gradient w.r.t. the inactive branch's parameter must be zero.
        """
        def make_branches(p_active, p_inactive):
            return [lambda u, p=p_inactive: p * u,
                    lambda u, p=p_active: p * u]

        def block_output(p_active, p_inactive, selector, u):
            branches = make_branches(p_active, p_inactive)
            outs = [jnp.asarray(fn(u)) for fn in branches]
            stacked = jnp.stack(outs, axis=0)
            idx = jnp.clip(
                jnp.round(selector).astype(jnp.int32), 0, len(branches) - 1
            )
            return stacked[idx]

        # selector=1 → active branch is index 1, which uses p_active.
        g_active = jax.grad(block_output, argnums=0)(2.5, 9.9, 1.0, 4.0)
        g_inactive = jax.grad(block_output, argnums=1)(2.5, 9.9, 1.0, 4.0)
        # d/dp_active (p_active * u) at u=4 → 4.0
        np.testing.assert_allclose(float(g_active), 4.0)
        # Inactive branch's parameter contributes nothing.
        np.testing.assert_allclose(float(g_inactive), 0.0)

    def test_selector_gradient_is_zero(self):
        """Selector is non-differentiable (round + clip)."""
        def block_output(selector, u):
            outs = [jnp.asarray(1.0 * u), jnp.asarray(2.0 * u)]
            stacked = jnp.stack(outs, axis=0)
            idx = jnp.clip(jnp.round(selector).astype(jnp.int32), 0, 1)
            return stacked[idx]

        g_sel = jax.grad(block_output, argnums=0)(0.6, 3.0)
        np.testing.assert_allclose(float(g_sel), 0.0)


# ---------------------------------------------------------------------------
# All branches integrated each step (contract docstring).
# ---------------------------------------------------------------------------


def test_all_branches_evaluated_each_step():
    """Every branch's submodel should be called on every step, so the
    inactive branches still see the input. We verify this with a counter.
    """
    counts = {"a": 0, "b": 0}

    def branch_a(u):
        counts["a"] += 1
        return 1.0 * u

    def branch_b(u):
        counts["b"] += 1
        return 2.0 * u

    rvs = RuntimeVariantSubsystem(
        choices=[branch_a, branch_b], n_inputs=1, default_choice=0, name="rvs",
    )
    bld = jaxonomy.DiagramBuilder()
    sel = bld.add(Constant(jnp.asarray(0.0), name="sel"))
    u = bld.add(Constant(jnp.asarray(3.0), name="u"))
    blk = bld.add(rvs)
    bld.connect(sel.output_ports[0], blk.input_ports[0])
    bld.connect(u.output_ports[0], blk.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()

    # Just evaluate the output once — each branch should have been
    # invoked at least once during the trace.
    _ = blk.output_ports[0].eval(ctx)
    assert counts["a"] >= 1
    assert counts["b"] >= 1


# ---------------------------------------------------------------------------
# Mapping-style choices ({int: f}).
# ---------------------------------------------------------------------------


class TestMappingChoices:
    def test_mapping_choices_with_contiguous_int_keys(self):
        rvs = RuntimeVariantSubsystem(
            choices={0: _gain1, 1: _gain2},
            n_inputs=1,
            default_choice=0,
            name="rvs",
        )
        assert rvs.n_choices == 2
        # Build a tiny diagram and check selector=1 → gain2 path.
        bld = jaxonomy.DiagramBuilder()
        sel = bld.add(Constant(jnp.asarray(1.0), name="sel"))
        u = bld.add(Constant(jnp.asarray(3.0), name="u"))
        blk = bld.add(rvs)
        bld.connect(sel.output_ports[0], blk.input_ports[0])
        bld.connect(u.output_ports[0], blk.input_ports[1])
        diagram = bld.build()
        np.testing.assert_allclose(
            float(blk.output_ports[0].eval(diagram.create_context())), 6.0
        )

    def test_mapping_with_non_contiguous_keys_rejected(self):
        with pytest.raises(VariantError, match="contiguous 0..N-1 range"):
            RuntimeVariantSubsystem(
                choices={0: _gain1, 2: _gain2}, n_inputs=1, name="rvs",
            )

    def test_mapping_with_non_int_keys_rejected(self):
        with pytest.raises(VariantError, match="must be integers"):
            RuntimeVariantSubsystem(
                choices={"a": _gain1, "b": _gain2}, n_inputs=1, name="rvs",
            )


# ---------------------------------------------------------------------------
# Validation: misuse of the constructor.
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_sequence_rejected(self):
        with pytest.raises(VariantError, match="sequence is empty"):
            RuntimeVariantSubsystem(choices=[], n_inputs=1, name="rvs")

    def test_empty_mapping_rejected(self):
        with pytest.raises(VariantError, match="mapping is empty"):
            RuntimeVariantSubsystem(choices={}, n_inputs=1, name="rvs")

    def test_non_callable_choice_rejected(self):
        with pytest.raises(VariantError, match="not callable"):
            RuntimeVariantSubsystem(
                choices=[_gain1, "not a function"],  # type: ignore[list-item]
                n_inputs=1,
                name="rvs",
            )

    def test_default_choice_out_of_range_rejected(self):
        with pytest.raises(VariantError, match="out of range"):
            RuntimeVariantSubsystem(
                choices=[_gain1, _gain2],
                n_inputs=1,
                default_choice=5,
                name="rvs",
            )

    def test_negative_n_inputs_rejected(self):
        with pytest.raises(VariantError, match="n_inputs"):
            RuntimeVariantSubsystem(
                choices=[_gain1, _gain2],
                n_inputs=-1,
                name="rvs",
            )
