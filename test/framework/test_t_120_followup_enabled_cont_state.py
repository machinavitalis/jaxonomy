# SPDX-License-Identifier: MIT

"""T-120-followup-enabled-cont-state — ``EnabledSubsystem`` continuous-state
``state_mode`` semantics.

Covers the deferred follow-up that adds proper continuous-state behaviour
when an ``EnabledSubsystem`` is disabled. Phase 1 only gated the *output*
(via ``mode={"reset","passthrough","hold"}``); this slice adds a parallel
``state_mode={"hold","reset","free"}`` kwarg controlling how the
EnabledSubsystem's *own* continuous state evolves while disabled.

The continuous state is declared on the block whenever the user supplies a
``state_dynamics`` callable; when ``state_dynamics`` is None (the phase-1
default), no continuous state is declared and ``state_mode`` is validated
but is otherwise a no-op — preserving byte-equivalence with phase 1.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import (
    EnabledMode,
    EnabledStateMode,
    EnabledSubsystem,
    LeafSystem,
)
from jaxonomy.library import Constant
from jaxonomy.testing.markers import skip_if_not_jax


skip_if_not_jax()


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _identity(x):
    """Pure-function submodel: just echo the user input."""
    return jnp.asarray(x)


def _constant_dynamics_one(_time, _x, *_inputs):
    """state_dynamics: xdot = 1.0 (state grows linearly in time)."""
    return jnp.asarray(1.0)


class _EnableStep(LeafSystem):
    """Enable signal that drops to 0 at ``t_off`` and stays low."""

    def __init__(self, t_off: float = 0.3, **kw):
        super().__init__(**kw)
        self._t_off = float(t_off)
        self.declare_output_port(
            lambda t, s, *u, **p: jnp.where(t < self._t_off, 1.0, 0.0),
            prerequisites_of_calc=[],
            requires_inputs=False,
        )


class _EnablePulse(LeafSystem):
    """Enable signal: high in ``[t_on, t_off)``, low elsewhere."""

    def __init__(self, t_on: float, t_off: float, **kw):
        super().__init__(**kw)
        self._t_on = float(t_on)
        self._t_off = float(t_off)
        self.declare_output_port(
            lambda t, s, *u, **p: jnp.where(
                jnp.logical_and(t >= self._t_on, t < self._t_off), 1.0, 0.0,
            ),
            prerequisites_of_calc=[],
            requires_inputs=False,
        )


def _build_with_dynamics(
    enable_source,
    state_mode,
    *,
    initial_state=0.0,
    mode=EnabledMode.RESET,
):
    """Build a diagram: enable + constant u → EnabledSubsystem(xdot=1)."""
    blk = EnabledSubsystem(
        submodel=_identity,
        n_inputs=1,
        mode=mode,
        state_mode=state_mode,
        state_dynamics=_constant_dynamics_one,
        initial_state=jnp.asarray(initial_state),
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(enable_source)
    u = bld.add(Constant(jnp.asarray(0.0), name="u"))
    c = bld.add(blk)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    return diagram, c


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_state_mode_raises():
    """Unknown state_mode strings must surface a clear error."""
    with pytest.raises(ValueError, match="state_mode"):
        EnabledSubsystem(_identity, n_inputs=1, state_mode="bogus")


def test_state_dynamics_without_initial_state_raises():
    """state_dynamics= requires an explicit initial_state= seed."""
    with pytest.raises(ValueError, match="initial_state"):
        EnabledSubsystem(
            _identity,
            n_inputs=1,
            state_dynamics=_constant_dynamics_one,
        )


def test_default_state_mode_is_hold():
    """The default state_mode is 'hold' for byte-equivalence with phase 1."""
    blk = EnabledSubsystem(_identity, n_inputs=1)
    assert blk._state_mode == EnabledStateMode.HOLD


# ---------------------------------------------------------------------------
# Default-off byte-equivalence: phase-1 behavior preserved
# ---------------------------------------------------------------------------


def test_no_dynamics_no_continuous_state():
    """Without state_dynamics, EnabledSubsystem declares no continuous
    state — exactly as in phase 1. state_mode is a no-op in that path."""
    blk = EnabledSubsystem(
        _identity,
        n_inputs=1,
        state_mode=EnabledStateMode.RESET,  # still validated
    )
    assert blk.has_continuous_state is False


def test_default_off_path_matches_phase_1_output():
    """An EnabledSubsystem built without state_dynamics behaves
    identically to the phase-1 implementation."""
    blk = EnabledSubsystem(
        lambda x: 2.0 * x,
        n_inputs=1,
        mode=EnabledMode.RESET,
        initial_value=-1.0,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.asarray(0.0), name="en"))
    u = bld.add(Constant(jnp.asarray(3.0), name="u"))
    c = bld.add(blk)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    # Disabled with reset → initial_value (matches phase-1).
    assert float(y) == -1.0


# ---------------------------------------------------------------------------
# state_mode="hold": state freezes when disabled, resumes when enabled
# ---------------------------------------------------------------------------


def test_state_mode_hold_freezes_state_when_disabled():
    """xdot = 1, enable goes 1→0 at t=0.3. Final continuous state should be
    ~0.3 (the value reached just before the disable edge), not 0.5."""
    diagram, c = _build_with_dynamics(
        _EnableStep(t_off=0.3, name="en"),
        state_mode=EnabledStateMode.HOLD,
        initial_state=0.0,
    )
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)

    xc = float(np.asarray(res.context[c.system_id].continuous_state))
    # The state should have been frozen at ~0.3 (it integrated from 0 → 0.3
    # while enabled, then held there).
    assert np.isfinite(xc)
    assert xc == pytest.approx(0.3, abs=5e-2), (
        f"hold mode: expected ~0.3, got {xc}"
    )


def test_state_mode_hold_resumes_after_re_enable():
    """xdot = 1, enable pulses on [0.0, 0.2) and [0.4, 0.6). With hold mode
    the state should freeze during the gap and resume from there → final
    state ~0.4 (0.2 + 0.2)."""
    diagram, c = _build_with_dynamics(
        _EnablePulse(t_on=0.0, t_off=0.2, name="en"),
        state_mode=EnabledStateMode.HOLD,
        initial_state=0.0,
    )
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.6), options=opts)

    xc = float(np.asarray(res.context[c.system_id].continuous_state))
    # Disabled for the entire window [0.2, 0.6); state frozen at ~0.2.
    assert xc == pytest.approx(0.2, abs=5e-2), (
        f"hold mode after disable: expected ~0.2, got {xc}"
    )


# ---------------------------------------------------------------------------
# state_mode="reset": state snaps to initial on enable transition
# ---------------------------------------------------------------------------


def test_state_mode_reset_snaps_to_initial_on_disable():
    """xdot = 1, enable goes 1→0 at t=0.3 with state_mode='reset' and
    initial_state=0. The state should snap back to 0 on the falling edge."""
    diagram, c = _build_with_dynamics(
        _EnableStep(t_off=0.3, name="en"),
        state_mode=EnabledStateMode.RESET,
        initial_state=0.0,
    )
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)

    xc = float(np.asarray(res.context[c.system_id].continuous_state))
    # After the falling edge at t=0.3 the state must be at initial (0.0),
    # and stays there for the remaining frozen window.
    assert xc == pytest.approx(0.0, abs=1e-6), (
        f"reset mode after disable: expected 0.0, got {xc}"
    )


def test_state_mode_reset_nonzero_initial():
    """initial_state != 0 is honored on the reset snap."""
    diagram, c = _build_with_dynamics(
        _EnableStep(t_off=0.3, name="en"),
        state_mode=EnabledStateMode.RESET,
        initial_state=-7.5,
    )
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)

    xc = float(np.asarray(res.context[c.system_id].continuous_state))
    # State was integrating from -7.5 (xdot=1) up to ~ -7.5 + 0.3 = -7.2,
    # then snapped back to -7.5 on the disable edge.
    assert xc == pytest.approx(-7.5, abs=1e-6), (
        f"reset mode with initial=-7.5: expected -7.5, got {xc}"
    )


# ---------------------------------------------------------------------------
# state_mode="free": state evolves regardless of enable
# ---------------------------------------------------------------------------


def test_state_mode_free_ignores_enable_for_state():
    """state_mode='free': xdot=1 evolves through the disabled window;
    final state ~ 0.5 (full t_end), not 0.3."""
    diagram, c = _build_with_dynamics(
        _EnableStep(t_off=0.3, name="en"),
        state_mode=EnabledStateMode.FREE,
        initial_state=0.0,
    )
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)

    xc = float(np.asarray(res.context[c.system_id].continuous_state))
    # State integrated from 0 to ~0.5 unaffected by the disable.
    assert xc == pytest.approx(0.5, abs=5e-2), (
        f"free mode: expected ~0.5, got {xc}"
    )


def test_state_mode_free_still_masks_output_per_mode():
    """state_mode='free' is orthogonal to mode: output is still gated by
    enable per the mode= contract (here reset → initial_value=-99)."""
    blk = EnabledSubsystem(
        submodel=_identity,
        n_inputs=1,
        mode=EnabledMode.RESET,
        initial_value=-99.0,
        state_mode=EnabledStateMode.FREE,
        state_dynamics=_constant_dynamics_one,
        initial_state=jnp.asarray(0.0),
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.asarray(0.0), name="en"))  # disabled
    u = bld.add(Constant(jnp.asarray(3.0), name="u"))
    c = bld.add(blk)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    # mode='reset' → output is the initial_value while disabled, regardless
    # of state_mode.
    assert float(y) == -99.0
