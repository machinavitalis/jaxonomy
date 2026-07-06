# SPDX-License-Identifier: MIT

"""Regression test for T-118-followup-state-machine-time-mode.

Before the followup, ``StateMachineBuilder.build()`` hardcoded
``time_mode="agnostic"`` and ``dt=None``. Users who wanted a discrete-time
state machine had to construct the builder, build the agnostic block, reach
into its private ``_sm`` attribute, and re-wrap with
``StateMachine(sm_data=_sm, ..., time_mode="discrete")`` — Simulink-refugee
surface area.

The followup adds ``time_mode=`` and ``dt=`` kwargs to ``build()`` so the
canonical pattern ``builder.build(time_mode="discrete", dt=0.1)`` returns a
discrete-time block directly. The default stays ``time_mode="agnostic"`` for
backwards compatibility.
"""

from __future__ import annotations

import pytest

from jaxonomy.framework.state_machine_builder import StateMachineBuilder


def _trivial_builder():
    """Two-state machine with a string guard (callables are rejected by
    StateMachineBuilder.add_transition).

    The builder discovers ``u`` automatically by scanning the guard string —
    no explicit ``add_input`` call is needed.
    """
    b = StateMachineBuilder()
    s1 = b.add_state("s1")
    s2 = b.add_state("s2")
    b.add_transition(s1, s2, guard="u > 0.5")
    b.set_initial_state(s1)
    return b


def test_build_default_is_agnostic():
    """The default ``time_mode`` must remain ``"agnostic"`` so existing
    builders that didn't pass the kwarg keep their byte-equivalent
    behaviour."""
    block = _trivial_builder().build()
    assert block.time_mode == "agnostic"


def test_build_discrete_requires_dt():
    with pytest.raises(ValueError, match="dt"):
        _trivial_builder().build(time_mode="discrete")


def test_build_discrete_with_dt():
    """A discrete-time state machine built directly from the builder must
    register the requested time mode; ``dt`` is internalised by the
    underlying ``StateMachine`` block as a periodic-event period and is
    not currently re-exposed as a public attribute (which is why this
    test asserts on ``time_mode`` rather than ``dt`` itself)."""
    block = _trivial_builder().build(time_mode="discrete", dt=0.05)
    assert block.time_mode == "discrete"


def test_build_rejects_unknown_time_mode():
    with pytest.raises(ValueError, match="time_mode"):
        _trivial_builder().build(time_mode="bogus")


def test_build_rejects_dt_with_non_discrete_mode():
    """``dt=`` is only meaningful with ``time_mode='discrete'`` — passing
    it elsewhere should fail loudly rather than be silently ignored."""
    with pytest.raises(ValueError, match="dt"):
        _trivial_builder().build(time_mode="agnostic", dt=0.05)
