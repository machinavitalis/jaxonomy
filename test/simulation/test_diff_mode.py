# SPDX-License-Identifier: MIT

"""Resolution semantics for ``SimulatorOptions.diff_mode``.

``diff_mode`` is a self-documenting selector that resolves into the canonical
``enable_autodiff`` flag (diff-mode follow-up). The footgun it removes:
forward-mode autodiff requires ``enable_autodiff=False`` (the reverse-mode
``custom_vjp`` intercepts ``jax.jacfwd``), so consumers previously had to turn
*off* a flag named ``enable_autodiff`` to *do* autodiff.
"""

import dataclasses

import pytest

from jaxonomy.simulation.types import SimulatorOptions


def test_reverse_mode_enables_adjoint():
    assert SimulatorOptions(diff_mode="reverse").enable_autodiff is True


def test_forward_mode_disables_adjoint():
    # The whole point: forward-mode wants enable_autodiff=False so JAX traces
    # the real solver ops, and the user never has to spell that out.
    assert SimulatorOptions(diff_mode="forward").enable_autodiff is False


def test_none_mode_disables_adjoint():
    assert SimulatorOptions(diff_mode="none").enable_autodiff is False


def test_auto_leaves_enable_autodiff_untouched():
    assert SimulatorOptions(diff_mode="auto").enable_autodiff is False
    assert SimulatorOptions(diff_mode="auto", enable_autodiff=True).enable_autodiff is True
    assert SimulatorOptions(diff_mode=None, enable_autodiff=True).enable_autodiff is True


def test_diff_mode_is_cleared_after_resolution():
    # Cleared so a later dataclasses.replace(enable_autodiff=...) is not
    # re-clobbered by __post_init__ re-applying the mode.
    assert SimulatorOptions(diff_mode="reverse").diff_mode is None


def test_replace_enable_autodiff_is_not_reclobbered():
    opts = SimulatorOptions(diff_mode="reverse")
    assert opts.enable_autodiff is True
    flipped = dataclasses.replace(opts, enable_autodiff=False)
    assert flipped.enable_autodiff is False


def test_invalid_diff_mode_raises():
    with pytest.raises(ValueError, match="diff_mode"):
        SimulatorOptions(diff_mode="backward")


def test_conflicting_forward_with_enable_autodiff_raises():
    # diff_mode='forward' + enable_autodiff=True is contradictory; fail loud.
    with pytest.raises(ValueError, match="Conflicting differentiation settings"):
        SimulatorOptions(diff_mode="forward", enable_autodiff=True)
    with pytest.raises(ValueError, match="Conflicting differentiation settings"):
        SimulatorOptions(diff_mode="none", enable_autodiff=True)


def test_reverse_with_enable_autodiff_true_is_consistent():
    # Not a conflict — both ask for the adjoint.
    assert SimulatorOptions(diff_mode="reverse", enable_autodiff=True).enable_autodiff is True
