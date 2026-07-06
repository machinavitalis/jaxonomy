# SPDX-License-Identifier: MIT
"""T-038: ``AcausalCompiler(..., condition_number_threshold=...)``.

The condition-number warning emitted by ``IndexReduction`` is now
configurable via a new ``AcausalCompiler`` parameter. The default
(``1e4``) preserves prior behaviour; users with intentionally
ill-conditioned Jacobians (e.g. multi-domain electrical/thermal)
can raise it to silence the warning, and CI suites that want to
fail loudly on borderline-singular configs can lower it.
"""

from __future__ import annotations

import warnings

from jaxonomy.acausal import (
    AcausalCompiler,
    AcausalDiagram,
    EqnEnv,
    electrical as elec,
)


def _build_rc_compiler(*, threshold):
    """Return an AcausalCompiler for a borderline-conditioned RC circuit.

    The default initial conditions of this network produce a Jacobian with
    condition number ~1.4e4, just above the legacy 1e4 threshold.
    """
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=1.0)
    r = elec.Resistor(ev, name="r", R=10.0)
    c = elec.Capacitor(
        ev, name="c", C=1e-3, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    return AcausalCompiler(ev, ad, condition_number_threshold=threshold)


def test_default_threshold_warns_on_ill_conditioned_rc():
    """At the default 1e4 threshold the borderline RC trips the warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _build_rc_compiler(threshold=1e4)()
    cond_warns = [x for x in w if "condition number" in str(x.message)]
    assert cond_warns, (
        "default threshold=1e4 should emit a condition-number warning "
        "for this RC configuration"
    )


def test_relaxed_threshold_suppresses_warning():
    """A user-relaxed threshold lets known-conditioned models compile clean."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _build_rc_compiler(threshold=1e8)()
    cond_warns = [x for x in w if "condition number" in str(x.message)]
    assert not cond_warns, (
        f"threshold=1e8 should suppress the warning but got: "
        f"{[str(x.message)[:120] for x in cond_warns]}"
    )


def test_strict_threshold_warns_on_well_conditioned_rc():
    """A user-tightened threshold can promote borderline-clean models to noisy."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _build_rc_compiler(threshold=1.0)()  # absurdly strict
    cond_warns = [x for x in w if "condition number" in str(x.message)]
    assert cond_warns, "threshold=1.0 should fire on any non-trivial Jacobian"
