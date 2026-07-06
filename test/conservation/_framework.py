# SPDX-License-Identifier: MIT
"""
Framework for conservation-law property tests (T-004).

``assert_conserved`` runs a simulation and checks that a scalar invariant
(energy, angular momentum, etc.) is preserved within a tolerance envelope
over the simulation horizon. On failure it surfaces the conserved
quantity, the drift magnitude, and the solver / tolerance settings —
enough to diagnose whether the failure is the solver, the model, or the
test's tolerance assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from jaxonomy.simulation import SimulatorOptions, simulate


class ConservationViolated(AssertionError):
    """Raised when an invariant drifts beyond the declared tolerance."""


@dataclass
class _Report:
    quantity: str
    solver: str
    rtol: float
    atol: float
    e0: float
    ef: float
    abs_drift: float
    rel_drift: float
    tspan: tuple[float, float]
    allowed_rel: float
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"conservation violated: {self.quantity}",
            f"  solver={self.solver} rtol={self.rtol:.1e} atol={self.atol:.1e}",
            f"  tspan={self.tspan}",
            f"  initial value = {self.e0:.8g}",
            f"  final value   = {self.ef:.8g}",
            f"  abs_drift     = {self.abs_drift:.3e}",
            f"  rel_drift     = {self.rel_drift:.3e}  (allowed {self.allowed_rel:.1e})",
        ]
        if self.extra:
            lines.append(f"  extra: {self.extra}")
        return "\n".join(lines)


def assert_conserved(
    system,
    context,
    tspan: tuple[float, float],
    invariant: Callable[[Any], float],
    *,
    solver: str,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_major_steps: int | None = None,
    allowed_rel_drift: float = 1e-4,
    quantity: str = "<invariant>",
    extra: dict[str, Any] | None = None,
    extract_state: Callable[[Any], Any] | None = None,
    mode: str = "endpoint",
    n_check: int = 64,
) -> tuple[float, float]:
    """Simulate, then assert that ``invariant(state)`` drifts less than
    ``allowed_rel_drift`` relative to the initial value.

    Returns ``(initial, final)`` values on success.  Raises
    ``ConservationViolated`` otherwise with a structured report.

    Args:
        system: System / Diagram to simulate.
        context: Initial context.
        tspan: ``(t0, tf)``.
        invariant: Scalar-valued function of the state.  If
            ``extract_state`` is provided it is called first; otherwise
            ``invariant`` receives the context directly.
        solver: ``"rk4"``, ``"dopri5"``, or ``"bdf"``.
        rtol / atol: Solver tolerances.
        max_major_steps: Passed to ``SimulatorOptions``.  Usually required
            for long horizons; estimated automatically by ``simulate`` when
            possible.
        allowed_rel_drift: Maximum allowed ``|Δinvariant| / |invariant_0|``.
        quantity: Human-readable name of the conserved quantity (used in
            error messages).
        extra: Extra fields to attach to the failure report.
        extract_state: Optional transform applied to the context before
            ``invariant`` — e.g. ``lambda ctx: ctx[integ.system_id]``.
        mode: ``"endpoint"`` (default) checks drift only at ``tf`` — fast,
            but misses mid-trajectory excursions where the invariant bulges
            and returns near its start value by ``tf``.  ``"max"`` evaluates
            the invariant at ``n_check`` boundaries across ``tspan`` (the
            simulation is run in segments, restarting from each segment's
            final context) and asserts the *maximum* relative drift over the
            whole trajectory — catching the bulge an endpoint check would
            hide. (T-B7-followup-assert-conserved-max)
        n_check: Number of trajectory checkpoints for ``mode="max"``
            (ignored for ``mode="endpoint"``). Default 64.
    """
    if mode not in ("endpoint", "max"):
        raise ValueError(
            f"assert_conserved: mode must be 'endpoint' or 'max', got {mode!r}"
        )

    sim_kwargs: dict[str, Any] = dict(
        math_backend="jax",
        ode_solver_method=solver,
        rtol=rtol,
        atol=atol,
    )
    if max_major_steps is not None:
        sim_kwargs["max_major_steps"] = max_major_steps
    if solver == "rk4":
        sim_kwargs.setdefault("max_minor_step_size", 0.01)
    opts = SimulatorOptions(**sim_kwargs)

    pre = context if extract_state is None else extract_state(context)
    e0 = float(invariant(pre))
    denom = max(abs(e0), 1e-30)

    def _violation(ef, abs_drift, rel_drift):
        return ConservationViolated(
            str(
                _Report(
                    quantity=quantity,
                    solver=solver,
                    rtol=rtol,
                    atol=atol,
                    e0=e0,
                    ef=ef,
                    abs_drift=abs_drift,
                    rel_drift=rel_drift,
                    tspan=tspan,
                    allowed_rel=allowed_rel_drift,
                    extra=extra or {},
                )
            )
        )

    if mode == "endpoint":
        result = simulate(system, context, tspan, options=opts)
        post = (
            result.context if extract_state is None else extract_state(result.context)
        )
        ef = float(invariant(post))
        abs_drift = abs(ef - e0)
        rel_drift = abs_drift / denom
        if rel_drift > allowed_rel_drift or not np.isfinite(rel_drift):
            raise _violation(ef, abs_drift, rel_drift)
        return e0, ef

    # mode == "max": run the simulation in n_check segments, evaluating the
    # invariant at each boundary and tracking the worst drift seen anywhere
    # along the trajectory (not just at tf).
    t0, tf = float(tspan[0]), float(tspan[1])
    boundaries = np.linspace(t0, tf, int(n_check) + 1)
    seg_ctx = context
    worst_abs = 0.0
    worst_rel = 0.0
    worst_e = e0
    ef = e0
    for i in range(int(n_check)):
        seg_span = (float(boundaries[i]), float(boundaries[i + 1]))
        result = simulate(system, seg_ctx, seg_span, options=opts)
        seg_ctx = result.context
        post = seg_ctx if extract_state is None else extract_state(seg_ctx)
        ef = float(invariant(post))
        abs_drift = abs(ef - e0)
        rel_drift = abs_drift / denom
        if not np.isfinite(rel_drift) or rel_drift > worst_rel:
            worst_rel = rel_drift
            worst_abs = abs_drift
            worst_e = ef
        if not np.isfinite(rel_drift):
            break

    if worst_rel > allowed_rel_drift or not np.isfinite(worst_rel):
        raise _violation(worst_e, worst_abs, worst_rel)
    return e0, ef
