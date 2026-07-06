# SPDX-License-Identifier: MIT

"""Diagram-integrated DPC: policy and plant as jaxonomy ``LeafSystem`` blocks.

The function-level :class:`~jaxonomy.control.dpc.ClosedLoopRollout` integrates
a closed loop with a hand-rolled fixed-step RK4. This module is the
Diagram-integrated upgrade (T-040-followup): the policy is wrapped as a
:class:`PolicyBlock` ``LeafSystem`` whose parameters live in the context (so
``jax.grad`` flows through them), the plant is a :class:`PlantBlock` (or any
user ``LeafSystem`` with a state-output and a control-input port), and the
closed loop runs under the real :func:`jaxonomy.simulate` — the same solver
stack, event handling, and recording that every other jaxonomy model uses.

Why this matters: a DPC policy authored as a block composes with the rest of
the library (other blocks, acausal sub-systems, FMUs, recorders, validation)
and is differentiable end-to-end through ``simulate``'s custom-VJP path. The
worked gradient-correctness check lives in
``test/control/test_t_040_diagram.py``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ...framework import LeafSystem


__all__ = [
    "PolicyBlock",
    "PlantBlock",
    "build_closed_loop",
    "simulate_closed_loop",
]


class PolicyBlock(LeafSystem):
    """A parameterised control policy wrapped as a ``LeafSystem``.

    The policy computes ``u = policy_fn(params, x, ref)`` from a measured
    state ``x`` (input port 0, ``"x"``) and a reference ``ref`` (input port
    1, ``"ref"``), and emits ``u`` on output port 0 (``"u"``).

    ``params`` is stored as a single flat dynamic parameter ``"theta"`` (the
    PyTree is ravelled at construction and reconstructed inside the output
    callback), so arbitrary policy parametrisations — a scalar gain, a gain
    matrix, an MLP weight PyTree — are handled uniformly and remain
    differentiable: ``jax.grad`` of a downstream cost w.r.t. the policy
    parameters flows through ``simulate`` into the context's ``theta`` slot.

    Args:
        policy_fn: ``policy_fn(params, x, ref) -> u`` — JAX-traceable in all
            three arguments. ``params`` has the same PyTree structure as the
            ``params`` passed here; ``x`` / ``ref`` are the port values.
        params: Initial policy parameters (PyTree). Ravelled to the flat
            ``theta`` dynamic parameter; recover the structure inside
            ``policy_fn`` automatically.
        name: Block name (default ``"policy"``).

    Notes:
        To set the parameters on a context, write the *flat* vector into the
        block's ``theta`` parameter::

            theta, _ = jax.flatten_util.ravel_pytree(params)
            subctx = ctx[policy.system_id].with_parameter("theta", theta)
            ctx = ctx.with_subcontext(policy.system_id, subctx)

        :func:`simulate_closed_loop` does this for you.
    """

    def __init__(
        self,
        policy_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        params: Any,
        *,
        name: str = "policy",
    ):
        super().__init__(name=name)
        theta0, unravel = ravel_pytree(params)
        self._unravel = unravel
        self._theta_size = int(theta0.shape[0])

        self.declare_input_port(name="x")
        self.declare_input_port(name="ref")
        self.declare_dynamic_parameter("theta", theta0)

        def _u(time, state, x, ref, **parameters):
            theta = parameters["theta"]
            return policy_fn(unravel(theta), x, ref)

        self.declare_output_port(
            _u,
            name="u",
            requires_inputs=True,
            prerequisites_of_calc=[
                self.input_ports[0].ticket,
                self.input_ports[1].ticket,
            ],
        )

    @staticmethod
    def flatten_params(params: Any) -> jnp.ndarray:
        """Ravel a ``params`` PyTree to the flat ``theta`` vector this block
        stores. Convenience for callers setting ``theta`` on a context."""
        theta, _ = ravel_pytree(params)
        return theta


class PlantBlock(LeafSystem):
    """A continuous-time plant ``dx/dt = plant_ode(t, x, u)`` as a block.

    Convenience wrapper so a bare ODE can be dropped into a closed-loop
    Diagram without hand-writing a ``LeafSystem``. The control ``u`` arrives
    on input port 0 (``"u"``); the state ``x`` is exposed on output port 0
    (``"x"``) via :meth:`declare_continuous_state_output`.

    Args:
        plant_ode: ``plant_ode(time, x, u) -> dx/dt`` — same structure as
            the state. JAX-traceable.
        x0: Initial / default continuous state (sets the state shape).
        name: Block name (default ``"plant"``).
    """

    def __init__(
        self,
        plant_ode: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        x0: jnp.ndarray,
        *,
        name: str = "plant",
    ):
        super().__init__(name=name)
        self.declare_input_port(name="u")

        def _ode(time, state, u, **parameters):
            return plant_ode(time, state.continuous_state, u)

        self.declare_continuous_state(
            default_value=jnp.asarray(x0), ode=_ode, requires_inputs=True
        )
        self.declare_continuous_state_output(name="x")


def build_closed_loop(
    plant: LeafSystem,
    policy: PolicyBlock,
    reference,
    *,
    name: str = "dpc_closed_loop",
):
    """Wire ``plant`` and ``policy`` into a feedback Diagram.

    Connections::

        plant.x   -> policy.x
        reference -> policy.ref
        policy.u  -> plant.u

    Args:
        plant: A ``LeafSystem`` whose output port 0 is the measured state and
            whose input port 0 is the control. :class:`PlantBlock` satisfies
            this; so does any compatible user block.
        policy: A :class:`PolicyBlock`.
        reference: Either a ``LeafSystem`` source block (e.g.
            ``library.Constant`` / a time-varying source) whose output feeds
            ``policy.ref``, or a constant array — in which case a
            ``library.Constant`` source is created automatically.
        name: Diagram name.

    Returns:
        ``(diagram, plant, policy, reference_block)``.
    """
    from ...framework import DiagramBuilder

    builder = DiagramBuilder()
    plant_s = builder.add(plant)
    policy_s = builder.add(policy)

    if isinstance(reference, LeafSystem):
        ref_block = reference
    else:
        from ...library import Constant

        ref_block = Constant(jnp.asarray(reference), name="reference")
    ref_s = builder.add(ref_block)

    builder.connect(plant_s.output_ports[0], policy_s.input_ports[0])  # x
    builder.connect(ref_s.output_ports[0], policy_s.input_ports[1])    # ref
    builder.connect(policy_s.output_ports[0], plant_s.input_ports[0])  # u

    diagram = builder.build(name=name)
    return diagram, plant_s, policy_s, ref_s


def simulate_closed_loop(
    plant: LeafSystem,
    policy: PolicyBlock,
    params: Any,
    reference,
    t_span,
    *,
    dt: float,
    x0=None,
    options=None,
    record: bool = False,
):
    """Run a DPC closed loop under :func:`jaxonomy.simulate`.

    Builds the feedback Diagram (:func:`build_closed_loop`), injects
    ``params`` into the policy's ``theta`` slot, and runs a fixed-step RK4
    simulation. The terminal plant state ``"x_final"`` is always returned
    and is read from the result context, so it is fully differentiable —
    ``jax.grad`` of a cost built on it flows through ``simulate`` into
    ``params``.

    Two mutually-exclusive modes (recording a dense time series is not
    supported under reverse-mode autodiff — the A1 / ``scalar_cost_simulate``
    constraint):

    - ``record=False`` (default): differentiable. Uses an autodiff-enabled
      solver config; ``"time"`` / ``"x"`` / ``"u"`` are ``None`` and only
      ``"x_final"`` is populated. This is the training path.
    - ``record=True``: returns the full ``"time"`` / ``"x"`` / ``"u"``
      trajectories (recorded at the RK4 step cadence) with autodiff off.
      This is the forward-inspection / plotting path.

    Args:
        plant: Plant ``LeafSystem`` (e.g. :class:`PlantBlock`).
        policy: :class:`PolicyBlock`.
        params: Policy parameters (PyTree matching the block's ``params``).
        reference: Constant array or source block (see
            :func:`build_closed_loop`).
        t_span: ``(t0, t1)`` simulation interval.
        dt: Fixed RK4 step (also the recording cadence when ``record``).
        x0: Optional initial plant state override.
        options: Optional :class:`SimulatorOptions`. When ``None`` a
            fixed-step RK4 config is built with ``enable_autodiff`` set to
            ``not record``.
        record: Whether to record the dense ``time`` / ``x`` / ``u``
            trajectories (forces autodiff off when ``options is None``).

    Returns:
        ``dict`` with keys ``"x_final"`` (always) and ``"time"`` /
        ``"x"`` / ``"u"`` (populated only when recording is active).
    """
    from ...simulation import SimulatorOptions, simulate

    diagram, plant_s, policy_s, _ = build_closed_loop(plant, policy, reference)

    ctx = diagram.create_context()

    if x0 is not None:
        sub = ctx[plant_s.system_id].with_continuous_state(jnp.asarray(x0))
        ctx = ctx.with_subcontext(plant_s.system_id, sub)

    theta = PolicyBlock.flatten_params(params)
    psub = ctx[policy_s.system_id].with_parameter("theta", theta)
    ctx = ctx.with_subcontext(policy_s.system_id, psub)

    if options is None:
        opt_kwargs = dict(
            math_backend="jax",
            ode_solver_method="rk4",
            enable_tracing=True,
            enable_autodiff=not record,
            max_major_step_length=dt,
        )
        if record:
            # Pin the minor step to dt so the recorder samples at a
            # predictable cadence (one sample per RK4 step), then size the
            # ring buffer to that count (+margin) so the full trajectory is
            # captured rather than just the tail (B3/B8 buffer-overflow
            # papercut — the recorder samples per minor step, not per major
            # step).
            t0, t1 = float(t_span[0]), float(t_span[1])
            n_steps = int(round((t1 - t0) / dt)) + 2
            opt_kwargs["max_minor_step_size"] = dt
            opt_kwargs["buffer_length"] = max(64, 2 * n_steps)
        options = SimulatorOptions(**opt_kwargs)

    # Recording a dense series is incompatible with reverse-mode autodiff.
    record_traj = record and not options.enable_autodiff
    recorded_signals = (
        {"x": plant_s.output_ports[0], "u": policy_s.output_ports[0]}
        if record_traj
        else None
    )

    res = simulate(
        diagram, ctx, t_span, options=options, recorded_signals=recorded_signals
    )

    x_final = res.context[plant_s.system_id].continuous_state
    return {
        "time": res.time if record_traj else None,
        "x": res.outputs["x"] if record_traj else None,
        "u": res.outputs["u"] if record_traj else None,
        # ``x_final`` is read from the result context and is authoritative. When
        # recording, the last buffered sample ``x[-1]`` is produced by a
        # separate mechanism (the ring buffer) and may differ from ``x_final``
        # at the buffer edge for non-integral ``(t1 - t0) / dt`` — prefer
        # ``x_final`` for the terminal state (e.g. terminal-cost terms).
        "x_final": x_final,
    }


class ClosedLoopRunner:
    """Build a DPC closed loop **once**, then run it **many** times.

    :func:`simulate_closed_loop` rebuilds the feedback ``Diagram`` and its
    ``Context`` on every call, so a training loop that calls it per gradient
    step pays a full rebuild each step.  ``ClosedLoopRunner`` constructs the
    diagram, base context, and solver options once and exposes a thin
    :meth:`run` that only swaps the policy ``theta`` (and optionally ``x0``)
    into the cached context before calling :func:`jaxonomy.simulate`.  This is
    the differentiable training path (``record=False`` semantics): it returns
    the terminal plant state ``x_final``, and ``jax.grad`` of a cost built on
    it flows through ``simulate`` into ``params``.

    Example::

        runner = ClosedLoopRunner(plant, policy, ref, (0.0, 5.0), dt=0.05)
        loss = lambda p: jnp.sum((runner.run(p) - target) ** 2)
        g = jax.grad(loss)(params)                 # one build, many grads
        batched = jax.vmap(runner.run)(params_batch)   # batched rollout

    For dense trajectory recording (plotting / inspection) use
    :func:`simulate_closed_loop` with ``record=True`` — recording a time series
    is incompatible with reverse-mode autodiff (the A1 constraint), so it is
    deliberately not offered here.

    Args:
        plant: Plant ``LeafSystem`` (e.g. :class:`PlantBlock`).
        policy: :class:`PolicyBlock`.
        reference: Constant array or source block (see
            :func:`build_closed_loop`).
        t_span: ``(t0, t1)`` simulation interval.
        dt: Fixed RK4 step.
        x0: Optional default initial plant state (overridable per
            :meth:`run` call).
        options: Optional :class:`SimulatorOptions`; defaults to a fixed-step
            autodiff-enabled RK4 config.
    """

    def __init__(
        self, plant, policy, reference, t_span, *, dt, x0=None, options=None
    ):
        from ...simulation import SimulatorOptions

        self.diagram, self.plant_s, self.policy_s, _ = build_closed_loop(
            plant, policy, reference
        )
        self.t_span = t_span
        base_ctx = self.diagram.create_context()
        if x0 is not None:
            sub = base_ctx[self.plant_s.system_id].with_continuous_state(
                jnp.asarray(x0)
            )
            base_ctx = base_ctx.with_subcontext(self.plant_s.system_id, sub)
        self.base_ctx = base_ctx

        if options is None:
            options = SimulatorOptions(
                math_backend="jax",
                ode_solver_method="rk4",
                enable_tracing=True,
                enable_autodiff=True,
                max_major_step_length=dt,
            )
        self.options = options

    def run(self, params: Any, x0=None):
        """Run the cached closed loop and return the terminal plant state.

        Only ``theta`` (from ``params``) and, optionally, ``x0`` are swapped
        into the pre-built context — no diagram/context rebuild.  Differentiable
        and ``vmap``-able in ``params`` / ``x0``.
        """
        from ...simulation import simulate

        ctx = self.base_ctx
        if x0 is not None:
            sub = ctx[self.plant_s.system_id].with_continuous_state(
                jnp.asarray(x0)
            )
            ctx = ctx.with_subcontext(self.plant_s.system_id, sub)

        theta = PolicyBlock.flatten_params(params)
        psub = ctx[self.policy_s.system_id].with_parameter("theta", theta)
        ctx = ctx.with_subcontext(self.policy_s.system_id, psub)

        res = simulate(self.diagram, ctx, self.t_span, options=self.options)
        return res.context[self.plant_s.system_id].continuous_state
