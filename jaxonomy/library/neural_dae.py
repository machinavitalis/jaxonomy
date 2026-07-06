# SPDX-License-Identifier: MIT

"""Neural correction terms inside acausal DAEs (T-044, NeuralDAEBlock).

A compiled acausal system is a semi-explicit DAE

    M ẋ = f(t, x, p),      0 = g(t, x, p)

with the differential rows ``[0:n_ode]`` (mass-matrix diagonal 1) and the
algebraic constraint rows ``[n_ode:]`` (mass-matrix diagonal 0).  This module
adds a learned correction ``f_NN(t, x; θ)`` to the **differential** rows:

    M ẋ = f(t, x, p) + pad(f_NN(t, x; θ)),      0 = g(t, x, p)

so the constraint structure (and the symbolic Pantelides index reduction that
produced it) is untouched, while ``θ`` becomes a dynamic parameter that
``jax.grad`` / ``optax`` flow into through :func:`jaxonomy.simulate`.  This is
the differentiable-acausal "moat": Modelica can't (no autodiff), Neuromancer
can't (no acausal / Pantelides), and causal-function-first tools can't express
the constraint coupling.

Two entry points, same injection site:

* **Phase 1 — post-hoc** (:func:`add_neural_correction`): re-points a *compiled*
  system's continuous-state RHS in place.  Reach for it when you already hold an
  ``AcausalSystem`` and want to bolt a correction on.

* **Phase 2 — first-class block** (:class:`NeuralDAEBlock`): authored *in the
  diagram* via :meth:`AcausalDiagram.add_neural_correction_block`.  The compiler
  resolves the block's target states to differential rows and injects the
  correction at the same post-index-reduction site, so the block never enters
  the symbolic / Pantelides path.  Targets are written against *physical*
  component states (``(component, "v")``) rather than raw row indices, and ``θ``
  is exposed as the component-style parameter ``f"{block.name}_theta"``.

Phase-2 example
---------------

    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(ev, name="m1", M=1.0, ...)
    # ... connect m1 to a spring / wall ...

    # learn a velocity-proportional drag on the mass's velocity state. nn_fn
    # receives only the targeted states (here ``v``), so no row indices leak in:
    block = NeuralDAEBlock(
        lambda t, v, theta: theta[0] * v,   # v == [m1's velocity]; returns 1 correction
        jnp.zeros(1),
        targets=[(m1, "v")],                # correct the differential row that *is* m1's velocity
        name="drag",
    )
    ad.add_neural_correction_block(block)

    sys = AcausalCompiler(ev, ad)(leaf_backend="jax")   # θ already wired in
    # θ now lives in the context as "drag_theta", differentiable through simulate.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..backend import numpy_api as npa


__all__ = ["add_neural_correction", "NeuralDAEBlock"]


def _resolve_rows(state_rows, n_ode):
    """Normalise ``state_rows`` (or ``None`` → all differential rows) and
    validate every index is a differential row."""
    if state_rows is None:
        return list(range(n_ode))
    rows = [int(r) for r in state_rows]
    for r in rows:
        if not (0 <= r < n_ode):
            raise ValueError(
                f"add_neural_correction: state_rows entry {r} is not a "
                f"differential row — must be in [0, n_ode={n_ode}). The "
                "neural term may only correct differential rows, never the "
                "algebraic constraint rows."
            )
    return rows


def add_neural_correction(
    acausal_system,
    nn_fn: Callable[[Any, Any, Any], Any],
    theta: Any,
    *,
    state_rows: Sequence[int] | None = None,
    param_name: str = "nn_theta",
):
    """Add a neural correction ``f_NN(t, x; θ)`` to a compiled acausal DAE's
    differential rows, **in place**.

    Wraps the system's continuous-state RHS so the differential rows become
    ``f + scatter(nn_fn(t, x_diff; θ), state_rows)`` and declares ``θ`` as a
    dynamic parameter named ``param_name``.  The algebraic constraint rows are
    left exactly as the compiler produced them, so the constraint residual and
    index reduction are unaffected.

    Calling this more than once **composes**: each correction wraps the
    currently-effective RHS, so multiple terms sum.  Distinct ``param_name``
    values are required when composing (the default collides).  This is the
    mechanism :class:`NeuralDAEBlock` rides on; you rarely call it directly more
    than once.

    Must be called on a *compiled* ``AcausalSystem`` (the output of
    ``AcausalCompiler(...)``), **before** ``create_context`` — the new ``θ``
    parameter is read into the context at creation time.

    Args:
        acausal_system: A compiled ``AcausalSystem`` (carries ``n_ode`` /
            ``n_alg`` and the ``_cs_base_ode`` set by the compiler).
        nn_fn: ``(time, x_diff, theta) -> correction`` where ``x_diff`` is the
            length-``n_ode`` differential state and ``correction`` is a vector
            of length ``len(state_rows)`` (default ``n_ode``).  Must be
            JAX-traceable.  For a true neural network, close an MLP over a flat
            ``theta`` (reuse the T-040 ``PolicyBlock.flatten_params`` ravel
            pattern so ``theta`` is a single leaf-flat dynamic parameter).
        theta: Initial parameter value (typically a flat array).  Stored as the
            dynamic parameter's default; differentiate w.r.t. the value injected
            into the context at run time.
        state_rows: Differential-row indices the correction is added to.
            ``None`` (default) targets all ``n_ode`` differential rows in order.
            Every index must be in ``[0, n_ode)`` — the correction may not enter
            the algebraic constraint rows.
        param_name: Context parameter name for ``θ`` (default ``"nn_theta"``).

    Returns:
        The same ``acausal_system`` (mutated), for chaining.

    Raises:
        TypeError: if ``acausal_system`` was not produced by the acausal
            compiler (missing ``_cs_base_ode``).
        ValueError: if any ``state_rows`` index is not a differential row.
    """
    base_ode = getattr(acausal_system, "_cs_base_ode", None)
    if base_ode is None:
        raise TypeError(
            "add_neural_correction: expected a compiled AcausalSystem (from "
            "AcausalCompiler(...)); got an object without the compiled "
            "continuous-state RHS. Did you pass the AcausalDiagram instead of "
            "its compiled output?"
        )

    n_ode = int(acausal_system.n_ode)
    n_alg = int(acausal_system.n_alg)
    n_total = n_ode + n_alg

    rows = _resolve_rows(state_rows, n_ode)

    # Constant 0/1 scatter matrix (n_total x len(rows)) mapping the correction
    # vector onto the selected differential rows.  Built with numpy at
    # wrap-time, then cast to the active backend so the wrapped RHS stays
    # backend-neutral (no jax-specific ``.at[]`` scatter in the hot path).
    scatter_np = np.zeros((n_total, len(rows)))
    for i, r in enumerate(rows):
        scatter_np[r, i] = 1.0
    scatter = npa.asarray(scatter_np)

    acausal_system.declare_dynamic_parameter(param_name, theta)

    # Compose on the *currently-effective* RHS so repeated calls (e.g. several
    # NeuralDAEBlocks) sum rather than clobber.  The first call sees no effective
    # ode yet and falls back to the bare compiled ``_cs_base_ode``, so a single
    # correction is byte-for-byte identical to the phase-1 behaviour.
    prev_ode = getattr(acausal_system, "_cs_effective_ode", None) or base_ode

    def _corrected_ode(time, state, *u, **params):
        base = prev_ode(time, state, *u, **params)
        x_diff = state.continuous_state[:n_ode]
        nn_out = npa.reshape(
            npa.asarray(nn_fn(time, x_diff, params[param_name])), (-1,)
        )
        return base + scatter @ nn_out

    # Swap the continuous-state ODE callback in place.  Re-running
    # ``configure_continuous_state`` would re-touch the output-port cache index
    # (assigned only at finalization), so instead we replace the wrapped
    # callback directly — shape / mass matrix / default value are already
    # configured by the compiler and unchanged here.  ``eval_time_derivatives``
    # already points at ``ode_callback.eval``, which reads ``_callback``
    # dynamically, so swapping it is sufficient.
    acausal_system._cs_effective_ode = _corrected_ode
    acausal_system.ode_callback._callback = acausal_system.wrap_callback(
        _corrected_ode
    )
    return acausal_system


def _resolve_target_row(component, state_name, dp, sed) -> int:
    """Resolve a physical ``(component, state_name)`` target to a differential
    row index in the compiled ``sed.x``.

    Acausal compilation aliases each component's potential/var symbols onto
    compiler-introduced node-potential symbols and then index-reduces, so a
    component's state (e.g. a mass's ``v``) is generally *not* the symbol that
    survives into ``sed.x``.  This follows ``dp.alias_map`` from the component
    symbol to whatever differential symbol it became, then looks that up in
    ``sed.x``.

    Raises ``ValueError`` (with a list of available names / the resolution it
    landed on) if the state does not exist on the component, resolves to an
    *algebraic* state (``sed.y`` — forbidden, would touch a constraint row), or
    was eliminated / held constant by index reduction.
    """
    name = getattr(component, "name", "<unnamed>")
    matches = [s for s in component.syms if s.sym_name == state_name]
    if not matches:
        avail = sorted({s.sym_name for s in component.syms})
        raise ValueError(
            f"NeuralDAEBlock target ({name!r}, {state_name!r}): component has no "
            f"state symbol named {state_name!r}. Available symbols: {avail}."
        )

    cur = matches[0].s
    seen = set()
    while cur in dp.alias_map and cur not in seen:
        seen.add(cur)
        cur = dp.alias_map[cur]

    if cur in sed.x:
        return sed.x.index(cur)
    if cur in sed.y:
        raise ValueError(
            f"NeuralDAEBlock target ({name!r}, {state_name!r}): resolves to the "
            f"algebraic state {cur} (a constraint row), but the neural "
            "correction may only target *differential* rows. Index reduction "
            "made this state algebraic — pick a differential state of the "
            f"component instead (compiled differential states: {list(sed.x)})."
        )
    raise ValueError(
        f"NeuralDAEBlock target ({name!r}, {state_name!r}): did not resolve to a "
        f"free differential state (resolved to {cur}). It was likely eliminated "
        "or held constant by index reduction (e.g. a fixed/grounded state). "
        f"Compiled differential states: {list(sed.x)}."
    )


class NeuralDAEBlock:
    """A learned correction term ``f_NN(t, x; θ)`` authored directly in an
    :class:`~jaxonomy.acausal.AcausalDiagram` (T-044 phase 2).

    Register the block on a diagram with
    :meth:`~jaxonomy.acausal.AcausalDiagram.add_neural_correction_block`; the
    :class:`~jaxonomy.acausal.AcausalCompiler` then resolves its target states
    to differential rows of the compiled DAE and injects ``f_NN`` at the same
    post-index-reduction RHS site as :func:`add_neural_correction`.  The block
    is **not** an acausal component you ``connect()`` — it carries no equations
    and never enters diagram processing or Pantelides index reduction, which is
    exactly what keeps the (non-symbolically-differentiable) neural term out of
    the symbolic pipeline.

    The contract is **gather-in / scatter-out**: ``nn_fn`` receives only the
    *targeted* differential states (in ``targets`` / ``state_rows`` order) and
    returns one correction per target, which is added back to those rows.  So
    you write the block against your own physical states without ever knowing
    the compiler's row ordering — ``targets=[(mass, "v")]`` means ``nn_fn``'s
    second argument is ``[v]``.  (The lower-level :func:`add_neural_correction`
    keeps the full-state contract for power users who need to read other states.)

    Args:
        nn_fn: ``(time, x_targets, theta) -> correction``.  ``x_targets`` is the
            vector of targeted differential states, in ``targets`` order
            (length ``len(targets)``, or ``n_ode`` when neither ``targets`` nor
            ``state_rows`` is given — then it is the full differential state).
            ``correction`` must have the same length as ``x_targets``.  Must be
            JAX-traceable.
        theta: Initial parameter value (typically a flat array; ravel an MLP's
            params into one leaf so ``θ`` is a single dynamic parameter).
        targets: List of ``(component, state_name)`` pairs naming the *physical*
            differential states to correct, e.g. ``[(mass, "v")]``.  The
            compiler maps each to a row of ``sed.x``.  Mutually exclusive with
            ``state_rows``.
        state_rows: Explicit differential-row indices, for callers who already
            know the compiled layout.  Mutually exclusive with ``targets``.
            ``None`` for both means "all differential rows".
        name: Block name; the learned parameter is exposed in the context as
            ``f"{name}_theta"``.  Must be unique across a diagram's blocks.

    Note:
        Authoring against physical states via ``targets`` is the recommended
        path; it gives a clear error if a chosen state turned out *algebraic*
        after index reduction (a constraint row the correction may not touch),
        listing the compiled differential states so you can pick a valid one.
    """

    def __init__(
        self,
        nn_fn: Callable[[Any, Any, Any], Any],
        theta: Any,
        *,
        targets: Sequence[tuple] | None = None,
        state_rows: Sequence[int] | None = None,
        name: str = "neural_dae",
    ):
        if targets is not None and state_rows is not None:
            raise ValueError(
                "NeuralDAEBlock: pass either `targets` (physical states) or "
                "`state_rows` (raw indices), not both."
            )
        if not name:
            raise ValueError("NeuralDAEBlock: `name` must be a non-empty string.")
        self.nn_fn = nn_fn
        self.theta = theta
        self.targets = list(targets) if targets is not None else None
        self.state_rows = list(state_rows) if state_rows is not None else None
        self.name = name

    @property
    def param_name(self) -> str:
        """Context parameter key for this block's ``θ``."""
        return f"{self.name}_theta"

    def _rows(self, dp, sed):
        """Resolve this block's targets to a list of differential-row indices,
        or ``None`` to mean "all differential rows"."""
        if self.targets is not None:
            return [_resolve_target_row(c, s, dp, sed) for (c, s) in self.targets]
        return self.state_rows


def apply_neural_blocks(acausal_system, diagram, dp, sed):
    """Inject every :class:`NeuralDAEBlock` registered on ``diagram`` into the
    compiled ``acausal_system``.

    Called by :class:`~jaxonomy.acausal.AcausalCompiler` after the system is
    built (so ``_cs_base_ode`` exists) and ``sed`` is available for target
    resolution.  A no-op for diagrams without blocks, so existing models are
    unaffected.  Block names must be unique (they key the ``θ`` parameters).
    """
    blocks = getattr(diagram, "neural_blocks", None) or []
    seen = set()
    n_ode = int(acausal_system.n_ode)
    for block in blocks:
        if block.name in seen:
            raise ValueError(
                f"Duplicate NeuralDAEBlock name {block.name!r}; block names must "
                "be unique within a diagram (each names a `θ` parameter)."
            )
        seen.add(block.name)

        rows = block._rows(dp, sed)
        rows = list(range(n_ode)) if rows is None else [int(r) for r in rows]

        # Gather-in: hand ``nn_fn`` only the targeted differential states (in
        # target order) via a constant 0/1 gather matrix, so the block is
        # written against its own physical states, not compiler row indices.
        # ``add_neural_correction`` then scatters the output back to ``rows``.
        gather_np = np.zeros((len(rows), n_ode))
        for i, r in enumerate(rows):
            gather_np[i, r] = 1.0
        gather = npa.asarray(gather_np)
        nn_fn = block.nn_fn

        def _gathered_fn(time, x_diff, theta, _gather=gather, _fn=nn_fn):
            return _fn(time, _gather @ x_diff, theta)

        add_neural_correction(
            acausal_system,
            _gathered_fn,
            block.theta,
            state_rows=rows,
            param_name=block.param_name,
        )
    return acausal_system
