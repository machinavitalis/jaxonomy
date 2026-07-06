# SPDX-License-Identifier: MIT

"""Static-parameter sweep helper for diagrams that must be rebuilt per-element.

``simulate_batch`` (in ``batch.py``) only handles **dynamic** parameters: those
exposed via the context tree and patchable through
:meth:`Diagram.with_parameters` or ``_pure_patch_context``.  Parameters declared
*static* (e.g. :class:`TransferFunction.num` / ``den``) are baked into the
:class:`LeafSystem` at construction time — sweeping them requires rebuilding the
diagram from scratch for every grid point.

This module ships the "helper" form of T-028: a thin Python loop that, given a
user-supplied ``diagram_factory(**static_params)``, builds a fresh diagram for
each grid point, calls :func:`simulate`, and stacks results into the same
:class:`BatchSimulationResults` shape that ``simulate_batch`` returns.

The harder "lift the restriction" form — promoting static params to dynamic ones
and threading them through every block's evaluation path — is left for future
work (see T-029, T-030).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Sequence

import jax.numpy as jnp

from ..framework.diagram import Diagram
from .batch import BatchSimulationResults, _interp_on_time
from .errors import remap_simulation_errors
from .simulator import simulate
from .types import ResultsOptions, SimulatorOptions

__all__ = ["simulate_static_sweep"]


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def _expand_grid(
    static_param_grid: dict[str, Sequence[Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Expand a name->values grid into a list of per-element kwarg dicts.

    Args:
        static_param_grid: Mapping ``param_name -> sequence of values``.
        mode: ``"zip"`` (default) requires every list to have the same length;
            ``"product"`` takes the cartesian product.

    Returns:
        List of ``{name: value}`` dicts, one per grid element.

    Raises:
        ValueError: empty grid, or ``mode="zip"`` with mismatched lengths,
            or unknown mode.
    """
    if not static_param_grid:
        raise ValueError(
            "simulate_static_sweep: static_param_grid must be a non-empty dict "
            "mapping parameter names to sequences of values."
        )

    keys = list(static_param_grid.keys())
    values = [list(static_param_grid[k]) for k in keys]

    for k, v in zip(keys, values):
        if len(v) == 0:
            raise ValueError(
                f"simulate_static_sweep: static_param_grid[{k!r}] is empty; "
                "every parameter must have at least one value."
            )

    if mode == "zip":
        n0 = len(values[0])
        for k, v in zip(keys, values):
            if len(v) != n0:
                raise ValueError(
                    f"simulate_static_sweep: mode='zip' requires every parameter "
                    f"list to have the same length; got {len(v)} for {k!r} but "
                    f"{n0} for {keys[0]!r}. Use mode='product' for the cartesian "
                    "product."
                )
        return [{k: values[i][j] for i, k in enumerate(keys)} for j in range(n0)]

    if mode == "product":
        import itertools
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

    raise ValueError(
        f"simulate_static_sweep: unknown mode {mode!r}; expected 'zip' or 'product'."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@remap_simulation_errors
def simulate_static_sweep(
    diagram_factory: Callable[..., Diagram],
    t_span: tuple[float, float],
    static_param_grid: dict[str, Sequence[Any]],
    options: SimulatorOptions,
    recorded_signals_factory: Callable[[Diagram], dict],
    results_options: ResultsOptions | None = None,
    mode: str = "zip",
) -> BatchSimulationResults:
    """Sweep over **static** parameters by rebuilding the diagram per element.

    Unlike :func:`simulate_batch`, which patches a single diagram's context with
    different dynamic parameter values, this helper accepts a factory that
    produces a *fresh* :class:`Diagram` for each combination of static param
    values.  Each element is simulated independently in a Python loop; outputs
    are stacked into a :class:`BatchSimulationResults`-shaped struct.

    Because each diagram is fresh, port references are also per-diagram;
    ``recorded_signals_factory`` is invoked with the freshly-built diagram and
    must return the same kind of ``{name: OutputPort}`` dict that
    :func:`simulate` accepts.

    No ``vmap`` or shared JIT cache: static parameters change the diagram's
    structure (e.g. state-space dimensions of a :class:`TransferFunction`) and
    cannot compose with ``jax.vmap`` by definition. Each element pays a JIT
    compilation cost.

    Args:
        diagram_factory: Callable taking the static-param keyword arguments
            specified by ``static_param_grid`` and returning a :class:`Diagram`.
        t_span: ``(t_start, t_stop)`` — same for every element.
        static_param_grid: Mapping parameter name -> sequence of values. Every
            list must have the same length under ``mode="zip"``; or any
            lengths under ``mode="product"`` (cartesian product).
        options: :class:`SimulatorOptions`. ``max_major_steps`` must be set if
            using ``math_backend="jax"``.
        recorded_signals_factory: Callable ``(diagram) -> {name: OutputPort}``.
            Invoked once per grid element with the freshly-built diagram.
        results_options: Optional :class:`ResultsOptions` passed to
            :func:`simulate`.
        mode: ``"zip"`` (default — pair lists element-wise) or ``"product"``
            (cartesian product over all keys).

    Returns:
        :class:`BatchSimulationResults` with ``outputs[name].shape == (N, T)``
        and ``time.shape == (T,)`` where ``N`` is the number of grid elements
        and ``T`` is the time-vector length of the first run (other runs are
        linearly interpolated onto this grid). The ``contexts`` attribute is
        attached to the returned object as a list of per-element final
        contexts.

    Raises:
        ValueError: empty grid, mismatched zip lengths, unknown mode, or
            missing required options.
        TypeError: ``diagram_factory`` did not return a :class:`Diagram`.
    """
    if options is None:
        raise ValueError("simulate_static_sweep requires a SimulatorOptions.")
    if not callable(diagram_factory):
        raise TypeError(
            f"simulate_static_sweep: diagram_factory must be callable, "
            f"got {type(diagram_factory)}"
        )
    if not callable(recorded_signals_factory):
        raise TypeError(
            "simulate_static_sweep: recorded_signals_factory must be callable "
            f"((diagram) -> dict), got {type(recorded_signals_factory)}"
        )

    grid = _expand_grid(static_param_grid, mode)
    n = len(grid)

    time_ref = None
    out_lists: dict[str, list] = {}
    contexts: list = []
    signal_names: list[str] | None = None

    for i, combo in enumerate(grid):
        d = diagram_factory(**combo)
        if not isinstance(d, Diagram):
            raise TypeError(
                f"simulate_static_sweep: diagram_factory(**{combo!r}) returned "
                f"{type(d)}, expected a Diagram."
            )
        sig = recorded_signals_factory(d)
        if not isinstance(sig, dict) or not sig:
            raise ValueError(
                "simulate_static_sweep: recorded_signals_factory must return a "
                f"non-empty dict of {{name: OutputPort}}; got {type(sig)} for "
                f"combo {combo!r}."
            )
        if signal_names is None:
            signal_names = list(sig.keys())
            out_lists = {k: [] for k in signal_names}
        elif list(sig.keys()) != signal_names:
            raise ValueError(
                f"simulate_static_sweep: recorded_signals_factory returned "
                f"different signal names at element {i} ({list(sig.keys())!r}) "
                f"vs element 0 ({signal_names!r})."
            )

        ctx = d.create_context()
        res = simulate(
            d,
            ctx,
            t_span,
            options=options,
            results_options=results_options,
            recorded_signals=sig,
        )
        if res.outputs is None:
            raise RuntimeError(
                f"simulate_static_sweep: simulate returned no outputs at "
                f"element {i} (combo={combo!r})."
            )
        contexts.append(res.context)

        if time_ref is None:
            time_ref = res.time
            for name in signal_names:
                out_lists[name].append(res.outputs[name])
        else:
            t_ref_end = float(time_ref[-1])
            t_run_end = float(res.time[-1])
            if abs(t_run_end - t_ref_end) / max(abs(t_ref_end), 1e-10) > 0.01:
                warnings.warn(
                    f"simulate_static_sweep: run {i} ended at t={t_run_end:.4g} "
                    f"but reference run ended at t={t_ref_end:.4g}. Outputs "
                    "will be interpolated (clamped) to fill the time grid.",
                    UserWarning,
                    stacklevel=2,
                )
            # Trim to the shorter time grid if lengths differ.
            t_ref_arr = jnp.asarray(time_ref)
            t_run_arr = jnp.asarray(res.time)
            if t_run_arr.shape[0] < t_ref_arr.shape[0]:
                # Shrink the reference grid and re-trim previously-collected
                # signals so all rows share the same length.
                new_len = int(t_run_arr.shape[0])
                time_ref = t_run_arr
                for name in signal_names:
                    out_lists[name] = [
                        jnp.asarray(y)[:new_len] for y in out_lists[name]
                    ]
                for name in signal_names:
                    out_lists[name].append(jnp.asarray(res.outputs[name])[:new_len])
            else:
                for name in signal_names:
                    out_lists[name].append(
                        _interp_on_time(res.outputs[name], res.time, time_ref)
                    )

    stacked = {k: jnp.stack(vs, axis=0) for k, vs in out_lists.items()}
    result = BatchSimulationResults(time=time_ref, outputs=stacked, used_vmap=False)
    # Attach per-element final contexts as a public attribute (dataclass with
    # default_factory would require redefining BatchSimulationResults; a simple
    # post-hoc attribute is sufficient and documented in the docstring).
    object.__setattr__(result, "contexts", contexts)
    return result
