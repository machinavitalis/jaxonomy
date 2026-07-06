# SPDX-License-Identifier: MIT

"""Variant-axis sweep convenience over :func:`simulate_batch`.

A diagram built with :class:`~jaxonomy.framework.variants.Variant` nodes
encodes a discrete choice — each variant configuration produces a
structurally different concrete diagram (different pytree shape, possibly
different state size). :func:`simulate_batch` vectorises *only* over
parameter-value batches keyed by dot-path; it cannot collapse the variant
axis because the pytree shape must be stable across vmap replicas.

The canonical workaround is a Python loop over
:func:`iter_variant_configurations` calling :func:`simulate_batch` once per
configuration. :func:`simulate_variant_sweep` packages that pattern as a
single call so tutorial authors and downstream UQ helpers don't reinvent it.

The returned structure mirrors the input: a dict keyed by the variant
configuration (a hashable tuple of ``(path, choice)`` pairs) whose values
are the per-variant :class:`BatchSimulationResults` (or
:class:`SimulationResults` when ``param_batches`` is ``None``).

Why this is *not* a vmap-over-variants:
    The pytree shape of the simulator state changes when a variant picks
    a different subdiagram (different block list, different dynamic-
    state arity). ``jax.vmap`` requires identical pytree structure across
    the batched axis, so the variant axis is genuinely orthogonal to the
    parameter-batch axis. The per-variant kernel JIT cost is paid once
    per configuration; the parameter-batch axis is still vectorised
    inside each configuration.
"""

from __future__ import annotations

from typing import Any, Callable

from ..framework.diagram import Diagram
from ..framework.variants import iter_variant_configurations
from ..framework.port import OutputPort
from .batch import BatchSimulationResults, simulate_batch
from .simulator import simulate
from .types import ResultsOptions, SimulationResults, SimulatorOptions

__all__ = ["simulate_variant_sweep"]


def _config_key(config: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Deterministic, hashable variant-configuration key."""
    return tuple(sorted(config.items()))


def simulate_variant_sweep(
    diagram: Diagram,
    t_span: tuple[float, float],
    *,
    param_batches: dict[str, Any] | None = None,
    options: SimulatorOptions | None = None,
    recorded_signals: Callable[[Diagram], dict[str, OutputPort]] | dict[str, OutputPort] | None = None,
    results_options: ResultsOptions | None = None,
    use_vmap: bool = False,
) -> dict[tuple[tuple[str, Any], ...], BatchSimulationResults | SimulationResults]:
    """Sweep every variant configuration of ``diagram``; for each, optionally
    sweep a parameter batch.

    Args:
        diagram: A built :class:`Diagram` containing one or more
            :class:`~jaxonomy.framework.variants.Variant` nodes.
        t_span: ``(t_start, t_stop)`` forwarded to the per-variant call.
        param_batches: Optional dot-path → ``(N,)``-shaped array dict
            forwarded to :func:`simulate_batch` once per variant. When
            ``None``, a single :func:`simulate` is run per variant
            (equivalent to ``N=1`` but without the batch axis).
        options: :class:`SimulatorOptions` forwarded per variant.
        recorded_signals: Either a static ``{name: OutputPort}`` dict —
            in which case the port references must be valid on every
            generated per-variant diagram — or a callable
            ``recorded_signals(variant_diagram) -> {name: OutputPort}``
            that re-resolves the ports against the concrete diagram. The
            callable form is the safer choice when variants substitute
            entire subdiagrams.
        results_options: Forwarded to :func:`simulate_batch`.
        use_vmap: Forwarded to :func:`simulate_batch` when
            ``param_batches`` is supplied; ignored otherwise.

    Returns:
        Dict keyed by the variant configuration (a sorted tuple of
        ``(path, choice)`` pairs) whose values are
        :class:`BatchSimulationResults` (when ``param_batches`` is set)
        or :class:`SimulationResults` (when it isn't).

    Example:
        .. code-block:: python

            results = simulate_variant_sweep(
                diagram,
                t_span=(0.0, 1.0),
                param_batches={"plant.gain": jnp.linspace(0.5, 2.0, 8)},
                recorded_signals=lambda diag: {
                    "y": diag["plant"].output_ports[0],
                },
                options=opts,
            )
            for cfg, batch_results in results.items():
                print(dict(cfg), batch_results.outputs["y"].shape)

    Notes:
        Each variant configuration triggers an independent JIT compile of
        the simulator. For a sweep over ``V`` variants and ``N`` parameter
        batches the cost is ``V`` compiles + ``V * N`` simulations (with
        the parameter axis vectorised inside each variant). Variant-axis
        vmap is genuinely not possible because the pytree shape is not
        stable across configurations — see the module docstring.
    """
    if not isinstance(diagram, Diagram):
        raise TypeError(
            f"simulate_variant_sweep expects a Diagram, got {type(diagram)}"
        )

    out: dict[tuple[tuple[str, Any], ...], Any] = {}
    for config, variant_diagram in iter_variant_configurations(diagram):
        # Resolve recorded_signals against the concrete per-variant diagram.
        if callable(recorded_signals):
            rs_for_variant = recorded_signals(variant_diagram)
        else:
            rs_for_variant = recorded_signals

        if param_batches is None:
            ctx = variant_diagram.create_context()
            res = simulate(
                variant_diagram,
                ctx,
                t_span,
                options=options,
                recorded_signals=rs_for_variant,
                results_options=results_options,
            )
        else:
            res = simulate_batch(
                variant_diagram,
                t_span,
                param_batches=param_batches,
                options=options,
                recorded_signals=rs_for_variant,
                results_options=results_options,
                use_vmap=use_vmap,
            )
        out[_config_key(config)] = res
    return out
