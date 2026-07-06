# SPDX-License-Identifier: MIT

"""Matplotlib helpers for :class:`~jaxonomy.simulation.types.SimulationResults` and
:class:`~jaxonomy.simulation.batch.BatchSimulationResults`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from jaxonomy.simulation.batch import BatchSimulationResults
    from jaxonomy.simulation.types import SimulationResults

__all__ = [
    "plot_results",
    "plot_batch_results",
    "plot_phase_portrait",
]


def _import_pyplot():
    if "matplotlib" in sys.modules and sys.modules["matplotlib"] is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e
    from matplotlib.figure import Figure

    return plt, Figure


def _to_numpy(x):
    x = np.asarray(x)
    try:
        import jax

        return np.asarray(jax.device_get(x))
    except Exception:
        return x


def _normalize_results_time_series(
    results: Union["SimulationResults", "BatchSimulationResults"],
    signals: Optional[list[str]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (time, {name: y}) for plotting."""
    if hasattr(results, "outputs") and results.outputs is None:
        raise ValueError("results.outputs is None; pass recorded_signals to simulate().")
    outs = results.outputs
    if signals is None:
        names = list(outs.keys())
    else:
        names = list(signals)
        for n in names:
            if n not in outs:
                raise KeyError(f"Signal {n!r} not in results.outputs")
    t = _to_numpy(results.time)
    series = {n: _to_numpy(outs[n]) for n in names}
    return t, series


def plot_results(
    results: "SimulationResults",
    signals: Optional[list[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
    ax=None,
):
    """Plot time series from :class:`~jaxonomy.simulation.types.SimulationResults`."""
    from jaxonomy.simulation.batch import BatchSimulationResults

    if isinstance(results, BatchSimulationResults):
        raise TypeError(
            "Use plot_batch_results() for BatchSimulationResults, not plot_results()."
        )

    plt, _ = _import_pyplot()

    t, series = _normalize_results_time_series(results, signals)
    if not series:
        raise ValueError("No signals to plot.")

    names = list(series.keys())
    if ax is not None and len(names) > 1:
        raise ValueError("ax= may only be used when plotting a single signal.")

    if len(names) == 1:
        name = names[0]
        y = series[name]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 4))
        else:
            fig = ax.figure
        ax.plot(t, y, label=name)
        ax.set_xlabel("time")
        ax.set_ylabel(name)
        ax.legend()
        if title:
            fig.suptitle(title)
    else:
        n = len(names)
        fig, axes = plt.subplots(n, 1, figsize=figsize or (8, 2.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax_i, name in zip(axes, names):
            ax_i.plot(t, series[name], label=name)
            ax_i.set_ylabel(name)
            ax_i.legend(loc="upper right")
        axes[-1].set_xlabel("time")
        if title:
            fig.suptitle(title)
        fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_batch_results(
    results: "BatchSimulationResults",
    signals: Optional[list[str]] = None,
    percentile_bands: tuple[float, float, float, float] = (5, 25, 75, 95),
    show_mean: bool = True,
    show_individual: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot ensemble results with percentile bands."""
    plt, Figure = _import_pyplot()

    p_lo_o, p_lo_i, p_hi_i, p_hi_o = percentile_bands
    t = _to_numpy(results.time)
    if signals is None:
        names = list(results.outputs.keys())
    else:
        names = list(signals)

    if not names:
        raise ValueError("No signals to plot.")

    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=figsize or (8, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        batch = _to_numpy(results.outputs[name])
        if batch.ndim < 2:
            raise ValueError(
                f"Batch output {name!r} must be at least 2-D (N, T, ...); got shape {batch.shape}"
            )
        # (N, T) or (N, T, D) where D >= 1
        if batch.ndim > 2:
            batch = batch.reshape(batch.shape[0], batch.shape[1], -1)
            n_components = batch.shape[2]
        else:
            n_components = 1

        if n_components == 1:
            # Scalar signal — squeeze to (N, T) and plot on single axes
            if batch.ndim == 3:
                batch = batch[:, :, 0]
            mean = _to_numpy(results.mean(name))
            y_o_lo = _to_numpy(results.percentile(name, p_lo_o))
            y_o_hi = _to_numpy(results.percentile(name, p_hi_o))
            y_i_lo = _to_numpy(results.percentile(name, p_lo_i))
            y_i_hi = _to_numpy(results.percentile(name, p_hi_i))
            ax.fill_between(t, y_o_lo, y_o_hi, alpha=0.2, color="C0", label=None)
            ax.fill_between(t, y_i_lo, y_i_hi, alpha=0.35, color="C0", label=None)
            if show_mean:
                ax.plot(t, mean, color="C0", linewidth=2, label="mean")
            if show_individual:
                for i in range(batch.shape[0]):
                    ax.plot(t, batch[i], color="0.7", linewidth=0.5, alpha=0.45)
            ax.set_ylabel(name)
            if show_mean:
                ax.legend(loc="upper right")
        else:
            # Vector signal — plot each component as a separate line on the same axes
            for d in range(n_components):
                comp_batch = batch[:, :, d]  # (N, T)
                mean_d = np.mean(comp_batch, axis=0)
                y_o_lo_d = np.percentile(comp_batch, p_lo_o, axis=0)
                y_o_hi_d = np.percentile(comp_batch, p_hi_o, axis=0)
                y_i_lo_d = np.percentile(comp_batch, p_lo_i, axis=0)
                y_i_hi_d = np.percentile(comp_batch, p_hi_i, axis=0)
                color = f"C{d}"
                ax.fill_between(t, y_o_lo_d, y_o_hi_d, alpha=0.15, color=color, label=None)
                ax.fill_between(t, y_i_lo_d, y_i_hi_d, alpha=0.25, color=color, label=None)
                lbl = f"mean[{d}]" if show_mean else None
                if show_mean:
                    ax.plot(t, mean_d, color=color, linewidth=2, label=lbl)
                if show_individual:
                    for i in range(comp_batch.shape[0]):
                        ax.plot(t, comp_batch[i], color=color, linewidth=0.5, alpha=0.3)
            ax.set_ylabel(name)
            if show_mean:
                ax.legend(loc="upper right")

    axes[-1].set_xlabel("time")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_phase_portrait(
    results: "SimulationResults",
    x_signal: str,
    y_signal: str,
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot ``y_signal`` vs ``x_signal`` (phase plane), not vs time."""
    plt, _ = _import_pyplot()

    if results.outputs is None:
        raise ValueError("results.outputs is None; pass recorded_signals to simulate().")
    for key in (x_signal, y_signal):
        if key not in results.outputs:
            raise KeyError(f"Signal {key!r} not in results.outputs")

    x = _to_numpy(results.outputs[x_signal])
    y = _to_numpy(results.outputs[y_signal])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y)
    ax.set_xlabel(x_signal)
    ax.set_ylabel(y_signal)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return fig
