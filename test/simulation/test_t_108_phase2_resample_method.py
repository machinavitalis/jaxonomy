# SPDX-License-Identifier: MIT

"""T-108 phase 2 — lazy resample with T-106 interpolation backend.

Phase 2 extends :meth:`LazyResults.resample` to accept a ``method=``
kwarg that routes per-channel interpolation through the T-106 backend
(``jaxonomy.library.lookup_table.interp_1d``).  ``method="linear"``
keeps the existing fast paths (``np.interp`` eager + native polars
asof-join + linear-interp pushdown) for byte-equivalence with phase 1;
non-linear methods materialise the upstream chain and route per-channel
through ``interp_1d``.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxonomy.library.lookup_table import interp_1d
from jaxonomy.simulation.lazy_results import LazyResults


def _make_lazy(t, outputs):
    """Build a LazyResults from a (time, {signal: array}) pair via the
    dataclass constructor — avoids needing a full SimulationResults
    fixture (which itself requires a built diagram + simulation)."""
    return LazyResults(
        _outputs={k: np.asarray(v) for k, v in outputs.items()},
        _time=np.asarray(t),
    )


# ---------------------------------------------------------------------------
# Default (method="linear") byte-equivalence with phase 1.
# ---------------------------------------------------------------------------


def test_resample_default_method_is_linear_and_matches_np_interp():
    t = np.linspace(0.0, 10.0, 11)
    y = np.sin(t)
    lazy = _make_lazy(t, {"y": y})

    t_new = np.linspace(0.5, 9.5, 19)
    out_default = lazy.resample(t_new).collect()
    out_explicit = lazy.resample(t_new, method="linear").collect()

    np.testing.assert_array_equal(out_default["y"], out_explicit["y"])
    np.testing.assert_allclose(out_default["y"], np.interp(t_new, t, y))


# ---------------------------------------------------------------------------
# Non-linear methods route through interp_1d.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pchip", "akima", "cubic", "nearest", "flat"])
def test_resample_nonlinear_method_matches_interp_1d(method):
    t = np.linspace(0.0, 10.0, 21)
    y = np.sin(t)
    lazy = _make_lazy(t, {"y": y})

    t_new = np.linspace(0.5, 9.5, 37)
    out = lazy.resample(t_new, method=method).collect()

    expected = np.asarray(interp_1d(t_new, t, y, method=method))
    np.testing.assert_allclose(out["y"], expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(out["time"], t_new)


def test_resample_pchip_is_smoother_than_linear():
    """On a smooth signal sampled coarsely then over-resampled, PCHIP
    error vs the analytical truth should be smaller than linear error
    (sanity-check that the new method actually does something)."""
    t = np.linspace(0.0, 2.0 * np.pi, 17)  # coarse
    y = np.sin(t)
    lazy = _make_lazy(t, {"y": y})

    t_new = np.linspace(0.1, 2.0 * np.pi - 0.1, 401)
    truth = np.sin(t_new)

    y_linear = lazy.resample(t_new, method="linear").collect()
    y_pchip = lazy.resample(t_new, method="pchip").collect()

    linear_err = float(np.max(np.abs(y_linear["y"] - truth)))
    pchip_err = float(np.max(np.abs(y_pchip["y"] - truth)))
    assert pchip_err < linear_err, (
        f"expected PCHIP ({pchip_err:.4g}) to beat linear "
        f"({linear_err:.4g}) on a coarsely-sampled sine"
    )


# ---------------------------------------------------------------------------
# Vector-valued (multi-channel) signals interpolate per-channel.
# ---------------------------------------------------------------------------


def test_resample_method_applied_to_vector_signals_per_channel():
    t = np.linspace(0.0, 10.0, 11)
    y = np.stack([np.sin(t), np.cos(t), t ** 2], axis=-1)  # shape (11, 3)
    lazy = _make_lazy(t, {"y": y})

    t_new = np.linspace(0.5, 9.5, 19)
    out = lazy.resample(t_new, method="pchip").collect()

    assert out["y"].shape == (19, 3)
    # Each channel matches the per-channel interp_1d call.
    for i in range(3):
        expected = np.asarray(interp_1d(t_new, t, y[:, i], method="pchip"))
        np.testing.assert_allclose(out["y"][:, i], expected, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Composability with the rest of the lazy chain.
# ---------------------------------------------------------------------------


def test_resample_pchip_after_select():
    t = np.linspace(0.0, 10.0, 11)
    lazy = _make_lazy(t, {"y": np.sin(t), "z": np.cos(t)})

    t_new = np.linspace(0.5, 9.5, 13)
    out = lazy.select("y").resample(t_new, method="pchip").collect()

    # collect() returns {"time": ..., "y": ...} — select dropped "z".
    assert set(out.keys()) == {"time", "y"}
    expected = np.asarray(interp_1d(t_new, t, np.sin(t), method="pchip"))
    np.testing.assert_allclose(out["y"], expected, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Polars backend behaviour for non-linear methods.
# ---------------------------------------------------------------------------


def test_polars_backend_handles_nonlinear_resample_via_materialisation():
    """Non-linear methods aren't expressible as a single polars
    expression; the backend must materialise + re-promote and still
    produce the same result as the eager path."""
    pl = pytest.importorskip("polars")

    t = np.linspace(0.0, 10.0, 21)
    y = np.sin(t)
    lazy = _make_lazy(t, {"y": y})

    t_new = np.linspace(0.5, 9.5, 25)
    eager_out = lazy.resample(t_new, method="pchip").collect()
    # to_polars goes through the polars _Op chain.
    df = lazy.resample(t_new, method="pchip").to_polars()

    np.testing.assert_allclose(
        df["y"].to_numpy(), eager_out["y"], rtol=1e-9, atol=1e-12
    )
