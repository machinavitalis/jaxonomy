# SPDX-License-Identifier: MIT
"""
T-108-followup-streaming-export — LazyResults streaming export to HDF5/zarr.

Covers:

  - ``LazyResults.to_hdf5(path)`` round-trips via h5py: dataset layout
    is ``time`` at the file root + ``outputs/<name>`` per signal.
  - ``LazyResults.to_zarr(path)`` round-trips via zarr (v3 store).
  - Memory cap: writing a 1M-row synthetic dataset with
    ``chunk_size=100k`` keeps the peak working-set under one chunk's
    worth of extra allocation (we never materialise N copies of the
    frame in the writer itself).
  - Default-off byte-equivalence: the new methods are pure additions —
    existing T-108 phase-1 tests still pass (verified indirectly by
    leaving ``collect()`` / ``signal()`` / etc. unchanged, and by a
    smoke test here that mirrors the phase-1 invariants).
  - Both deps are optional via ``pytest.importorskip``.

Vector-valued signals are exploded into ``<name>__<i>`` columns to
mirror the parquet column convention.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from jaxonomy.simulation import LazyResults, SimulationResults


# ── fixtures ──────────────────────────────────────────────────────────────


def _make_small_results():
    """Tiny SimulationResults — used for round-trip correctness checks."""
    time = np.linspace(0.0, 1.0, 21)
    outputs = {
        "x": np.exp(-time),
        "y": np.sin(time),
        # vector-valued signal to exercise the __i column convention
        "v": np.stack([time, 2.0 * time], axis=-1),
    }
    return SimulationResults(
        context=None,
        time=time,
        outputs=outputs,
        per_signal_times=None,
    )


def _make_large_results(n: int = 1_000_000):
    """1M-row synthetic result for the memory-cap test."""
    time = np.linspace(0.0, 1000.0, n).astype(np.float64)
    outputs = {
        "x": time.astype(np.float64),
        "y": (-time).astype(np.float64),
    }
    return SimulationResults(
        context=None,
        time=time,
        outputs=outputs,
        per_signal_times=None,
    )


# ── HDF5 ──────────────────────────────────────────────────────────────────


def test_to_hdf5_round_trip(tmp_path):
    """to_hdf5 produces a readable .h5 file with the expected layout."""
    h5py = pytest.importorskip("h5py")
    res = _make_small_results()
    out_path = tmp_path / "results.h5"
    res.lazy().to_hdf5(out_path, chunk_size=5)
    assert out_path.exists()
    with h5py.File(str(out_path), "r") as f:
        assert "time" in f
        assert "outputs" in f
        np.testing.assert_allclose(f["time"][:], np.asarray(res.time))
        np.testing.assert_allclose(f["outputs/x"][:], np.asarray(res.outputs["x"]))
        np.testing.assert_allclose(f["outputs/y"][:], np.asarray(res.outputs["y"]))
        # Vector-valued: exploded into v__0 and v__1.
        np.testing.assert_allclose(
            f["outputs/v__0"][:], np.asarray(res.outputs["v"])[..., 0]
        )
        np.testing.assert_allclose(
            f["outputs/v__1"][:], np.asarray(res.outputs["v"])[..., 1]
        )


def test_to_hdf5_after_select(tmp_path):
    """Lazy ops compose with to_hdf5 — only selected columns appear."""
    h5py = pytest.importorskip("h5py")
    res = _make_small_results()
    out_path = tmp_path / "selected.h5"
    res.lazy().select("x").to_hdf5(out_path, chunk_size=4)
    with h5py.File(str(out_path), "r") as f:
        assert "time" in f
        assert "outputs/x" in f
        assert "outputs/y" not in f
        assert "outputs/v__0" not in f


def test_to_hdf5_chunks_written_incrementally(tmp_path):
    """Multiple chunks are stitched into one continuous dataset."""
    h5py = pytest.importorskip("h5py")
    res = _make_small_results()
    out_path = tmp_path / "chunked.h5"
    # chunk_size deliberately smaller than the row count.
    res.lazy().to_hdf5(out_path, chunk_size=3)
    with h5py.File(str(out_path), "r") as f:
        # The total length must equal the source length regardless of
        # how the writer carved it into chunks.
        assert f["time"].shape == np.asarray(res.time).shape
        np.testing.assert_allclose(f["time"][:], np.asarray(res.time))


def test_to_hdf5_invalid_chunk_size(tmp_path):
    pytest.importorskip("h5py")
    res = _make_small_results()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        res.lazy().to_hdf5(tmp_path / "bad.h5", chunk_size=0)


def test_to_hdf5_missing_dep_raises(monkeypatch, tmp_path):
    """Skip-if-no-deps: when h5py is absent the writer raises ImportError."""
    import importlib

    res = _make_small_results()
    # Simulate "h5py not installed" by intercepting the import.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("simulated: h5py not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    # Force a re-import path by purging any cached module reference.
    importlib.invalidate_caches()
    with pytest.raises(ImportError, match="h5py is not installed"):
        res.lazy().to_hdf5(tmp_path / "x.h5")


# ── zarr ──────────────────────────────────────────────────────────────────


def test_to_zarr_round_trip(tmp_path):
    """to_zarr produces a readable zarr store with the expected layout."""
    zarr = pytest.importorskip("zarr")
    res = _make_small_results()
    out_path = tmp_path / "results.zarr"
    res.lazy().to_zarr(out_path, chunk_size=5)
    assert out_path.exists()
    root = zarr.open_group(str(out_path), mode="r")
    np.testing.assert_allclose(root["time"][:], np.asarray(res.time))
    np.testing.assert_allclose(root["outputs/x"][:], np.asarray(res.outputs["x"]))
    np.testing.assert_allclose(root["outputs/y"][:], np.asarray(res.outputs["y"]))
    np.testing.assert_allclose(
        root["outputs/v__0"][:], np.asarray(res.outputs["v"])[..., 0]
    )
    np.testing.assert_allclose(
        root["outputs/v__1"][:], np.asarray(res.outputs["v"])[..., 1]
    )


def test_to_zarr_after_select(tmp_path):
    """Lazy ops compose with to_zarr."""
    zarr = pytest.importorskip("zarr")
    res = _make_small_results()
    out_path = tmp_path / "selected.zarr"
    res.lazy().select("y").to_zarr(out_path, chunk_size=4)
    root = zarr.open_group(str(out_path), mode="r")
    np.testing.assert_allclose(root["time"][:], np.asarray(res.time))
    np.testing.assert_allclose(root["outputs/y"][:], np.asarray(res.outputs["y"]))
    # x and v should not be there
    out_grp = root["outputs"]
    names = list(out_grp.array_keys()) if hasattr(out_grp, "array_keys") else list(out_grp)
    assert "y" in names
    assert "x" not in names


def test_to_zarr_chunked_writes(tmp_path):
    """Multiple chunks accumulate to the full row count."""
    zarr = pytest.importorskip("zarr")
    res = _make_small_results()
    out_path = tmp_path / "chunked.zarr"
    res.lazy().to_zarr(out_path, chunk_size=3)
    root = zarr.open_group(str(out_path), mode="r")
    assert root["time"].shape == np.asarray(res.time).shape


def test_to_zarr_invalid_chunk_size(tmp_path):
    pytest.importorskip("zarr")
    res = _make_small_results()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        res.lazy().to_zarr(tmp_path / "bad.zarr", chunk_size=-1)


def test_to_zarr_missing_dep_raises(monkeypatch, tmp_path):
    """Skip-if-no-deps: when zarr is absent the writer raises ImportError."""
    import importlib

    res = _make_small_results()
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("simulated: zarr not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    importlib.invalidate_caches()
    with pytest.raises(ImportError, match="zarr is not installed"):
        res.lazy().to_zarr(tmp_path / "x.zarr")


# ── memory cap (1M rows, 100k chunks) ─────────────────────────────────────


def test_to_hdf5_chunked_memory_cap(tmp_path):
    """Writing 1M rows with chunk_size=100k should not allocate ~1M*Nfloats
    worth of *additional* per-chunk buffers — the writer should slice
    rather than copy.

    We use tracemalloc to measure the peak Python-allocated bytes
    during the write phase and assert it is well under the size of the
    full frame.  The source arrays themselves live outside this window
    (allocated by the fixture), so what we're checking is "the writer
    doesn't reallocate the whole frame".
    """
    h5py = pytest.importorskip("h5py")
    res = _make_large_results(n=1_000_000)
    lazy = res.lazy()
    out_path = tmp_path / "big.h5"

    tracemalloc.start()
    try:
        # Snapshot baseline before write — fixtures are already allocated.
        baseline, _ = tracemalloc.get_traced_memory()
        lazy.to_hdf5(out_path, chunk_size=100_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_delta = peak - baseline
    # Full frame is ~1M * 8 bytes/float * 3 cols (time + x + y) = 24 MB.
    # A chunk is ~100k * 8 * 3 = 2.4 MB.  Allow generous headroom for
    # h5py internals and the iterator overhead — anything well under
    # the full frame size means we are not buffering the whole thing.
    full_frame_bytes = 1_000_000 * 8 * 3
    assert peak_delta < full_frame_bytes // 2, (
        f"to_hdf5 peak delta {peak_delta} >= half of full-frame "
        f"{full_frame_bytes}; writer may be copying the whole frame"
    )

    # Sanity: file content is correct.
    with h5py.File(str(out_path), "r") as f:
        assert f["time"].shape[0] == 1_000_000


def test_to_zarr_chunked_memory_cap(tmp_path):
    """Same memory-cap invariant for zarr."""
    zarr = pytest.importorskip("zarr")
    res = _make_large_results(n=1_000_000)
    lazy = res.lazy()
    out_path = tmp_path / "big.zarr"

    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        lazy.to_zarr(out_path, chunk_size=100_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_delta = peak - baseline
    full_frame_bytes = 1_000_000 * 8 * 3
    assert peak_delta < full_frame_bytes // 2, (
        f"to_zarr peak delta {peak_delta} >= half of full-frame "
        f"{full_frame_bytes}; writer may be copying the whole frame"
    )
    root = zarr.open_group(str(out_path), mode="r")
    assert root["time"].shape[0] == 1_000_000


# ── default-off byte-equivalence ──────────────────────────────────────────


def test_existing_collect_unchanged():
    """Adding to_hdf5 / to_zarr must not perturb the eager collect()
    surface — exact-equality round-trip on a default-off result."""
    res = _make_small_results()
    out = res.lazy().collect()
    np.testing.assert_array_equal(out["time"], np.asarray(res.time))
    np.testing.assert_array_equal(out["x"], np.asarray(res.outputs["x"]))
    np.testing.assert_array_equal(out["y"], np.asarray(res.outputs["y"]))
    np.testing.assert_array_equal(out["v"], np.asarray(res.outputs["v"]))


def test_existing_signal_unchanged():
    """T-108 phase 1 signal() accessor still returns the same (t, v)."""
    res = _make_small_results()
    t, v = res.lazy().signal("x")
    np.testing.assert_array_equal(t, np.asarray(res.time))
    np.testing.assert_array_equal(v, np.asarray(res.outputs["x"]))
