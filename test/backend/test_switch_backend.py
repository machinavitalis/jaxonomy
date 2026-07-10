# SPDX-License-Identifier: MIT

import warnings

import pytest
import numpy as np
import jax.numpy as jnp

from jaxonomy.backend import numpy_api as npa
from jaxonomy.backend import get_dispatcher, set_backend as _raw_set_backend
from jaxonomy.backend.backend import MathDispatcher
from jaxonomy.testing import set_backend

try:
    import torch
except ImportError:
    warnings.warn("torch not installed - skipping relevant checks")
    torch = None

# The torch backend is opt-in: MathDispatcher only registers it when
# JAXONOMY_BACKEND=torch is set (see backend.py). Having torch importable is
# not enough — set_backend("torch") raises KeyError unless it was requested.
if torch is not None and "torch" not in MathDispatcher._backends:
    warnings.warn(
        "torch installed but torch backend not registered "
        "(set JAXONOMY_BACKEND=torch to exercise it) - skipping torch branches"
    )
    torch = None

float_dtypes = ["float64", "float32", "float16"]
int_dtypes = ["int64", "int32", "int16"]


@pytest.fixture(autouse=True)
def _restore_backend():
    """These tests mutate the process-global backend dispatcher. Restore the
    entry backend afterwards so a mid-test failure (or a torch-installed run,
    whose tests end on set_backend("torch")) cannot leak backend state into
    the rest of the suite. That leak was the reason this module was skipped
    (the old "see PR 6523" quarantine)."""
    before = get_dispatcher().active_backend
    yield
    _raw_set_backend(before)


def test_switch_backend():
    set_backend("numpy")
    x = npa.array([0.0, 1.0])
    sin_x = npa.sin(x)
    assert isinstance(sin_x, np.ndarray)
    assert not isinstance(sin_x, jnp.ndarray)
    assert torch is None or not isinstance(sin_x, torch.Tensor)

    set_backend("jax")
    x = npa.array([0.0, 1.0])
    sin_x = npa.sin(x)
    assert not isinstance(sin_x, np.ndarray)
    assert isinstance(sin_x, jnp.ndarray)
    assert torch is None or not isinstance(sin_x, torch.Tensor)

    if torch is not None:
        set_backend("torch")
        x = npa.array([0.0, 1.0])
        sin_x = npa.sin(x)
        assert not isinstance(sin_x, np.ndarray)
        assert not isinstance(sin_x, jnp.ndarray)
        assert isinstance(sin_x, torch.Tensor)


@pytest.mark.parametrize("dtype_str", [*float_dtypes, *int_dtypes])
def test_array(dtype_str):
    x = [1, 2, 3]

    set_backend("numpy")
    dtype = getattr(npa, dtype_str)
    y = npa.array(x, dtype=dtype)
    assert isinstance(y, np.ndarray)
    assert y.dtype == dtype
    assert y.shape == (3,)
    assert np.allclose(y, x)

    set_backend("jax")
    dtype = getattr(npa, dtype_str)
    y = npa.array(x, dtype=dtype)
    assert isinstance(y, jnp.ndarray)
    assert y.dtype == dtype
    assert y.shape == (3,)
    assert np.allclose(y, x)

    if torch is not None:
        set_backend("torch")
        dtype = getattr(npa, dtype_str)
        y = npa.array(x, dtype=dtype)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == dtype
        assert y.shape == (3,)
        assert np.allclose(y, x)


@pytest.mark.parametrize("dtype_str", [*float_dtypes, *int_dtypes])
def test_zeros_like_vec(dtype_str):
    set_backend("numpy")
    dtype = getattr(npa, dtype_str)
    x = npa.array([1, 2, 3], dtype=dtype)
    z = npa.zeros_like(x)
    assert isinstance(z, np.ndarray)
    assert z.dtype == dtype
    assert z.shape == (3,)
    assert np.all(z == 0.0)

    set_backend("jax")
    dtype = getattr(npa, dtype_str)
    x = npa.array([1, 2, 3], dtype=dtype)
    z = npa.zeros_like(x)
    assert isinstance(z, jnp.ndarray)
    assert z.dtype == dtype
    assert z.shape == (3,)
    assert np.all(z == 0.0)

    if torch is not None:
        set_backend("torch")
        dtype = getattr(npa, dtype_str)
        x = npa.array([1, 2, 3], dtype=dtype)
        z = npa.zeros_like(x)
        assert isinstance(z, torch.Tensor)
        assert z.dtype == dtype
        assert z.shape == (3,)


@pytest.mark.parametrize("dtype_str", [*float_dtypes, *int_dtypes])
def test_zeros_like_array(dtype_str):
    set_backend("numpy")
    dtype = getattr(npa, dtype_str)
    x = npa.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    z = npa.zeros_like(x)
    assert isinstance(z, np.ndarray)
    assert z.dtype == dtype
    assert z.shape == (2, 3)
    assert np.all(z == 0.0)

    set_backend("jax")
    dtype = getattr(npa, dtype_str)
    x = npa.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    z = npa.zeros_like(x)
    assert isinstance(z, jnp.ndarray)
    assert z.dtype == dtype
    assert z.shape == (2, 3)
    assert np.all(z == 0.0)

    if torch is not None:
        set_backend("torch")
        dtype = getattr(npa, dtype_str)
        x = npa.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        z = npa.zeros_like(x)
        assert isinstance(z, torch.Tensor)
        assert z.dtype == dtype
        assert z.shape == (2, 3)
        assert torch.all(z == 0.0)


def test_reshape():
    set_backend("numpy")
    x = npa.array([[1, 2, 3], [4, 5, 6]])
    y = npa.reshape(x, (3, 2))
    assert isinstance(y, np.ndarray)
    assert y.shape == (3, 2)

    set_backend("jax")
    x = npa.array([[1, 2, 3], [4, 5, 6]])
    y = npa.reshape(x, (3, 2))
    assert isinstance(y, jnp.ndarray)
    assert y.shape == (3, 2)

    if torch is not None:
        set_backend("torch")
        x = npa.array([[1, 2, 3], [4, 5, 6]])
        y = npa.reshape(x, (3, 2))
        assert isinstance(y, torch.Tensor)
        assert y.shape == (3, 2)
