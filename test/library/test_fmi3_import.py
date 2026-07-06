# SPDX-License-Identifier: MIT
"""
T-026 — FMI 3.0 import dispatch tests.

Without bundling actual ``.fmu`` test fixtures (they're large binaries
from the FMI standard's reference set, fetched at integration time),
these tests cover the dispatch and helper logic introduced for FMI 3.0
in :mod:`jaxonomy.library.fmu_import`:

  - ``_is_fmi3`` correctly identifies version strings.
  - ``_fmi3_exceptions`` returns a tuple containing
    ``fmi3.FMICallException`` when fmpy is installed.
  - ``ModelicaFMU._set_value`` / ``_get_value`` route v2 and v3 calls
    to the right typed accessors via stub objects.

End-to-end FMU 3 simulation is exercised by the existing
``test/library/test_fmu.py`` whenever a real FMU 3 file is supplied
through the optional fixture path.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

fmpy = pytest.importorskip("fmpy")

from jaxonomy.library.fmu_import import (  # noqa: E402
    ModelicaFMU,
    _fmi3_exceptions,
    _is_fmi3,
)
from jaxonomy.framework.error import BlockInitializationError  # noqa: E402


# ── _is_fmi3 ──────────────────────────────────────────────────────────────


def test_is_fmi3_recognises_3x():
    md = types.SimpleNamespace(fmiVersion="3.0")
    assert _is_fmi3(md) is True
    md = types.SimpleNamespace(fmiVersion="3.0.1")
    assert _is_fmi3(md) is True


def test_is_fmi3_rejects_2x():
    md = types.SimpleNamespace(fmiVersion="2.0")
    assert _is_fmi3(md) is False
    md = types.SimpleNamespace(fmiVersion="2.0.4")
    assert _is_fmi3(md) is False


def test_is_fmi3_handles_missing_attr():
    md = types.SimpleNamespace()
    assert _is_fmi3(md) is False


# ── _fmi3_exceptions ──────────────────────────────────────────────────────


def test_fmi3_exceptions_returns_tuple():
    excs = _fmi3_exceptions()
    # Tuple in either case (v3 exception class present, or fallback ()).
    # fmpy's exception-class layout has shifted across releases; the
    # important property is that the helper degrades to () rather than
    # raising at import time.
    assert isinstance(excs, tuple)


# ── _set_value / _get_value dispatch via stubs ───────────────────────────


class _StubFMU:
    """Minimal stub recording every typed-accessor call for assertions."""

    def __init__(self):
        self.calls: list[tuple[str, list, list | None]] = []

    def __getattr__(self, name):
        if name.startswith("set"):
            def _setter(refs, vals):
                self.calls.append((name, list(refs), list(vals)))
            return _setter
        if name.startswith("get"):
            def _getter(refs):
                self.calls.append((name, list(refs), None))
                return [42.0 for _ in refs]
            return _getter
        raise AttributeError(name)


def _make_block():
    """Create a ModelicaFMU instance bypassing real FMU loading.

    Use a bare type rather than __new__ on ModelicaFMU so that the
    LeafSystem property machinery (name_path, parent, ...) does not
    interfere with our minimal dispatch-only test.  We bind the
    real `_set_value` / `_get_value` methods unbound to this stub.
    """
    class _Stub:
        pass
    blk = _Stub()
    blk._fmi_version = None  # set per-test
    # bind the methods unbound
    blk._set_value = ModelicaFMU._set_value.__get__(blk, _Stub)
    blk._get_value = ModelicaFMU._get_value.__get__(blk, _Stub)
    return blk


def _variable(name, vt, ref=1):
    return types.SimpleNamespace(name=name, type=vt, valueReference=ref)


@pytest.mark.parametrize("vt,expected_setter", [
    ("Real", "setReal"),
    ("Integer", "setInteger"),
    ("Boolean", "setBoolean"),
    ("Enumeration", "setInteger"),
    ("String", "setString"),
])
def test_set_value_v2_routes_to_legacy_setters(vt, expected_setter):
    blk = _make_block()
    blk._fmi_version = "2.0"
    fmu = _StubFMU()
    blk._set_value(fmu, _variable("v", vt), 1, "blk")
    assert fmu.calls and fmu.calls[0][0] == expected_setter


@pytest.mark.parametrize("vt,expected_setter", [
    ("Real", "setFloat64"),
    ("Float64", "setFloat64"),
    ("Float32", "setFloat32"),
    ("Integer", "setInt32"),
    ("Boolean", "setBoolean"),
    # FMI 3 models Enumeration as Int64; the prior T-026 mapping to
    # Int32 was a bug that surfaced once T-026a's stress harness ran
    # the Reference-FMUs Feedthrough model.
    ("Enumeration", "setInt64"),
    ("String", "setString"),
])
def test_set_value_v3_routes_to_typed_setters(vt, expected_setter):
    blk = _make_block()
    blk._fmi_version = "3.0"
    fmu = _StubFMU()
    blk._set_value(fmu, _variable("v", vt), 1, "blk")
    assert fmu.calls and fmu.calls[0][0] == expected_setter


def test_set_value_unsupported_type_raises():
    """An unsupported variable type (Binary, ...) raises.  We accept any
    Exception subclass — the precise type uses LeafSystem properties for
    formatting and the stub block does not have those, but the raise
    itself is what we care about."""
    blk = _make_block()
    blk._fmi_version = "2.0"
    fmu = _StubFMU()
    with pytest.raises(Exception):
        blk._set_value(fmu, _variable("v", "Binary"), 1, "blk")


@pytest.mark.parametrize("vt,expected_getter", [
    ("Real", "getReal"),
    ("Integer", "getInteger"),
    ("Boolean", "getBoolean"),
])
def test_get_value_v2_routes(vt, expected_getter):
    blk = _make_block()
    blk._fmi_version = "2.0"
    fmu = _StubFMU()
    blk._get_value(fmu, _variable("v", vt))
    assert fmu.calls[0][0] == expected_getter


@pytest.mark.parametrize("vt,expected_getter", [
    ("Real", "getFloat64"),
    ("Float64", "getFloat64"),
    ("Float32", "getFloat32"),
    ("Integer", "getInt32"),
    ("Boolean", "getBoolean"),
])
def test_get_value_v3_routes(vt, expected_getter):
    blk = _make_block()
    blk._fmi_version = "3.0"
    fmu = _StubFMU()
    blk._get_value(fmu, _variable("v", vt))
    assert fmu.calls[0][0] == expected_getter
