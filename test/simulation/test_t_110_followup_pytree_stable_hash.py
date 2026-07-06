# SPDX-License-Identifier: MIT

"""Tests for T-110-followup-pytree-stable-hash.

The ``config_hash`` field on :class:`ProvenanceManifest` must be
byte-identical across Python 3.10 / 3.11 / 3.12 for the same input
semantics.  The previous implementation used
``json.dumps(..., default=repr)`` which is Python-version-stable for
primitives but not for numpy/JAX arrays, dataclasses, NamedTuples,
sets, or dataclass-field re-orderings.

These tests pin the canonical-form contract for the supported types:

* numpy / JAX arrays hash by ``(shape, dtype, sha256(tobytes))``, so
  two arrays with the same values hash the same regardless of the
  object identity of the wrapper.
* Dataclasses hash by sorted-field-name, so a cosmetic field re-order
  in a future jaxonomy minor version does not change the hash.
* NamedTuples likewise hash by sorted-field-name.
* Sets / frozensets hash insertion-order-invariantly.
* Dicts hash key-order-invariantly (recursively).
* A pinned input has a pinned hash — the regression test that catches
  any future drift in the canonical form.
"""

from __future__ import annotations

import dataclasses
import json
import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.simulation.provenance import (
    _compute_config_hash,
    canonicalize_for_hash,
)


pytestmark = pytest.mark.minimal


# =====================================================================
# canonicalize_for_hash — primitive / container coverage
# =====================================================================


class TestCanonicalizePrimitives:
    def test_none_bool_int_str_passthrough(self):
        assert canonicalize_for_hash(None) is None
        assert canonicalize_for_hash(True) is True
        assert canonicalize_for_hash(False) is False
        assert canonicalize_for_hash(7) == 7
        assert canonicalize_for_hash("hi") == "hi"

    def test_float_is_tagged_repr(self):
        # ``json.dumps(nan)`` raises by default; the canonical form
        # captures floats as a tagged repr so nan/inf survive.
        cf = canonicalize_for_hash(1.5)
        assert cf == ["__float__", "1.5"]
        # Round-trips through json safely.
        json.dumps(cf)
        # Special floats are also captured.
        assert canonicalize_for_hash(float("inf"))[1] == "inf"
        assert canonicalize_for_hash(float("nan"))[1] == "nan"

    def test_bytes_canonical_uses_sha256(self):
        cf = canonicalize_for_hash(b"hello")
        assert cf[0] == "__bytes__"
        # Length of sha256 hex digest.
        assert len(cf[1]) == 64

    def test_list_and_tuple_canonicalise_identically(self):
        # ``list`` and ``tuple`` collapse to the same ``__seq__`` tag
        # — JSON has no notion of tuple, so we don't try to preserve it.
        list_form = canonicalize_for_hash([1, 2, 3])
        tuple_form = canonicalize_for_hash((1, 2, 3))
        assert list_form == tuple_form
        assert list_form[0] == "__seq__"

    def test_set_is_sorted(self):
        a = canonicalize_for_hash({3, 1, 2})
        b = canonicalize_for_hash({2, 3, 1})
        assert a == b
        # Frozensets canonicalise the same way.
        c = canonicalize_for_hash(frozenset({1, 2, 3}))
        assert c == a

    def test_dict_key_order_invariant(self):
        a = canonicalize_for_hash({"b": 1, "a": 2, "c": 3})
        b = canonicalize_for_hash({"c": 3, "a": 2, "b": 1})
        assert a == b


# =====================================================================
# canonicalize_for_hash — numpy / JAX arrays + PRNG keys
# =====================================================================


class TestCanonicalizeArrays:
    def test_numpy_array_same_values_same_canonical_form(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert canonicalize_for_hash(a) == canonicalize_for_hash(b)

    def test_numpy_array_different_values_different_canonical_form(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        assert canonicalize_for_hash(a) != canonicalize_for_hash(b)

    def test_numpy_dtype_matters(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int64)
        assert canonicalize_for_hash(a) != canonicalize_for_hash(b)

    def test_numpy_shape_matters(self):
        a = np.arange(6, dtype=np.float64)
        b = a.reshape(2, 3)
        assert canonicalize_for_hash(a) != canonicalize_for_hash(b)

    def test_jax_array_same_values_same_canonical_form(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 2.0, 3.0])
        ca = canonicalize_for_hash(a)
        cb = canonicalize_for_hash(b)
        assert ca == cb
        assert ca[0] == "__jax_array__"

    def test_jax_prng_key_tagged_separately(self):
        k1 = jax.random.PRNGKey(0)
        k2 = jax.random.PRNGKey(0)
        ck1 = canonicalize_for_hash(k1)
        ck2 = canonicalize_for_hash(k2)
        assert ck1 == ck2
        # The tag depends on dtype/shape rather than provenance, so a
        # ``(2,)`` uint32 array always lands in the PRNG bucket — that's
        # what the legacy key format actually is, semantically.
        assert ck1[0] == "__jax_prng_key__"

    def test_jax_prng_key_different_seeds_different_hash(self):
        k0 = jax.random.PRNGKey(0)
        k1 = jax.random.PRNGKey(1)
        assert canonicalize_for_hash(k0) != canonicalize_for_hash(k1)


# =====================================================================
# canonicalize_for_hash — dataclasses & NamedTuples
# =====================================================================


@dataclasses.dataclass
class _Cfg:
    alpha: float
    beta: int


@dataclasses.dataclass
class _CfgReordered:
    # Same fields as _Cfg but declared in the opposite order.  This is
    # the scenario the canonical form must collapse: a cosmetic field
    # re-order should not perturb the hash when the values match.
    beta: int
    alpha: float


class _Point(typing.NamedTuple):
    x: float
    y: float


class _PointReordered(typing.NamedTuple):
    y: float
    x: float


class TestCanonicalizeRecords:
    def test_dataclass_field_order_invariant_for_values(self):
        # Two dataclasses with identical (qualname-agnostic) fields and
        # values must canonicalise to *structurally-equal* trees when
        # we strip the qualname tag.  We can't equate the full forms
        # because the class qualname is part of the canonical form for
        # type safety — but the field-payload subtree must match.
        a = _Cfg(alpha=0.5, beta=3)
        b = _CfgReordered(beta=3, alpha=0.5)
        ca = canonicalize_for_hash(a)
        cb = canonicalize_for_hash(b)
        # Both forms are __dataclass__ tagged.
        assert ca[0] == "__dataclass__"
        assert cb[0] == "__dataclass__"
        # Field subtree (third element) is order-invariant: sorted by
        # field name.
        assert ca[2] == cb[2]

    def test_namedtuple_field_order_invariant_for_values(self):
        a = _Point(x=1.0, y=2.0)
        b = _PointReordered(y=2.0, x=1.0)
        ca = canonicalize_for_hash(a)
        cb = canonicalize_for_hash(b)
        assert ca[0] == "__namedtuple__"
        assert cb[0] == "__namedtuple__"
        assert ca[2] == cb[2]

    def test_dataclass_value_change_changes_canonical_form(self):
        a = _Cfg(alpha=0.5, beta=3)
        b = _Cfg(alpha=0.5, beta=4)
        assert canonicalize_for_hash(a) != canonicalize_for_hash(b)


# =====================================================================
# Honest fallback for opaque types
# =====================================================================


class TestCanonicalizeFallback:
    def test_lambda_falls_back_to_repr(self):
        f = lambda x: x  # noqa: E731 - deliberate lambda for the test
        cf = canonicalize_for_hash(f)
        assert cf[0] == "__repr__"
        assert isinstance(cf[1], str)


# =====================================================================
# _compute_config_hash — end-to-end determinism
# =====================================================================


class TestConfigHashDeterminismExtended:
    def test_numpy_array_in_options_stable(self):
        opts = {"arr": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        sys = {"type": "S", "system_id": 0, "parameter_names": []}
        h1 = _compute_config_hash(
            options=opts, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        # Re-create the array (different object) — hash must match.
        opts2 = {"arr": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        h2 = _compute_config_hash(
            options=opts2, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        assert h1 == h2

    def test_jax_array_in_options_stable(self):
        opts = {"arr": jnp.array([1.0, 2.0, 3.0])}
        sys = {"type": "S", "system_id": 0, "parameter_names": []}
        h1 = _compute_config_hash(
            options=opts, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        opts2 = {"arr": jnp.array([1.0, 2.0, 3.0])}
        h2 = _compute_config_hash(
            options=opts2, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        assert h1 == h2

    def test_dataclass_reorder_same_hash_subtree(self):
        # Wrap dataclasses inside a top-level dict so the hash compares
        # them via the canonicaliser end-to-end.  We hash the
        # field-payload subtree directly via the canonicaliser, since
        # ``_compute_config_hash`` also includes the class qualname.
        a = _Cfg(alpha=0.5, beta=3)
        b = _CfgReordered(beta=3, alpha=0.5)
        sub_a = canonicalize_for_hash(a)[2]
        sub_b = canonicalize_for_hash(b)[2]
        assert sub_a == sub_b

    def test_dict_key_order_does_not_affect_hash(self):
        opts_a = {"a": 1, "b": 2, "c": 3}
        opts_b = {"c": 3, "a": 1, "b": 2}
        sys = {"type": "S", "system_id": 0, "parameter_names": []}
        h_a = _compute_config_hash(
            options=opts_a, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        h_b = _compute_config_hash(
            options=opts_b, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        assert h_a == h_b

    def test_set_member_order_does_not_affect_hash(self):
        opts_a = {"tags": frozenset({"a", "b", "c"})}
        opts_b = {"tags": frozenset({"c", "a", "b"})}
        sys = {"type": "S", "system_id": 0, "parameter_names": []}
        h_a = _compute_config_hash(
            options=opts_a, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        h_b = _compute_config_hash(
            options=opts_b, system=sys,
            jaxonomy_version="0", jax_version="0",
        )
        assert h_a == h_b


# =====================================================================
# Cross-version regression: pinned hash for a known config
# =====================================================================


class TestCrossVersionPinnedHash:
    """If any future change perturbs the canonical-form bytes for a
    known input, these tests fail and the change is forced to be
    explicit (bump the manifest schema version, document the break)."""

    def test_pinned_hash_primitive_mix(self):
        # Pinned via a one-off Python script at task implementation
        # time on Python 3.11.15.  The same input must produce the
        # same hash on 3.10 / 3.11 / 3.12.
        options = {
            "rtol": 1e-6,
            "atol": 1e-9,
            "flag": True,
            "note": "hello",
            "nested": {"b": 2, "a": 1, "c": 3},
            "sequence": [1, 2.5, "x"],
            "tags": frozenset({"a", "b", "c"}),
        }
        system = {
            "type": "TestSystem", "system_id": 42,
            "parameter_names": ["p", "q"],
        }
        h = _compute_config_hash(
            options=options, system=system,
            jaxonomy_version="pin-test-0", jax_version="pin-test-1",
        )
        # Re-pinned after T-110-followup-stable-fingerprint: the
        # ``system_id`` key is now stripped from the system dict before
        # hashing (per the cross-process stability contract), so the
        # digest differs from the pre-fix pin.
        assert h == (
            "d2673a5c17b21adf9b7cec738505e5b8f29330a7646957b0ae36731fd9fae177"
        )

    def test_pinned_hash_with_numpy_array(self):
        # A second pin anchors the numpy-array branch of the
        # canonicaliser — if the ndarray-digest format ever changes,
        # this test catches it.
        options = {
            "rtol": 1e-6,
            "arr": np.arange(6, dtype=np.float64).reshape(2, 3),
        }
        system = {"type": "S", "system_id": 1, "parameter_names": []}
        h = _compute_config_hash(
            options=options, system=system,
            jaxonomy_version="pin-arr-0", jax_version="pin-arr-1",
        )
        # Re-pinned after T-110-followup-stable-fingerprint: ``system_id``
        # stripped from the hashed system dict.
        assert h == (
            "56762c180d5932f7a380fb92647a0ad1fb988468b68f7efbe92bd1e2a345340e"
        )
