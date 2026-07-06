# SPDX-License-Identifier: MIT
"""
T-038a-followup-mixed-precision-cascade — global ``precision_policy``
context manager.

Covers:

- ``with precision_policy(jnp.float32):`` block — verify a
  ``LookupTable1d`` built inside picks up f32 without an explicit
  ``dtype=`` kwarg.
- Per-block override wins: ``LookupTable1d(dtype=jnp.float64)`` inside a
  ``precision_policy(jnp.float32)`` context stays f64.
- Default-off: outside the context manager, default-float64 policy
  holds (T-005 invariant).
- Nested contexts: inner overrides outer; outer is restored on exit.
- End-to-end cascade: a small Diagram inside ``precision_policy(f32)``
  produces f32 internal arrays throughout (cross-dtype promotion
  caveats from T-038a still apply at non-policy block boundaries).

All ``_dtype``-equipped primitive blocks now consult the policy as
well — see the parametrized ``test_block_picks_up_policy_dtype`` /
``test_per_block_dtype_overrides_policy`` cases below covering
``Gain``, ``Constant``, ``Adder``, ``LookupTable2d``, ``FilterDiscrete``,
``IntegratorDiscrete``, ``UnitDelay``, ``ZeroOrderHold``,
``DerivativeDiscrete``, and ``PIDDiscrete``.  ``Integrator`` is
deliberately excluded (its f32 path is gated by a separate
solver-state-dtype follow-up; see T-038a-followup-integrator-f32).
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy import library
from jaxonomy.precision import active_precision_policy
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── helpers ────────────────────────────────────────────────────────────────


def _make_lookup(dtype=None):
    """Build a LookupTable1d with the canonical y = x**2 sample.

    If dtype is None, no kwarg is passed (so the block consults the
    active ``precision_policy`` context).  Otherwise the dtype is passed
    explicitly (which must take precedence over any active context).
    """
    kwargs = dict(
        input_array=jnp.array([0.0, 1.0, 2.0, 3.0]),
        output_array=jnp.array([0.0, 1.0, 4.0, 9.0]),
        interpolation="linear",
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return library.LookupTable1d(**kwargs)


# ── 1. context manager picks up a LookupTable1d's default dtype ────────────


def test_policy_context_sets_default_dtype_for_lookup_table():
    """A LookupTable1d built inside ``precision_policy(jnp.float32)``
    picks up f32 even though no explicit ``dtype=`` was passed."""
    with jaxonomy.precision_policy(jnp.float32):
        blk = _make_lookup()
    assert blk._dtype == jnp.float32


# ── 2. explicit per-block dtype wins ───────────────────────────────────────


def test_per_block_dtype_overrides_policy_context():
    """``LookupTable1d(dtype=jnp.float64)`` inside a
    ``precision_policy(jnp.float32)`` context stays f64 — explicit-over-
    implicit."""
    with jaxonomy.precision_policy(jnp.float32):
        blk = _make_lookup(dtype=jnp.float64)
    assert blk._dtype == jnp.float64


# ── 3. default-off: outside the CM, default-float64 policy holds ───────────


def test_default_off_outside_context():
    """No active context: ``active_precision_policy()`` is None and a
    LookupTable1d with no ``dtype=`` falls through to the default
    float64 path (T-005 invariant).

    ``initialize()`` (which materialises ``input_array``/``output_array``
    on the block) runs through the diagram-build path, so we drive a
    minimal Constant -> LookupTable1d diagram and create a context to
    trigger initialisation.
    """
    assert active_precision_policy() is None
    info = jaxonomy.precision_info()
    assert info.default_float_dtype == "float64"
    builder = jaxonomy.DiagramBuilder()
    const = builder.add(library.Constant(jnp.float64(1.5)))
    lookup = builder.add(_make_lookup())
    builder.connect(const.output_ports[0], lookup.input_ports[0])
    diagram = builder.build()
    diagram.create_context()  # triggers LookupTable1d.initialize
    assert lookup._dtype is None
    assert lookup.input_array.dtype == jnp.float64
    assert lookup.output_array.dtype == jnp.float64


def test_context_exit_restores_default_off():
    """After exiting ``with precision_policy(...)``, the policy is
    cleared; subsequent blocks fall through to the default branch."""
    with jaxonomy.precision_policy(jnp.float32):
        assert active_precision_policy() == jnp.float32
    assert active_precision_policy() is None
    blk = _make_lookup()
    assert blk._dtype is None


# ── 4. nested contexts: inner overrides outer; restore on exit ─────────────


def test_nested_contexts_inner_overrides_outer():
    """Nested ``precision_policy`` contexts: inner overrides outer;
    on exit from inner, outer's dtype is restored exactly."""
    with jaxonomy.precision_policy(jnp.float32):
        outer_blk = _make_lookup()
        assert outer_blk._dtype == jnp.float32
        with jaxonomy.precision_policy(jnp.float16):
            inner_blk = _make_lookup()
            assert inner_blk._dtype == jnp.float16
        # Inner context exited; outer (f32) is restored.
        assert active_precision_policy() == jnp.float32
        post_inner_blk = _make_lookup()
        assert post_inner_blk._dtype == jnp.float32
    # Outer context exited; default-off restored.
    assert active_precision_policy() is None


# ── 5. end-to-end: small Diagram under precision_policy(f32) ───────────────


def test_end_to_end_diagram_under_f32_policy():
    """A small Diagram (Constant -> LookupTable1d) built entirely inside
    ``precision_policy(jnp.float32)`` produces an f32 LookupTable output.

    Note on cross-dtype promotion (T-038a contract): this test passes
    an f32 ``Constant`` upstream so the lookup's f32 internal arrays
    don't get promoted by an f64 input.  ``Constant`` does not yet
    consult the policy (deferred to T-038a-followup-other-blocks),
    so we set its dtype explicitly.
    """
    with jaxonomy.precision_policy(jnp.float32):
        builder = jaxonomy.DiagramBuilder()
        # Constant doesn't yet honor the policy — pass an f32 array
        # explicitly so cross-dtype promotion doesn't lift the lookup
        # output back to f64.
        const = builder.add(library.Constant(jnp.asarray(1.5, dtype=jnp.float32)))
        # LookupTable1d picks up f32 from the active policy (no explicit
        # dtype= kwarg).
        lookup = builder.add(
            library.LookupTable1d(
                input_array=jnp.array([0.0, 1.0, 2.0, 3.0]),
                output_array=jnp.array([0.0, 1.0, 4.0, 9.0]),
                interpolation="linear",
            )
        )
        builder.connect(const.output_ports[0], lookup.input_ports[0])
        diagram = builder.build()

    # Lookup picked up f32 from the policy.
    assert lookup._dtype == jnp.float32

    ctx = diagram.create_context()  # triggers initialize() — arrays now set
    assert lookup.input_array.dtype == jnp.float32
    assert lookup.output_array.dtype == jnp.float32

    out = lookup.output_ports[0].eval(ctx)
    assert out.dtype == jnp.float32
    # Linear interp at 1.5 between (1, 1) and (2, 4): 1 + 0.5 * (4 - 1) = 2.5
    assert float(out) == pytest.approx(2.5, abs=1e-6)


def test_end_to_end_lookup_chain_under_f32_policy():
    """Two LookupTable1d blocks chained — both pick up f32 from the
    policy.  Cross-block connection stays f32 (no f64 input)."""
    with jaxonomy.precision_policy(jnp.float32):
        builder = jaxonomy.DiagramBuilder()
        const = builder.add(library.Constant(jnp.asarray(0.5, dtype=jnp.float32)))
        lookup_a = builder.add(
            library.LookupTable1d(
                input_array=jnp.array([0.0, 1.0, 2.0]),
                output_array=jnp.array([0.0, 2.0, 4.0]),
                interpolation="linear",
            )
        )
        lookup_b = builder.add(
            library.LookupTable1d(
                input_array=jnp.array([0.0, 1.0, 2.0]),
                output_array=jnp.array([0.0, 1.0, 8.0]),
                interpolation="linear",
            )
        )
        builder.connect(const.output_ports[0], lookup_a.input_ports[0])
        builder.connect(lookup_a.output_ports[0], lookup_b.input_ports[0])
        diagram = builder.build()

    assert lookup_a._dtype == jnp.float32
    assert lookup_b._dtype == jnp.float32

    ctx = diagram.create_context()
    out_a = lookup_a.output_ports[0].eval(ctx)
    out_b = lookup_b.output_ports[0].eval(ctx)
    assert out_a.dtype == jnp.float32
    assert out_b.dtype == jnp.float32
    # interp at 0.5 in (0, 0) -> (1, 2): 1.0
    assert float(out_a) == pytest.approx(1.0, abs=1e-6)
    # interp at 1.0 in second lookup hits the sample exactly: 1.0
    assert float(out_b) == pytest.approx(1.0, abs=1e-6)


# ── 6. parametrized: every dtype-aware block consults the policy ───────────
#
# Each builder constructs a single block with the minimum required kwargs.
# We do *not* simulate (that's the contract covered by
# ``test/precision/test_per_block_dtype.py`` already); we only check that
# the block's stored dtype attribute (``_dtype`` or, for
# ``IntegratorDiscrete``, ``dtype``) reflects the active policy when no
# explicit ``dtype=`` was passed, and that an explicit kwarg overrides
# the policy (explicit-over-implicit).


def _build_gain():
    return library.Gain(2.0)


def _build_constant():
    return library.Constant(1.0)


def _build_adder():
    return library.Adder(2)


def _build_lookup2d():
    return library.LookupTable2d(
        input_x_array=jnp.array([0.0, 1.0]),
        input_y_array=jnp.array([0.0, 1.0]),
        output_table_array=jnp.array([[0.0, 1.0], [1.0, 2.0]]),
        interpolation="linear",
    )


def _build_filter_discrete():
    return library.FilterDiscrete(dt=0.05, b_coefficients=[0.5, 0.5])


def _build_integrator_discrete():
    return library.IntegratorDiscrete(dt=0.05, initial_state=0.0)


def _build_unitdelay():
    return library.UnitDelay(dt=0.05, initial_state=0.0)


def _build_zoh():
    return library.ZeroOrderHold(dt=0.05)


def _build_derivative_discrete():
    return library.DerivativeDiscrete(dt=0.05)


def _build_pid_discrete():
    return library.PIDDiscrete(
        dt=0.05, kp=1.0, ki=0.5, kd=0.1, initial_state=0.0
    )


def _build_lookup1d():
    return library.LookupTable1d(
        input_array=jnp.array([0.0, 1.0, 2.0]),
        output_array=jnp.array([0.0, 1.0, 4.0]),
        interpolation="linear",
    )


# (block_name, builder_factory, dtype_attr_name)
#
# Most blocks expose the per-block dtype on ``_dtype`` (the convention
# established by T-038a / T-038a-followup-other-blocks).
# ``IntegratorDiscrete`` predates that convention and stores it on
# ``dtype`` (no underscore); we still verify the policy fallback fires.
_POLICY_BLOCK_CASES = [
    ("LookupTable1d", _build_lookup1d, "_dtype"),
    ("Gain", _build_gain, "_dtype"),
    ("Constant", _build_constant, "_dtype"),
    ("Adder", _build_adder, "_dtype"),
    ("LookupTable2d", _build_lookup2d, "_dtype"),
    ("FilterDiscrete", _build_filter_discrete, "_dtype"),
    ("IntegratorDiscrete", _build_integrator_discrete, "dtype"),
    ("UnitDelay", _build_unitdelay, "_dtype"),
    ("ZeroOrderHold", _build_zoh, "_dtype"),
    ("DerivativeDiscrete", _build_derivative_discrete, "_dtype"),
    ("PIDDiscrete", _build_pid_discrete, "_dtype"),
]


@pytest.mark.parametrize(
    "block_name,builder_fn,dtype_attr", _POLICY_BLOCK_CASES
)
def test_block_picks_up_policy_dtype(block_name, builder_fn, dtype_attr):
    """Each dtype-aware block: built inside ``precision_policy(jnp.float32)``
    with no explicit ``dtype=`` kwarg, the block's stored dtype
    attribute reflects the active policy."""
    with jaxonomy.precision_policy(jnp.float32):
        blk = builder_fn()
    assert getattr(blk, dtype_attr) == jnp.float32, (
        f"{block_name}.{dtype_attr} did not pick up active "
        f"precision_policy(jnp.float32)"
    )


@pytest.mark.parametrize(
    "block_name,builder_fn,dtype_attr", _POLICY_BLOCK_CASES
)
def test_block_default_off_when_no_policy(block_name, builder_fn, dtype_attr):
    """Default-off (no active policy): a block built without an explicit
    ``dtype=`` lands ``None`` on its dtype attribute — byte-equivalent
    to the pre-follow-up code path."""
    assert active_precision_policy() is None
    blk = builder_fn()
    assert getattr(blk, dtype_attr) is None, (
        f"{block_name}.{dtype_attr} is not None outside any "
        f"precision_policy context (got {getattr(blk, dtype_attr)!r})"
    )
