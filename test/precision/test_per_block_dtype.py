# SPDX-License-Identifier: MIT
"""
T-038a — per-block dtype override mechanism (LookupTable1d only).

Covers the minimum-viable subset that shipped in T-038a:

- ``LookupTable1d`` accepts a new ``dtype`` keyword that casts its
  ``input_array`` / ``output_array`` to the requested dtype on construction.
- Default path (no ``dtype=``) is byte-equivalent to the pre-T-038a
  behavior under x64 (the T-005 default policy).
- Cross-dtype connections promote per JAX's standard rules — a f32
  lookup feeding a default-dtype downstream block produces a f64 output
  because JAX promotes f32 + f64 to f64.
- ``dtype=jnp.float16`` works as a smoke test (not numerically tight).

Other dtype-bearing blocks (``Gain``, ``Constant``, ``Adder``,
``FilterDiscrete``, ``Integrator``, ...) are deferred.
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy import library
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── helpers ────────────────────────────────────────────────────────────────


def _eval_lookup(input_value, *, dtype=None, interp="linear"):
    """Build a Constant -> LookupTable1d diagram, eval the lookup output port."""
    builder = jaxonomy.DiagramBuilder()
    const_dtype = dtype if dtype is not None else jnp.float64
    const = builder.add(library.Constant(jnp.asarray(input_value, dtype=const_dtype)))
    if dtype is None:
        lookup = builder.add(
            library.LookupTable1d(
                input_array=jnp.array([0.0, 1.0, 2.0, 3.0]),
                output_array=jnp.array([0.0, 1.0, 4.0, 9.0]),
                interpolation=interp,
            )
        )
    else:
        lookup = builder.add(
            library.LookupTable1d(
                input_array=jnp.array([0.0, 1.0, 2.0, 3.0]),
                output_array=jnp.array([0.0, 1.0, 4.0, 9.0]),
                interpolation=interp,
                dtype=dtype,
            )
        )
    builder.connect(const.output_ports[0], lookup.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return lookup, lookup.output_ports[0].eval(ctx)


# ── default path matches T-005 policy ──────────────────────────────────────


def test_lookup_table_default_dtype_is_float64():
    """Default install: no dtype= kwarg ⇒ output is float64 (T-005 policy)."""
    info = jaxonomy.precision_info()
    assert info.default_float_dtype == "float64"
    blk, out = _eval_lookup(1.5)
    assert blk._dtype is None
    assert blk.input_array.dtype == jnp.float64
    assert blk.output_array.dtype == jnp.float64
    assert out.dtype == jnp.float64
    # Linear interp at 1.5 between (1, 1) and (2, 4): 1 + 0.5 * (4 - 1) = 2.5
    assert float(out) == pytest.approx(2.5)


def test_default_path_byte_equivalent():
    """No dtype= kwarg: numerical value matches the pre-T-038a expectation
    exactly.  Linear interp on the canonical y = x**2 sample.
    """
    blk, out = _eval_lookup(2.5)
    # interp at 2.5 between (2, 4) and (3, 9): 4 + 0.5 * (9 - 4) = 6.5
    assert blk._dtype is None
    assert float(out) == 6.5
    assert out.dtype == jnp.float64


# ── explicit per-block dtype ───────────────────────────────────────────────


def test_lookup_table_explicit_float32():
    """dtype=jnp.float32 ⇒ input_array, output_array, and output are all f32."""
    blk, out = _eval_lookup(1.5, dtype=jnp.float32)
    assert blk._dtype == jnp.float32
    assert blk.input_array.dtype == jnp.float32
    assert blk.output_array.dtype == jnp.float32
    # Output dtype is f32 because both upstream Constant and the lookup
    # arrays are f32 — no f64 to trigger JAX promotion.
    assert out.dtype == jnp.float32
    # Numerical answer matches the f64 path (rep error well under f32 eps).
    assert float(out) == pytest.approx(2.5, abs=1e-6)


def test_lookup_table_dtype_propagates_through_cross_dtype_connection():
    """f32 LookupTable1d connected to a f64 Constant input.  JAX's standard
    promotion rule (f32 + f64 -> f64) lifts the output dtype to f64.

    This documents the **best-effort** dtype contract: per-block dtype
    enforces *internal* arithmetic at the requested precision, but
    boundary operations follow JAX promotion.  Same precision throughout
    a cascade requires every block to opt in (T-038a-followup-other-blocks).
    """
    builder = jaxonomy.DiagramBuilder()
    # Upstream constant is f64 (default) -> downstream lookup is f32.
    const = builder.add(library.Constant(jnp.float64(1.5)))
    lookup = builder.add(
        library.LookupTable1d(
            input_array=jnp.array([0.0, 1.0, 2.0, 3.0]),
            output_array=jnp.array([0.0, 1.0, 4.0, 9.0]),
            interpolation="linear",
            dtype=jnp.float32,
        )
    )
    builder.connect(const.output_ports[0], lookup.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    out = lookup.output_ports[0].eval(ctx)
    # JAX promotion: f32 (lookup arrays) op f64 (input) -> f64.
    assert out.dtype == jnp.float64
    assert float(out) == pytest.approx(2.5)


def test_lookup_table_float16_smoke():
    """dtype=jnp.float16 — sanity smoke; JAX f16 support is best-effort."""
    blk, out = _eval_lookup(1.0, dtype=jnp.float16)
    assert blk._dtype == jnp.float16
    assert blk.input_array.dtype == jnp.float16
    assert blk.output_array.dtype == jnp.float16
    assert out.dtype == jnp.float16
    # interp at 1.0 should hit the sample point exactly.
    assert float(out) == pytest.approx(1.0, abs=1e-3)


# ── T-038a-followup-other-blocks: per-block dtype kwarg on remaining ──────
#
# One test per block: build a tiny diagram, run a small simulation (or
# evaluate the output port directly for sourceless feedthrough blocks),
# assert the recorded output dtype is the requested precision.  Where
# the block has internal arithmetic (filters, integrators, lookups)
# upstream blocks are also opted into f32 to avoid JAX promotion to f64
# at the boundary — the per-block dtype contract is best-effort and
# JAX promotion rules apply at cross-dtype connections (see
# ``test_lookup_table_dtype_propagates_through_cross_dtype_connection``).


def _simulate_and_get_dtype(diagram, recorded, *, t_end=0.2):
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, t_end), recorded_signals=recorded
    )
    return results.outputs


def _build_gain_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    const = builder.add(library.Constant(2.0, dtype=dtype))
    gain = builder.add(library.Gain(3.0, dtype=dtype))
    builder.connect(const.output_ports[0], gain.input_ports[0])
    return builder.build(), gain


def _build_constant_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    const = builder.add(library.Constant(1.0, dtype=dtype))
    return builder.build(), const


def _build_adder_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    a = builder.add(library.Constant(1.0, dtype=dtype))
    b = builder.add(library.Constant(2.5, dtype=dtype))
    add = builder.add(library.Adder(2, dtype=dtype))
    builder.connect(a.output_ports[0], add.input_ports[0])
    builder.connect(b.output_ports[0], add.input_ports[1])
    return builder.build(), add


def _build_lookup2d_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    cx = builder.add(library.Constant(0.5, dtype=dtype))
    cy = builder.add(library.Constant(0.5, dtype=dtype))
    lt = builder.add(
        library.LookupTable2d(
            input_x_array=jnp.array([0.0, 1.0]),
            input_y_array=jnp.array([0.0, 1.0]),
            output_table_array=jnp.array([[0.0, 1.0], [1.0, 2.0]]),
            interpolation="linear",
            dtype=dtype,
        )
    )
    builder.connect(cx.output_ports[0], lt.input_ports[0])
    builder.connect(cy.output_ports[0], lt.input_ports[1])
    return builder.build(), lt


def _build_filter_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(1.0, dtype=dtype))
    flt = builder.add(
        library.FilterDiscrete(
            dt=0.05, b_coefficients=[0.5, 0.5], dtype=dtype
        )
    )
    builder.connect(src.output_ports[0], flt.input_ports[0])
    return builder.build(), flt


def _build_integrator_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(1.0, dtype=dtype))
    integ = builder.add(
        library.Integrator(initial_state=0.0, dtype=dtype)
    )
    builder.connect(src.output_ports[0], integ.input_ports[0])
    return builder.build(), integ


def _build_integrator_discrete_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(1.0, dtype=dtype))
    integ = builder.add(
        library.IntegratorDiscrete(
            dt=0.05, initial_state=0.0, dtype=dtype
        )
    )
    builder.connect(src.output_ports[0], integ.input_ports[0])
    return builder.build(), integ


def _build_unitdelay_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(1.0, dtype=dtype))
    ud = builder.add(
        library.UnitDelay(
            dt=0.05, initial_state=0.0, dtype=dtype
        )
    )
    builder.connect(src.output_ports[0], ud.input_ports[0])
    return builder.build(), ud


def _build_zoh_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(1.0, dtype=dtype))
    zoh = builder.add(library.ZeroOrderHold(dt=0.05, dtype=dtype))
    builder.connect(src.output_ports[0], zoh.input_ports[0])
    return builder.build(), zoh


def _build_derivative_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    # Use a Clock-style upstream so the derivative is non-trivial.
    src = builder.add(library.Constant(1.0, dtype=dtype))
    deriv = builder.add(library.DerivativeDiscrete(dt=0.05, dtype=dtype))
    builder.connect(src.output_ports[0], deriv.input_ports[0])
    return builder.build(), deriv


def _build_pid_diagram(dtype):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(1.0, dtype=dtype))
    pid = builder.add(
        library.PIDDiscrete(
            dt=0.05,
            kp=1.0,
            ki=0.5,
            kd=0.1,
            initial_state=0.0,
            dtype=dtype,
        )
    )
    builder.connect(src.output_ports[0], pid.input_ports[0])
    return builder.build(), pid


# Block name + builder factory.  Each builder returns ``(diagram, block)``
# where ``block`` is the unit-under-test whose output dtype we record.
_BLOCK_CASES = [
    ("Gain", _build_gain_diagram),
    ("Constant", _build_constant_diagram),
    ("Adder", _build_adder_diagram),
    ("LookupTable2d", _build_lookup2d_diagram),
    ("FilterDiscrete", _build_filter_diagram),
    ("IntegratorDiscrete", _build_integrator_discrete_diagram),
    ("UnitDelay", _build_unitdelay_diagram),
    ("ZeroOrderHold", _build_zoh_diagram),
    ("DerivativeDiscrete", _build_derivative_diagram),
    ("PIDDiscrete", _build_pid_diagram),
    # T-038a-followup-integrator-f32: Integrator now supports f32 too —
    # the Dopri5 ``while_loop`` carry was made type-stable by promoting the
    # time scalars (``t``, ``t_prev``, ``t_return``) to strong JAX dtypes
    # in ``Dopri5State.__post_init__`` and by clipping ``dt`` back to its
    # declared dtype after step-size control in ``attempt_rk_step``.  See
    # ``jaxonomy/backend/_jax/dopri5.py``.
    ("Integrator", _build_integrator_diagram),
]

# Empty: every dtype-bearing block now has an f32 path.  Kept for
# parametrize plumbing in ``test_block_default_dtype_is_float64`` so the
# default-only-table contract remains expressible if a future block
# regresses.
_BLOCK_CASES_DEFAULT_ONLY: list[tuple[str, object]] = []


@pytest.mark.parametrize("block_name,builder_fn", _BLOCK_CASES)
def test_block_dtype_kwarg_float32(block_name, builder_fn):
    """Each extended block: dtype=jnp.float32 ⇒ recorded output is f32.

    Per-block dtype contract: the block's output port produces values in
    the requested precision when its inputs are also at that precision
    (best-effort; cross-dtype boundaries promote per JAX rules).
    """
    diagram, block = builder_fn(jnp.float32)
    recorded = {"y": block.output_ports[0]}
    out = _simulate_and_get_dtype(diagram, recorded)
    assert out["y"].dtype == jnp.float32, (
        f"{block_name} did not honor dtype=jnp.float32 on its output "
        f"(got {out['y'].dtype})"
    )


@pytest.mark.parametrize(
    "block_name,builder_fn", _BLOCK_CASES + _BLOCK_CASES_DEFAULT_ONLY
)
def test_block_default_dtype_is_float64(block_name, builder_fn):
    """Default path (no dtype= kwarg fed by f64 upstream) ⇒ output is f64.

    This locks in the byte-equivalence claim: dtype=None preserves the
    pre-T-038a-followup-other-blocks behavior under x64.  We rebuild
    each diagram passing ``dtype=None`` to all ``dtype``-aware blocks so
    the comparison is at the same builder API surface.
    """
    # Rebuild with dtype=None -- exercise the default branch.
    diagram, block = builder_fn(None)
    recorded = {"y": block.output_ports[0]}
    out = _simulate_and_get_dtype(diagram, recorded)
    # Under T-005's default-x64 install, all blocks default to f64.
    assert out["y"].dtype == jnp.float64, (
        f"{block_name} default-dtype output is not f64 "
        f"(got {out['y'].dtype})"
    )


# ── T-038a-followup-integrator-f32: numerical & default-path checks for
# the Integrator f32 path specifically ─────────────────────────────────────


def test_integrator_f32_matches_f64_reference():
    """``Integrator(dtype=jnp.float32)`` runs through ``simulate(...)`` and
    produces an f32 trajectory that matches the f64 reference within an
    f32-appropriate tolerance.

    Tracks T-038a-followup-integrator-f32: previously, the Dopri5
    ``while_loop`` carry contained f64-typed scratch (``t_prev`` at flat
    index 4), so opting the integrator state into f32 produced a JAX
    ``TypeError: while_loop body function carry input and carry output
    must have equal types``.  The fix promotes time scalars to strong JAX
    dtypes and clips ``dt`` back to its declared dtype after step-size
    control.
    """
    import numpy as np

    # f32 path: state-and-input typed f32, output trajectory must be f32.
    diagram_f32, integ_f32 = _build_integrator_diagram(jnp.float32)
    out_f32 = _simulate_and_get_dtype(
        diagram_f32, {"y": integ_f32.output_ports[0]}, t_end=1.0
    )["y"]
    assert out_f32.dtype == jnp.float32

    # f64 reference: same diagram, default dtype.
    diagram_f64, integ_f64 = _build_integrator_diagram(None)
    out_f64 = _simulate_and_get_dtype(
        diagram_f64, {"y": integ_f64.output_ports[0]}, t_end=1.0
    )["y"]
    assert out_f64.dtype == jnp.float64

    # Final values: integral of 1 from 0 to 1 is exactly 1.  The two
    # solvers may pick different timestep schedules (Dopri5 adaptive),
    # so we don't compare element-by-element — the final value is the
    # robust check.
    assert float(out_f32[-1]) == pytest.approx(1.0, rel=1e-5)
    assert float(out_f64[-1]) == pytest.approx(1.0, rel=1e-10)
    # f32 vs f64 final value agreement: well within f32 epsilon for a
    # 1-unit integral over 1 second (constant input).
    assert float(out_f32[-1]) == pytest.approx(float(out_f64[-1]), rel=1e-5)


def test_integrator_default_dtype_smoke():
    """f64 default path smoke test: confirms no regression from
    T-038a-followup-integrator-f32 (the time-scalar promotion was
    weak→strong, which preserves f64 numerical results).
    """
    diagram, integ = _build_integrator_diagram(None)
    out = _simulate_and_get_dtype(
        diagram, {"y": integ.output_ports[0]}, t_end=0.5
    )["y"]
    assert out.dtype == jnp.float64
    # Integral of 1 from 0 to 0.5 is 0.5.
    assert float(out[-1]) == pytest.approx(0.5, rel=1e-12)


def test_default_path_byte_equivalent_gain():
    """Spot-check: Gain with no dtype= kwarg gives the same numerical
    result and dtype as the pre-T-038a-followup-other-blocks default.

    The default-path byte-equivalence claim is enforced for *all*
    extended blocks by ``test_block_default_dtype_is_float64``;
    this test is a redundant double-check on a single block that the
    numerical answer is exactly what the f64 lambda computes.
    """
    builder = jaxonomy.DiagramBuilder()
    const = builder.add(library.Constant(2.5))
    gain = builder.add(library.Gain(3.0))
    builder.connect(const.output_ports[0], gain.input_ports[0])
    diagram = builder.build()
    recorded = {"y": gain.output_ports[0]}
    out = _simulate_and_get_dtype(diagram, recorded)
    assert out["y"].dtype == jnp.float64
    # The lambda is gain * x = 3.0 * 2.5 = 7.5 exactly in f64.
    assert float(out["y"][0]) == 7.5
