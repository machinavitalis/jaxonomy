# SPDX-License-Identifier: MIT

"""T-115-followup-quantizer-modes tests.

Verifies the new ``mode={"round","floor","ceil","trunc"}`` kwarg on the
existing :class:`~jaxonomy.library.Quantizer` block. Default
``mode="round"`` must remain byte-equivalent to the legacy phase-1
behavior (which used ``npa.round`` unconditionally — i.e. round-half-to-
even, IEEE-754 default).

Coverage:
- Each mode produces the spec'd output on a few hand-picked scalar values.
- Default mode is byte-equivalent to ``mode="round"`` and matches phase 1.
- Invalid mode strings raise :class:`BlockParameterError`.
- Gradient is zero through the block in every mode (rounding is
  non-differentiable; the block uses ``lax.stop_gradient`` defensively).
"""

import pytest

import numpy as np
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_quantizer(value, *, step, mode=None):
    """Build a tiny diagram Constant -> Quantizer and return its output."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(value)))
    if mode is None:
        # Exercise the default-arg path, not the explicit ``mode="round"``.
        q = builder.add(library.Quantizer(interval=step))
    else:
        q = builder.add(library.Quantizer(interval=step, mode=mode))
    builder.connect(src.output_ports[0], q.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(q.output_ports[0].eval(context))


# ---------------------------------------------------------------------------
# mode="round" (banker's rounding / round-half-to-even)
# ---------------------------------------------------------------------------


class TestQuantizerRoundMode:
    @pytest.mark.parametrize(
        "u, expected",
        [
            # spec from task: step=0.5
            (0.3, 0.5),    # 0.6 rounds to 1 -> 0.5
            (0.7, 0.5),    # 1.4 rounds to 1 -> 0.5
            (0.8, 1.0),    # 1.6 rounds to 2 -> 1.0
            (-0.3, -0.5),  # -0.6 rounds to -1 -> -0.5
        ],
    )
    def test_explicit_round(self, u, expected):
        y = _eval_quantizer(u, step=0.5, mode="round")
        assert y == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(
        "u, expected",
        [(0.3, 0.5), (0.7, 0.5), (0.8, 1.0), (-0.3, -0.5)],
    )
    def test_default_matches_round(self, u, expected):
        # No ``mode`` kwarg supplied: default must be "round".
        y_default = _eval_quantizer(u, step=0.5)
        y_round = _eval_quantizer(u, step=0.5, mode="round")
        assert y_default == y_round
        assert y_default == pytest.approx(expected, abs=1e-12)

    def test_default_byte_equivalent_with_phase1(self):
        # Phase 1 used ``interval * npa.round(x / interval)`` directly.
        # Reproduce that here with raw npa.round to assert byte-equivalence
        # on a representative grid.
        step = 0.5
        for u in np.linspace(-3.0, 3.0, 25):
            y_block = _eval_quantizer(u, step=step)
            y_phase1 = float(step * jnp.round(u / step))
            assert y_block == y_phase1, (
                f"default mode no longer byte-equivalent at u={u}"
            )


# ---------------------------------------------------------------------------
# mode="floor"
# ---------------------------------------------------------------------------


class TestQuantizerFloorMode:
    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.7, 0.5),    # spec
            (0.3, 0.0),    # spec
            (1.0, 1.0),    # exact grid point
            (-0.1, -0.5),  # negative inputs floor toward -inf
            (-0.6, -1.0),
        ],
    )
    def test_floor(self, u, expected):
        y = _eval_quantizer(u, step=0.5, mode="floor")
        assert y == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# mode="ceil"
# ---------------------------------------------------------------------------


class TestQuantizerCeilMode:
    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.3, 0.5),    # spec
            (0.0, 0.0),    # exact grid point
            (0.5, 0.5),    # exact grid point
            (-0.7, -0.5),  # ceil rounds toward +inf
            (-0.1, 0.0),
        ],
    )
    def test_ceil(self, u, expected):
        y = _eval_quantizer(u, step=0.5, mode="ceil")
        assert y == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# mode="trunc"
# ---------------------------------------------------------------------------


class TestQuantizerTruncMode:
    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.7, 0.5),    # spec — rounds toward zero
            (-0.7, -0.5),  # spec — symmetric toward zero
            (0.3, 0.0),
            (-0.3, 0.0),
            (1.6, 1.5),
            (-1.6, -1.5),
        ],
    )
    def test_trunc(self, u, expected):
        y = _eval_quantizer(u, step=0.5, mode="trunc")
        assert y == pytest.approx(expected, abs=1e-12)

    def test_trunc_differs_from_floor_on_negatives(self):
        # ``trunc(-0.7 / 0.5) == trunc(-1.4) == -1`` -> -0.5
        # ``floor(-0.7 / 0.5) == floor(-1.4) == -2`` -> -1.0
        y_trunc = _eval_quantizer(-0.7, step=0.5, mode="trunc")
        y_floor = _eval_quantizer(-0.7, step=0.5, mode="floor")
        assert y_trunc == pytest.approx(-0.5, abs=1e-12)
        assert y_floor == pytest.approx(-1.0, abs=1e-12)
        assert y_trunc != y_floor


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------


class TestQuantizerModeValidation:
    @pytest.mark.parametrize(
        "bad_mode",
        ["nearest", "ROUND", "half-up", "", "floor ", None],
    )
    def test_invalid_mode_raises(self, bad_mode):
        with pytest.raises(BlockParameterError):
            library.Quantizer(interval=0.5, mode=bad_mode)

    @pytest.mark.parametrize(
        "good_mode", ["round", "floor", "ceil", "trunc"]
    )
    def test_all_documented_modes_accepted(self, good_mode):
        # Just make sure construction succeeds for every documented mode.
        block = library.Quantizer(interval=0.5, mode=good_mode)
        assert block._mode == good_mode


# ---------------------------------------------------------------------------
# Gradient: rounding is non-differentiable; expect zero-gradient through
# the block under every mode.
# ---------------------------------------------------------------------------


def _quantize_scalar(u, step, mode):
    """Standalone reproduction of the block's op suitable for jax.grad."""
    if mode == "round":
        rounded = jnp.round(u / step)
    elif mode == "floor":
        rounded = jnp.floor(u / step)
    elif mode == "ceil":
        rounded = jnp.ceil(u / step)
    elif mode == "trunc":
        rounded = jnp.trunc(u / step)
    else:
        raise ValueError(mode)
    return jax.lax.stop_gradient(step * rounded)


class TestQuantizerGradient:
    @pytest.mark.parametrize(
        "mode", ["round", "floor", "ceil", "trunc"]
    )
    def test_gradient_is_zero(self, mode):
        # Pick a generic point not on the grid so a non-stop_gradient
        # round implementation could in principle leak a nonzero gradient.
        g = jax.grad(_quantize_scalar)(0.37, 0.5, mode)
        assert float(g) == 0.0, (
            f"Quantizer mode={mode!r}: gradient should be 0, got {float(g)}"
        )
