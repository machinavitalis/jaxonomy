# SPDX-License-Identifier: MIT

"""T-115-followup-deadzone-bilinear tests.

Verifies:

* :class:`~jaxonomy.library.DeadZone` ``output_shifted`` kwarg.
  Default ``output_shifted=False`` keeps the legacy Coulomb-style hard
  dead zone (output jumps at the band boundary -- ``y = u`` outside the
  band). ``output_shifted=True`` switches to the shifted-output
  form where the output is continuous across the band boundary
  (``y = u - half_range*sign(u)`` outside, slope 1, value 0 at the edge).
* :class:`~jaxonomy.library.DeadZoneInverse` -- the dual of the hard
  dead-zone block. Outputs ``u`` INSIDE the band and ``0`` outside, with
  zero-crossing events declared at the band edges.
"""

import math

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _eval_deadzone(value, *, half_range=1.0, **kwargs):
    """Run a DeadZone block on a single Constant input and return the output."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(value)))
    dz = builder.add(library.DeadZone(half_range=half_range, **kwargs))
    builder.connect(src.output_ports[0], dz.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(dz.output_ports[0].eval(context)), dz


def _eval_deadzone_inverse(value, *, half_range=1.0, **kwargs):
    """Run a DeadZoneInverse block on a single Constant input."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(value)))
    dz = builder.add(library.DeadZoneInverse(half_range=half_range, **kwargs))
    builder.connect(src.output_ports[0], dz.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(dz.output_ports[0].eval(context)), dz


# ---------------------------------------------------------------------------
# DeadZone: output_shifted=True (shifted-output variant, continuous)
# ---------------------------------------------------------------------------


class TestDeadZoneOutputShiftedTrue:
    def test_input_above_band_is_shifted(self):
        y, _ = _eval_deadzone(1.0, half_range=0.5, output_shifted=True)
        assert math.isclose(y, 0.5, abs_tol=1e-12)

    def test_input_below_band_is_shifted(self):
        y, _ = _eval_deadzone(-1.0, half_range=0.5, output_shifted=True)
        assert math.isclose(y, -0.5, abs_tol=1e-12)

    def test_input_inside_band_zero(self):
        y, _ = _eval_deadzone(0.3, half_range=0.5, output_shifted=True)
        assert y == 0.0

    def test_continuous_at_band_boundary(self):
        # Just outside the band -> ~0, just inside -> 0; output continuous.
        eps = 1e-9
        y_inside, _ = _eval_deadzone(
            0.5 - eps, half_range=0.5, output_shifted=True
        )
        y_outside, _ = _eval_deadzone(
            0.5 + eps, half_range=0.5, output_shifted=True
        )
        assert abs(y_outside - y_inside) < 1e-6

    @pytest.mark.parametrize("u", [-3.0, -1.5, 1.5, 3.0])
    def test_slope_one_outside_band(self, u):
        """Outside the band the shifted output has slope 1 wrt input."""
        hr = 0.5
        y, _ = _eval_deadzone(u, half_range=hr, output_shifted=True)
        # y(u) = u - hr*sign(u)
        expected = u - hr * (1.0 if u > 0 else -1.0)
        assert math.isclose(y, expected, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# DeadZone: output_shifted=False (legacy Coulomb-style, jumps)
# ---------------------------------------------------------------------------


class TestDeadZoneOutputShiftedFalse:
    def test_input_above_band_unshifted(self):
        y, _ = _eval_deadzone(1.0, half_range=0.5, output_shifted=False)
        assert math.isclose(y, 1.0, abs_tol=1e-12)

    def test_input_below_band_unshifted(self):
        y, _ = _eval_deadzone(-1.0, half_range=0.5, output_shifted=False)
        assert math.isclose(y, -1.0, abs_tol=1e-12)

    def test_input_inside_band_zero(self):
        y, _ = _eval_deadzone(0.3, half_range=0.5, output_shifted=False)
        assert y == 0.0


# ---------------------------------------------------------------------------
# DeadZone: default is byte-equivalent to phase 1 (output_shifted=False)
# ---------------------------------------------------------------------------


class TestDeadZoneDefaultByteEquivalence:
    @pytest.mark.parametrize("u", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_default_matches_legacy(self, u):
        """Default DeadZone matches legacy (unshifted, Coulomb) formula."""
        y, _ = _eval_deadzone(u, half_range=1.0)
        # Legacy formula: where(|u| < hr, 0, u).
        y_ref = 0.0 if abs(u) < 1.0 else u
        assert y == y_ref

    @pytest.mark.parametrize("u", [-2.0, 0.5, 2.0])
    def test_default_matches_explicit_false(self, u):
        y_default, _ = _eval_deadzone(u, half_range=1.0)
        y_explicit, _ = _eval_deadzone(
            u, half_range=1.0, output_shifted=False
        )
        assert y_default == y_explicit


# ---------------------------------------------------------------------------
# DeadZone: validation of output_shifted kwarg
# ---------------------------------------------------------------------------


class TestDeadZoneOutputShiftedValidation:
    def test_non_bool_raises(self):
        with pytest.raises(BlockParameterError):
            library.DeadZone(half_range=1.0, output_shifted="yes")

    def test_int_raises(self):
        with pytest.raises(BlockParameterError):
            library.DeadZone(half_range=1.0, output_shifted=1)

    def test_initialize_cannot_change_output_shifted(self):
        dz = library.DeadZone(half_range=1.0, output_shifted=True)
        with pytest.raises(ValueError):
            dz.initialize(half_range=1.0, output_shifted=False)


# ---------------------------------------------------------------------------
# DeadZoneInverse: outputs input INSIDE band, zero OUTSIDE
# ---------------------------------------------------------------------------


class TestDeadZoneInverseBasic:
    def test_input_inside_band_passes_through(self):
        y, _ = _eval_deadzone_inverse(0.3, half_range=0.5)
        assert math.isclose(y, 0.3, abs_tol=1e-12)

    def test_input_above_band_zero(self):
        y, _ = _eval_deadzone_inverse(1.0, half_range=0.5)
        assert y == 0.0

    def test_input_below_band_zero(self):
        y, _ = _eval_deadzone_inverse(-1.0, half_range=0.5)
        assert y == 0.0

    def test_origin_passes_through(self):
        y, _ = _eval_deadzone_inverse(0.0, half_range=0.5)
        assert y == 0.0

    @pytest.mark.parametrize("u", [-0.49, -0.25, 0.0, 0.25, 0.49])
    def test_pass_through_inside_band(self, u):
        y, _ = _eval_deadzone_inverse(u, half_range=0.5)
        assert math.isclose(y, u, abs_tol=1e-12)

    @pytest.mark.parametrize("u", [-2.0, -1.5, 0.51, 1.5, 2.0])
    def test_zero_outside_band(self, u):
        y, _ = _eval_deadzone_inverse(u, half_range=0.5)
        assert y == 0.0


# ---------------------------------------------------------------------------
# DeadZoneInverse: validation
# ---------------------------------------------------------------------------


class TestDeadZoneInverseValidation:
    def test_negative_half_range_raises(self):
        with pytest.raises(BlockParameterError):
            library.DeadZoneInverse(half_range=-1.0)

    def test_zero_half_range_raises(self):
        with pytest.raises(BlockParameterError):
            library.DeadZoneInverse(half_range=0.0)


# ---------------------------------------------------------------------------
# DeadZoneInverse: zero-crossing events declared
# ---------------------------------------------------------------------------


class TestDeadZoneInverseZeroCrossings:
    def test_zero_crossings_declared(self):
        """DeadZoneInverse declares ZC events at the band edges."""
        builder = jaxonomy.DiagramBuilder()
        ramp = builder.add(library.Ramp(start_value=-2.0, start_time=0.0))
        dz = builder.add(library.DeadZoneInverse(half_range=0.5))
        integ = builder.add(library.Integrator(0.0))
        builder.connect(ramp.output_ports[0], dz.input_ports[0])
        builder.connect(dz.output_ports[0], integ.input_ports[0])
        diagram = builder.build()
        # Building the diagram triggers initialize_static_data which
        # declares the ZC events. Verify the events were attached.
        diagram.create_context()
        assert dz.has_zero_crossing_events


# ---------------------------------------------------------------------------
# Default-off byte-equivalence: simulated trajectory matches legacy
# ---------------------------------------------------------------------------


class TestDeadZoneDefaultSimulationByteEquivalence:
    def test_simulation_matches_legacy_formula(self):
        """End-to-end: default DeadZone in a sim produces the legacy output."""
        builder = jaxonomy.DiagramBuilder()
        half_range = 0.25
        start_value = -2.0
        ramp = builder.add(
            library.Ramp(start_time=0.0, start_value=start_value)
        )
        dz = builder.add(library.DeadZone(half_range=half_range))
        builder.connect(ramp.output_ports[0], dz.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "dz": dz.output_ports[0],
            "ramp": ramp.output_ports[0],
        }
        r = jaxonomy.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        ramp_arr = r.outputs["ramp"]
        dz_arr = r.outputs["dz"]
        # Legacy reference: where(|u| < hr, 0, u). Default-off behavior.
        ref = np.zeros_like(ramp_arr)
        ref[ramp_arr < -half_range] = ramp_arr[ramp_arr < -half_range]
        ref[ramp_arr > half_range] = ramp_arr[ramp_arr > half_range]
        assert np.allclose(ref, dz_arr)
