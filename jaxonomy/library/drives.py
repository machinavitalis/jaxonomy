# SPDX-License-Identifier: MIT

"""Electric-drive blocks: dq PMSM machine, Clarke/Park transforms, inverter.

The building blocks of a field-oriented-control (FOC) drive, in the
amplitude-invariant dq convention:

* :class:`PMSM` — interior permanent-magnet synchronous machine in the rotor
  (dq) reference frame, with saliency (``Ld != Lq``) and electromagnetic
  torque ``Te = 1.5*p*(lambda_m*i_q + (Ld - Lq)*i_d*i_q)``.
* :class:`Clarke` / :class:`InverseClarke` — three-phase abc <-> stationary
  alpha-beta frame (amplitude-invariant scaling).
* :class:`Park` / :class:`InversePark` — stationary alpha-beta <-> rotor dq
  frame at electrical angle ``theta_e``.
* :class:`AveragedInverter` — averaged (switching-free) voltage-source
  inverter: passes the commanded ``v_dq`` through a bus-voltage amplitude
  limit (``V_dc/sqrt(3)`` for SVPWM, ``V_dc/2`` for SPWM), preserving the
  command angle when clipping.

These blocks were seeded from the inline implementations validated in the
electric-drives tutorial series (``docs/examples/motor_part_1/2``): the
machine against analytic RL time constants, the torque map, and steady-state
power balance; the transforms against round-trip identity.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..framework import DependencyTicket, LeafSystem, parameters
from ..backend import numpy_api as npa
from .generic import FeedthroughBlock

if TYPE_CHECKING:
    from ..backend.typing import Array

__all__ = [
    "PMSM",
    "Clarke",
    "InverseClarke",
    "Park",
    "InversePark",
    "AveragedInverter",
]

_SQRT3 = 3.0**0.5


class PMSM(LeafSystem):
    """Interior permanent-magnet synchronous machine in the rotor (dq) frame.

    Electrical dynamics (amplitude-invariant dq, electrical speed
    ``w_e = pole_pairs * w_m``)::

        Ld * di_d/dt = v_d - R*i_d + w_e*Lq*i_q
        Lq * di_q/dt = v_q - R*i_q - w_e*(Ld*i_d + lambda_m)

    Electromagnetic torque (magnet + reluctance)::

        Te = 1.5 * pole_pairs * (lambda_m*i_q + (Ld - Lq)*i_d*i_q)

    Mechanical dynamics::

        J * dw_m/dt = Te - B*w_m - T_load
        dtheta_m/dt = w_m

    A surface PMSM or BLDC (sinusoidal back-EMF approximation) is the special
    case ``Ld == Lq``.

    Input ports:
        (0) v_dq: rotor-frame stator voltage ``[v_d, v_q]`` (V).
        (1) T_load: load torque (N*m) — only when ``enable_load_port=True``;
            otherwise the ``T_load`` parameter is used.

    Output ports:
        (0) state: ``[i_d, i_q, w_m, theta_m]`` (A, A, rad/s, rad).
        (1) torque: electromagnetic torque ``Te`` (N*m).

    Parameters:
        R: Stator phase resistance (ohm).
        Ld: d-axis inductance (H).
        Lq: q-axis inductance (H). ``Lq > Ld`` models an interior PMSM.
        lambda_m: Permanent-magnet flux linkage (Wb).
        pole_pairs: Number of pole pairs (electrical/mechanical speed ratio).
        J: Rotor inertia (kg*m^2).
        B: Viscous friction coefficient (N*m*s).
        T_load: Constant load torque (N*m); ignored when the load port is on.
        initial_state: Initial ``[i_d, i_q, w_m, theta_m]`` (default zeros).
        locked: Clamp the rotor (``dw_m = dtheta_m = 0``) — a standstill
            bench test that reduces each axis to an RL circuit with time
            constant ``L/R``.
        enable_load_port: Expose input port (1) for the load torque.

    The electrical angle for the Park transforms is
    ``theta_e = pole_pairs * theta_m``.
    """

    @parameters(
        dynamic=["R", "Ld", "Lq", "lambda_m", "pole_pairs", "J", "B", "T_load"],
        static=["initial_state", "locked", "enable_load_port"],
    )
    def __init__(
        self,
        R: float = 0.45,
        Ld: float = 3.2e-3,
        Lq: float = 5.8e-3,
        lambda_m: float = 0.0533,
        pole_pairs: float = 4.0,
        J: float = 1.2e-3,
        B: float = 8.0e-5,
        T_load: float = 0.0,
        initial_state=None,
        locked: bool = False,
        enable_load_port: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._locked = bool(locked)
        self._enable_load_port = bool(enable_load_port)
        self._validate_initial_state(initial_state)

        self.declare_input_port(name="v_dq")
        if self._enable_load_port:
            self.declare_input_port(name="T_load")

        # Declared here (not via declare_continuous_state_output in
        # initialize()) so the state is output port 0 and torque port 1 —
        # initialize() runs after __init__'s port declarations.
        self.declare_output_port(
            self._state_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            name="state",
        )
        self.declare_output_port(
            self._torque_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            name="torque",
        )

    @staticmethod
    def _validate_initial_state(initial_state):
        if initial_state is None:
            return npa.zeros(4)
        initial_state = npa.asarray(initial_state)
        if initial_state.shape != (4,):
            raise ValueError(
                "PMSM initial_state must have shape (4,) "
                f"[i_d, i_q, w_m, theta_m]; got {initial_state.shape}."
            )
        return initial_state

    def initialize(
        self,
        R,
        Ld,
        Lq,
        lambda_m,
        pole_pairs,
        J,
        B,
        T_load,
        initial_state=None,
        locked=False,
        enable_load_port=False,
    ):
        initial_state = self._validate_initial_state(initial_state)
        self.declare_continuous_state(default_value=initial_state, ode=self._ode)

    def _state_output(self, _time, state, *_inputs, **_p) -> Array:
        return state.continuous_state

    @staticmethod
    def _torque(i_d, i_q, lambda_m, pole_pairs, Ld, Lq):
        return 1.5 * pole_pairs * (lambda_m * i_q + (Ld - Lq) * i_d * i_q)

    def _ode(self, _time, state, *inputs, **p) -> Array:
        i_d, i_q, w_m, _theta_m = state.continuous_state
        v_dq = inputs[0]
        v_d, v_q = v_dq[0], v_dq[1]
        T_load = inputs[1] if self._enable_load_port else p["T_load"]

        R, Ld, Lq = p["R"], p["Ld"], p["Lq"]
        lambda_m, pole_pairs = p["lambda_m"], p["pole_pairs"]

        w_e = pole_pairs * w_m
        di_d = (v_d - R * i_d + w_e * Lq * i_q) / Ld
        di_q = (v_q - R * i_q - w_e * (Ld * i_d + lambda_m)) / Lq

        if self._locked:
            dw_m = 0.0 * w_m
            dtheta = 0.0 * w_m
        else:
            Te = self._torque(i_d, i_q, lambda_m, pole_pairs, Ld, Lq)
            dw_m = (Te - p["B"] * w_m - T_load) / p["J"]
            dtheta = w_m
        return npa.stack([di_d, di_q, dw_m, dtheta])

    def _torque_output(self, _time, state, *_inputs, **p) -> Array:
        i_d, i_q = state.continuous_state[0], state.continuous_state[1]
        return self._torque(
            i_d, i_q, p["lambda_m"], p["pole_pairs"], p["Ld"], p["Lq"]
        )


class Clarke(FeedthroughBlock):
    """Clarke transform: three-phase ``[a, b, c]`` -> stationary ``[alpha, beta]``.

    Amplitude-invariant scaling: a balanced sinusoidal three-phase set of
    peak amplitude ``M`` maps to an alpha-beta vector of magnitude ``M``.
    The zero-sequence component is discarded.
    """

    def __init__(self, **kwargs):
        def _clarke(abc):
            a, b, c = abc[0], abc[1], abc[2]
            return npa.stack([(2 * a - b - c) / 3.0, (b - c) / _SQRT3])

        super().__init__(_clarke, **kwargs)


class InverseClarke(FeedthroughBlock):
    """Inverse Clarke transform: ``[alpha, beta]`` -> three-phase ``[a, b, c]``.

    Amplitude-invariant, zero-sequence-free: ``a + b + c = 0`` by
    construction, and ``InverseClarke(Clarke(abc)) == abc`` for any
    zero-sequence-free input.
    """

    def __init__(self, **kwargs):
        def _inv_clarke(ab):
            al, be = ab[0], ab[1]
            return npa.stack(
                [al, -0.5 * al + (_SQRT3 / 2) * be, -0.5 * al - (_SQRT3 / 2) * be]
            )

        super().__init__(_inv_clarke, **kwargs)


class _AngleTransform(LeafSystem):
    """Shared two-input (vector, angle) -> rotated-vector block."""

    _rotate = None  # set by subclass: staticmethod (x0, x1, cos, sin) -> (y0, y1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port(name=self._input_name)
        self.declare_input_port(name="theta")
        self.declare_output_port(
            self._output,
            requires_inputs=True,
            prerequisites_of_calc=[
                self.input_ports[0].ticket,
                self.input_ports[1].ticket,
            ],
            name=self._output_name,
        )

    def _output(self, _time, _state, *inputs, **_p) -> Array:
        vec, theta = inputs
        theta = npa.reshape(npa.asarray(theta), ())
        ct, st = npa.cos(theta), npa.sin(theta)
        y0, y1 = self._rotate(vec[0], vec[1], ct, st)
        return npa.stack([y0, y1])


class Park(_AngleTransform):
    """Park transform: stationary ``[alpha, beta]`` -> rotor ``[d, q]``.

    Input ports:
        (0) alpha_beta: stationary-frame vector.
        (1) theta: electrical angle ``theta_e`` (rad); for :class:`PMSM`,
            ``theta_e = pole_pairs * theta_m``.

    Output ports:
        (0) dq: ``[d, q] = [ cos*alpha + sin*beta, -sin*alpha + cos*beta ]``.
    """

    _input_name = "alpha_beta"
    _output_name = "dq"

    @staticmethod
    def _rotate(al, be, ct, st):
        return ct * al + st * be, -st * al + ct * be


class InversePark(_AngleTransform):
    """Inverse Park transform: rotor ``[d, q]`` -> stationary ``[alpha, beta]``.

    Input ports:
        (0) dq: rotor-frame vector.
        (1) theta: electrical angle ``theta_e`` (rad).

    Output ports:
        (0) alpha_beta: ``[cos*d - sin*q, sin*d + cos*q]``.
    """

    _input_name = "dq"
    _output_name = "alpha_beta"

    @staticmethod
    def _rotate(d, q, ct, st):
        return ct * d - st * q, st * d + ct * q


class AveragedInverter(LeafSystem):
    """Averaged voltage-source inverter with the bus-voltage amplitude limit.

    Switching-free (averaged) model: the commanded rotor-frame voltage passes
    through unchanged while its magnitude is realizable, and is scaled down
    onto the voltage circle — preserving its angle — when it is not::

        v_lim = V_dc / sqrt(3)   (SVPWM)   or   V_dc / 2   (SPWM)
        v_out = v_cmd * min(1, v_lim / |v_cmd|)

    Input ports:
        (0) v_dq_cmd: commanded ``[v_d, v_q]`` (V).

    Output ports:
        (0) v_dq: realizable ``[v_d, v_q]`` after the amplitude limit.

    Parameters:
        V_dc: DC bus voltage (V).
        modulation: ``"svpwm"`` (default, limit ``V_dc/sqrt(3)``) or
            ``"spwm"`` (limit ``V_dc/2``).
    """

    @parameters(dynamic=["V_dc"], static=["modulation"])
    def __init__(self, V_dc: float = 48.0, modulation: str = "svpwm", **kwargs):
        super().__init__(**kwargs)
        if modulation not in ("svpwm", "spwm"):
            raise ValueError(
                f"AveragedInverter modulation must be 'svpwm' or 'spwm', "
                f"got {modulation!r}."
            )
        self._limit_divisor = _SQRT3 if modulation == "svpwm" else 2.0
        self.declare_input_port(name="v_dq_cmd")
        self.declare_output_port(
            self._output,
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            name="v_dq",
        )

    def initialize(self, V_dc, modulation="svpwm"):
        pass

    def _output(self, _time, _state, *inputs, **p) -> Array:
        (v_cmd,) = inputs
        v_lim = p["V_dc"] / self._limit_divisor
        mag = npa.sqrt(v_cmd[0] ** 2 + v_cmd[1] ** 2) + 1e-12
        scale = npa.minimum(1.0, v_lim / mag)
        return v_cmd * scale
