# SPDX-License-Identifier: MIT

"""Electric-drives blocks (``jaxonomy.library.drives``).

Validations mirror the electric-drives tutorial series that seeded these
blocks (``docs/examples/motor_part_1/2``):

* locked-rotor bench test: each dq axis is an RL circuit with tau = L/R,
* the torque output implements ``1.5*p*(lam*i_q + (Ld-Lq)*i_d*i_q)``,
* no-load steady state matches an independent scipy root-solve of the
  equilibrium equations, and satisfies the steady-state power balance
  ``P_elec = copper loss + friction loss``,
* Clarke/Park round-trip identity and synchronous-frame alignment,
* averaged inverter: passthrough below the bus limit, angle-preserving
  scaling onto the ``V_dc/sqrt(3)`` (or ``V_dc/2``) circle above it.
"""

import numpy as np
import pytest

import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions, simulate
from jaxonomy.library import (
    PMSM,
    AveragedInverter,
    Clarke,
    Constant,
    InverseClarke,
    InversePark,
    Park,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

PARAMS = dict(
    R=0.45,
    Ld=3.2e-3,
    Lq=5.8e-3,
    lambda_m=0.0533,
    pole_pairs=4.0,
    J=1.2e-3,
    B=8.0e-5,
)


def _torque(i_d, i_q, p=PARAMS):
    return 1.5 * p["pole_pairs"] * (
        p["lambda_m"] * i_q + (p["Ld"] - p["Lq"]) * i_d * i_q
    )


def _simulate_pmsm(v_dq, t_end, n=2000, **pmsm_kwargs):
    b = DiagramBuilder()
    src = b.add(Constant(jnp.asarray(v_dq, dtype=float), name="v"))
    mot = b.add(PMSM(**PARAMS, **pmsm_kwargs, name="pmsm"))
    b.connect(src.output_ports[0], mot.input_ports[0])
    diagram = b.build()
    ctx = diagram.create_context()
    res = simulate(
        diagram,
        ctx,
        (0.0, t_end),
        options=SimulatorOptions(max_major_step_length=t_end / n),
        recorded_signals={
            "x": mot.output_ports[0],
            "Te": mot.output_ports[1],
        },
    )
    return (
        np.asarray(res.time),
        np.asarray(res.outputs["x"]),
        np.asarray(res.outputs["Te"]),
    )


def _eval_block(block, inputs):
    for port, val in zip(block.input_ports, inputs):
        port.fix_value(jnp.asarray(val, dtype=float))
    ctx = block.create_context()
    return np.asarray(block.output_ports[0].eval(ctx))


# ---------------------------------------------------------------------------
# PMSM
# ---------------------------------------------------------------------------


def test_locked_rotor_axes_are_rl_circuits():
    # With the rotor clamped (w_e = 0) each axis decouples to L*di/dt = v - R*i
    # with time constant L/R and DC gain v/R.
    v_d, v_q = 5.0, 3.0
    tau_d = PARAMS["Ld"] / PARAMS["R"]
    t_end = 8.0 * max(tau_d, PARAMS["Lq"] / PARAMS["R"])
    t, X, _ = _simulate_pmsm([v_d, v_q], t_end, locked=True)

    i_d_ref = (v_d / PARAMS["R"]) * (1.0 - np.exp(-t * PARAMS["R"] / PARAMS["Ld"]))
    i_q_ref = (v_q / PARAMS["R"]) * (1.0 - np.exp(-t * PARAMS["R"] / PARAMS["Lq"]))
    np.testing.assert_allclose(X[:, 0], i_d_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(X[:, 1], i_q_ref, rtol=2e-3, atol=2e-3)
    # Rotor stayed clamped.
    np.testing.assert_allclose(X[:, 2:], 0.0, atol=1e-12)


def test_torque_output_matches_formula():
    t, X, Te = _simulate_pmsm([2.0, 12.0], 0.2)
    np.testing.assert_allclose(
        Te.ravel(), _torque(X[:, 0], X[:, 1]), rtol=1e-9, atol=1e-11
    )
    assert np.max(np.abs(Te)) > 0.1  # the trajectory actually produced torque


def test_no_load_steady_state_matches_root_solve():
    fsolve = pytest.importorskip("scipy.optimize").fsolve
    v_d, v_q = 0.0, 15.0  # open-loop-stable regime (tutorial section 7)
    p = PARAMS

    def residual(z):
        i_d, i_q, w_m = z
        w_e = p["pole_pairs"] * w_m
        return [
            v_d - p["R"] * i_d + w_e * p["Lq"] * i_q,
            v_q - p["R"] * i_q - w_e * (p["Ld"] * i_d + p["lambda_m"]),
            _torque(i_d, i_q) - p["B"] * w_m,
        ]

    z_ss = fsolve(residual, [0.0, v_q / p["R"], 100.0], full_output=False)
    assert np.max(np.abs(residual(z_ss))) < 1e-6

    t, X, _ = _simulate_pmsm([v_d, v_q], 0.6, n=4000)
    np.testing.assert_allclose(X[-1, :3], z_ss, rtol=2e-2, atol=1e-3)

    # Steady-state power balance: 1.5*(v_d*i_d + v_q*i_q) = copper + friction.
    i_d, i_q, w_m = X[-1, :3]
    p_in = 1.5 * (v_d * i_d + v_q * i_q)
    p_cu = 1.5 * p["R"] * (i_d**2 + i_q**2)
    p_fric = p["B"] * w_m**2
    np.testing.assert_allclose(p_in, p_cu + p_fric, rtol=2e-2)


def test_load_port_shifts_torque_balance():
    T_load = 0.8
    b = DiagramBuilder()
    v = b.add(Constant(jnp.array([0.0, 15.0]), name="v"))
    load = b.add(Constant(jnp.asarray(T_load), name="load"))
    mot = b.add(PMSM(**PARAMS, enable_load_port=True, name="pmsm"))
    b.connect(v.output_ports[0], mot.input_ports[0])
    b.connect(load.output_ports[0], mot.input_ports[1])
    diagram = b.build()
    ctx = diagram.create_context()
    res = simulate(
        diagram,
        ctx,
        (0.0, 0.6),
        options=SimulatorOptions(max_major_step_length=0.6 / 4000),
        recorded_signals={"x": mot.output_ports[0], "Te": mot.output_ports[1]},
    )
    X = np.asarray(res.outputs["x"])
    Te = np.asarray(res.outputs["Te"]).ravel()
    # Steady state: Te = B*w + T_load.
    np.testing.assert_allclose(
        Te[-1], PARAMS["B"] * X[-1, 2] + T_load, rtol=2e-2
    )


def test_pmsm_rejects_bad_initial_state():
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        PMSM(initial_state=np.zeros(3))


# ---------------------------------------------------------------------------
# Clarke / Park transforms
# ---------------------------------------------------------------------------


def test_clarke_park_round_trip_identity():
    abc = np.array([2.0, -1.3, -0.7])
    abc = abc - abc.mean()  # zero-sequence-free
    theta = 0.9

    ab = _eval_block(Clarke(name="c"), [abc])
    dq = _eval_block(Park(name="p"), [ab, theta])
    ab_back = _eval_block(InversePark(name="ip"), [dq, theta])
    abc_back = _eval_block(InverseClarke(name="ic"), [ab_back])

    np.testing.assert_allclose(ab_back, ab, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(abc_back, abc, rtol=1e-12, atol=1e-12)


def test_park_of_balanced_set_is_dc():
    # A balanced three-phase set of peak M at electrical angle theta maps to
    # [M, 0] in the synchronous frame (amplitude-invariant convention).
    M, theta = 7.3, 1.234
    abc = M * np.array(
        [np.cos(theta), np.cos(theta - 2 * np.pi / 3), np.cos(theta + 2 * np.pi / 3)]
    )
    ab = _eval_block(Clarke(name="c"), [abc])
    dq = _eval_block(Park(name="p"), [ab, theta])
    np.testing.assert_allclose(dq, [M, 0.0], rtol=1e-12, atol=1e-10)


# ---------------------------------------------------------------------------
# AveragedInverter
# ---------------------------------------------------------------------------


def test_inverter_passthrough_below_limit():
    v_cmd = np.array([5.0, -8.0])  # |v| ~ 9.4 < 48/sqrt(3) ~ 27.7
    out = _eval_block(AveragedInverter(V_dc=48.0, name="inv"), [v_cmd])
    np.testing.assert_allclose(out, v_cmd, rtol=1e-9)


@pytest.mark.parametrize(
    "modulation,divisor", [("svpwm", np.sqrt(3.0)), ("spwm", 2.0)]
)
def test_inverter_clamps_to_voltage_circle(modulation, divisor):
    V_dc = 48.0
    v_cmd = np.array([30.0, 40.0])  # |v| = 50, over both limits
    out = _eval_block(
        AveragedInverter(V_dc=V_dc, modulation=modulation, name="inv"), [v_cmd]
    )
    v_lim = V_dc / divisor
    np.testing.assert_allclose(np.linalg.norm(out), v_lim, rtol=1e-6)
    # Angle preserved.
    np.testing.assert_allclose(
        np.arctan2(out[1], out[0]), np.arctan2(v_cmd[1], v_cmd[0]), rtol=1e-9
    )


def test_inverter_rejects_unknown_modulation():
    with pytest.raises(ValueError, match="svpwm"):
        AveragedInverter(modulation="six-step")


# ---------------------------------------------------------------------------
# Composition: inverter-limited PMSM
# ---------------------------------------------------------------------------


def test_pmsm_behind_saturated_inverter():
    # Command far above the bus limit; the machine must respond exactly as if
    # driven by the limited voltage.
    v_cmd = [0.0, 100.0]
    V_dc = 48.0
    v_lim = V_dc / np.sqrt(3.0)

    b = DiagramBuilder()
    src = b.add(Constant(jnp.asarray(v_cmd, dtype=float), name="cmd"))
    inv = b.add(AveragedInverter(V_dc=V_dc, name="inv"))
    mot = b.add(PMSM(**PARAMS, locked=True, name="pmsm"))
    b.connect(src.output_ports[0], inv.input_ports[0])
    b.connect(inv.output_ports[0], mot.input_ports[0])
    diagram = b.build()
    ctx = diagram.create_context()
    tau_q = PARAMS["Lq"] / PARAMS["R"]
    res = simulate(
        diagram,
        ctx,
        (0.0, 8 * tau_q),
        options=SimulatorOptions(max_major_step_length=8 * tau_q / 2000),
        recorded_signals={"x": mot.output_ports[0]},
    )
    t = np.asarray(res.time)
    i_q = np.asarray(res.outputs["x"])[:, 1]
    i_q_ref = (v_lim / PARAMS["R"]) * (
        1.0 - np.exp(-t * PARAMS["R"] / PARAMS["Lq"])
    )
    np.testing.assert_allclose(i_q, i_q_ref, rtol=2e-3, atol=2e-3)
