# SPDX-License-Identifier: MIT
"""Check every shipped reference solution against an independent result.

``generate.py`` produces the references by driving the FMUs it just
built. That catches nothing: a model that is wrong in-process stays
wrong through the FMI boundary, and the reference looks plausible
either way. This script re-derives each trajectory a second way —
analytically, or through a solver that is not Jaxonomy — and fails
loudly when they disagree.

Run from the repository root, after ``generate.py``::

    python fmi-compatibility/verify.py

Requires ``scipy`` (ships with jaxonomy), plus ``control`` and ``onnx``
for the DCMotor and ONNXPolicy cross-checks. The tolerances are the
figures quoted in README.md's "How each reference was checked" table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# FMI co-simulation holds inputs constant across a step, so the output
# recorded at t[k+1] is the response to the input applied at t[k].
# Comparing an input-driven model against its analytic map has to shift
# by one sample or it measures the hold, not the model.
INPUT_HOLD_SHIFT = 1

FAILURES: list[str] = []


def _load(model: str, kind: str = "ref") -> np.ndarray:
    path = HERE / model / f"{model}_{kind}.csv"
    return np.genfromtxt(path, delimiter=",", names=True)


def _check(model: str, label: str, ok: bool, detail: str) -> None:
    if not ok:
        FAILURES.append(f"{model}: {label} — {detail}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {model:14s} {label:34s} {detail}")


def _complete(model: str, stop_time: float = 10.0) -> np.ndarray:
    """Load a reference and assert it actually runs to its stop time.

    A truncated reference is the failure mode that looks most like a
    success: the columns are all present and the values all sane.
    """
    data = _load(model)
    reached = float(data["time"][-1])
    _check(model, "reaches stop time", abs(reached - stop_time) < 1e-6,
           f"t[-1]={reached:.4f}, want {stop_time}")
    return data


def check_mixed_types() -> None:
    data, given = _complete("MixedTypes"), _load("MixedTypes", "in")
    out, inp = data[INPUT_HOLD_SHIFT:], given[:-INPUT_HOLD_SHIFT or None]
    err = np.abs(out["y_real"] - 2 * inp["u_real"]).max()
    _check("MixedTypes", "y_real == 2*u_real", err < 1e-8, f"max|err|={err:.2e}")
    _check("MixedTypes", "y_int == u_int (Integer)",
           np.array_equal(out["y_int"], inp["u_int"]),
           f"integer-valued throughout, y_int[-1]={out['y_int'][-1]:.0f}")
    _check("MixedTypes", "y_bool == not u_bool (Boolean)",
           np.array_equal(out["y_bool"].astype(bool), ~inp["u_bool"].astype(bool)),
           f"{int(out['y_bool'].sum())} true of {len(out)}")
    _check("MixedTypes", "y_over == (u_real > 0.5)",
           np.array_equal(out["y_over"].astype(bool), inp["u_real"] > 0.5),
           f"{int(out['y_over'].sum())} true of {len(out)}")


def check_vector_io() -> None:
    data = _complete("VectorIO")
    columns = [c for c in data.dtype.names if c != "time"]
    expected = [2.0, 4.0, 6.0, 12.0]  # 2*[1,2,3] and its sum
    ok = len(columns) == 4 and all(
        np.allclose(data[c], v) for c, v in zip(columns, expected)
    )
    _check("VectorIO", "elements == 2*[1,2,3], sum 12", ok,
           f"{columns} -> {[float(data[c][-1]) for c in columns]}")


def check_parameterized() -> None:
    data = _complete("Parameterized")
    time = data["time"]
    # x0 and decay_rate are the FMI parameters generate.py sets at init.
    exact = 2.5 * np.exp(-0.8 * time)
    err = np.abs(data["x"] - exact).max()
    _check("Parameterized", "x == 2.5*exp(-0.8t)", err < 1e-6,
           f"max|err|={err:.3e}; confirms parameters applied at init")


def check_thermostat() -> None:
    data = _complete("Thermostat")
    time, temp, heater = data["time"], data["T"], data["heater"]
    c_th, r_th, t_amb, p_heat = 2.0, 1.0, 20.0, 40.0
    t_set, band = 50.0, 1.0
    t_inf, tau = t_amb + p_heat * r_th, r_th * c_th

    settled = time > 3.0
    _check("Thermostat", "T held inside the hysteresis band",
           temp[settled].min() > t_set - band - 1e-3
           and temp[settled].max() < t_set + band + 1e-3,
           f"[{temp[settled].min():.4f}, {temp[settled].max():.4f}] "
           f"want [{t_set - band}, {t_set + band}]")

    rising = time < 2.0
    err = np.abs(
        temp[rising] - (t_inf - (t_inf - t_amb) * np.exp(-time[rising] / tau))
    ).max()
    _check("Thermostat", "first rise == analytic", err < 1e-6,
           f"max|err|={err:.3e}")

    # Each half-cycle is an exponential segment between the two
    # thresholds, so its duration is analytic.
    on_dt = tau * np.log((t_inf - (t_set - band)) / (t_inf - (t_set + band)))
    off_dt = tau * np.log(((t_set + band) - t_amb) / ((t_set - band) - t_amb))
    switches = np.flatnonzero(np.diff(heater) != 0)
    durations = np.diff(time[switches])[-6:]
    worst = max(min(abs(d - on_dt), abs(d - off_dt)) for d in durations)
    both = (any(abs(d - on_dt) < 0.011 for d in durations)
            and any(abs(d - off_dt) < 0.011 for d in durations))
    _check("Thermostat", "switch durations == analytic",
           worst < 0.011 and both,
           f"{durations.round(4)} vs on={on_dt:.4f} off={off_dt:.4f}")


def check_stiff_chemical() -> None:
    from scipy.integrate import solve_ivp

    data = _complete("StiffChemical")
    time = data["time"]
    species = np.column_stack([data["y1"], data["y2"], data["y3"]])

    drift = np.abs(species.sum(axis=1) - 1.0).max()
    _check("StiffChemical", "mass conserved (sum == 1)", drift < 1e-10,
           f"max|sum-1|={drift:.3e}")

    def robertson(_t, y):
        return [-0.04 * y[0] + 1e4 * y[1] * y[2],
                0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
                3e7 * y[1] ** 2]

    reference = solve_ivp(robertson, (0.0, time[-1]), [1.0, 0.0, 0.0],
                          method="Radau", t_eval=time, rtol=1e-10, atol=1e-12)
    rel = (np.abs(species - reference.y.T).max(axis=0)
           / np.abs(reference.y.T).max(axis=0))
    _check("StiffChemical", "matches SciPy Radau", rel.max() < 1e-5,
           f"rel err per species={np.array2string(rel, precision=2)}")


def check_dc_motor() -> None:
    import control

    data = _complete("DCMotor")
    time = data["time"]
    r_a, k_m, l_a, j_m, b_l, v_in = 1.0, 0.05, 0.5, 0.01, 1e-4, 12.0
    # The same equations as a plain state space, solved by a tool that
    # shares no code with jaxonomy:
    #   L di/dt = V - R i - K w ;  J dw/dt = K i - B w
    a_mat = np.array([[-r_a / l_a, -k_m / l_a], [k_m / j_m, -b_l / j_m]])
    b_mat = np.array([[1.0 / l_a], [0.0]])
    system = control.ss(a_mat, b_mat, np.eye(2), np.zeros((2, 1)))
    _, states = control.forced_response(
        system, T=time, U=np.full_like(time, v_in), X0=[0.0, 0.0]
    )
    for column, reference, label in ((data["amp_i"], states[0], "current"),
                                     (data["speed_w_rel"], states[1], "speed")):
        rel = np.abs(column - reference).max() / np.abs(reference).max()
        _check("DCMotor", f"{label} vs python-control", rel < 1e-4,
               f"rel={rel:.3e}, end={column[-1]:.6f}")


def check_onnx_policy() -> None:
    import onnx
    from onnx import numpy_helper

    data = _complete("ONNXPolicy")
    time, state = data["time"], data["x"]
    graph = onnx.load(str(HERE / "ONNXPolicy" / "policy.onnx")).graph
    w = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    def policy(x: float) -> float:
        hidden = np.tanh(np.array([[x]], np.float32) @ w["W1"] + w["b1"])
        return float((hidden @ w["W2"] + w["b2"])[0, 0])

    # u is held across each sample, so dx/dt = -x + u integrates exactly
    # over the interval — the reference needs no solver at all.
    ts, x = 0.1, 1.0
    times, values = [0.0], [x]
    for _ in range(int(round(time[-1] / ts))):
        u = policy(x)
        x = u + (x - u) * np.exp(-ts)
        times.append(times[-1] + ts)
        values.append(x)
    err = np.abs(np.interp(np.array(times), time, state) - np.array(values)).max()
    _check("ONNXPolicy", "closed loop == exact ZOH reference", err < 1e-6,
           f"max|err|={err:.3e} (float32 policy weights bound this)")


def check_units_annotated() -> None:
    data = _complete("UnitsAnnotated")
    time = data["time"]
    area, r_out, q_in = 2.0, 4.0, 0.5
    exact = q_in * r_out * (1.0 - np.exp(-time / (area * r_out)))
    err = np.abs(data["level"] - exact).max()
    _check("UnitsAnnotated", "level == analytic tank fill", err < 1e-6,
           f"max|err|={err:.3e}")


def check_feedthrough() -> None:
    data, given = _complete("Feedthrough"), _load("Feedthrough", "in")
    out, inp = data[INPUT_HOLD_SHIFT:], given[:-INPUT_HOLD_SHIFT or None]
    for column, gain in (("y_scalar", 1.0), ("y_gain", 2.0)):
        err = np.abs(out[column] - gain * inp["u_scalar"]).max()
        _check("Feedthrough", f"{column} == {gain:g}*u_scalar", err < 1e-9,
               f"max|err|={err:.3e}")


def check_spring_damper() -> None:
    import control

    data, given = _complete("SpringDamper"), _load("SpringDamper", "in")
    mass, damping, stiffness, step = 1.0, 0.5, 1.0, 0.01
    system = control.ss(
        np.array([[0.0, 1.0], [-stiffness / mass, -damping / mass]]),
        np.array([[0.0], [1.0 / mass]]),
        np.array([[1.0, 0.0]]), np.array([[0.0]]),
    )
    # The FMU holds F across each communication step, so the exact
    # comparison is the ZOH-discretised system, not a continuous solve.
    discrete = control.c2d(system, step, method="zoh")
    _, position = control.forced_response(
        discrete, T=data["time"], U=given["F"], X0=[0.0, 0.0]
    )
    err = np.abs(data["x"] - position).max()
    _check("SpringDamper", "x == ZOH-discretised python-control", err < 1e-6,
           f"max|err|={err:.3e}")


def check_pi_controller() -> None:
    data, given = _complete("PIController"), _load("PIController", "in")
    k_p, k_i, step = 2.0, 0.5, 0.01
    error = given["setpoint"] - given["measurement"]
    # x[k+1] = x[k] + dt*e[k]; u[k] = Kp*e[k] + Ki*x[k].
    integral = np.concatenate([[0.0], np.cumsum(error[:-1]) * step])
    expected = k_p * error + k_i * integral
    out, ref = data["u"][INPUT_HOLD_SHIFT:], expected[:-INPUT_HOLD_SHIFT or None]
    err = np.abs(out - ref).max()
    _check("PIController", "u == analytic discrete PI", err < 1e-8,
           f"max|err|={err:.3e}")


def check_bouncing_ball() -> None:
    # Zeno accumulation stops this one early; _ref.opt records where.
    data = _load("BouncingBall")
    time, height = data["time"], data["h"]
    gravity, restitution = 9.81, 0.7
    _check("BouncingBall", "terminates on Zeno accumulation",
           2.0 < float(time[-1]) < 3.0, f"t[-1]={float(time[-1]):.4f}")
    _check("BouncingBall", "height never goes negative", height.min() > -1e-6,
           f"min h={height.min():.3e}")

    # Free flight before the first impact is exact: h = h0 - g t^2 / 2.
    first_impact = float(np.sqrt(2.0 * 1.0 / gravity))
    flight = time < first_impact - 1e-3
    err = np.abs(height[flight] - (1.0 - 0.5 * gravity * time[flight] ** 2)).max()
    _check("BouncingBall", "first arc == analytic free flight", err < 1e-6,
           f"max|err|={err:.3e}")

    # Each apex is restitution^2 times the previous one. Only apexes the
    # output grid actually resolves count: near a peak the trajectory is
    # parabolic, so a 0.01 s grid misses the true maximum by up to
    # g*(dt/2)^2/2 ~ 1.2e-4 m, which swamps the ratio once the bounces
    # decay below a few centimetres.
    resolvable = 0.05
    apexes = [height[i] for i in range(1, len(height) - 1)
              if height[i] > height[i - 1] and height[i] >= height[i + 1]]
    ratios = [b / a for a, b in zip(apexes, apexes[1:]) if a > resolvable]
    _check("BouncingBall", "apex ratio == e^2",
           bool(ratios) and max(abs(r - restitution ** 2) for r in ratios) < 5e-3,
           f"{np.round(ratios, 4)} vs e^2={restitution ** 2:.4f} "
           f"(apexes above {resolvable} m)")


def check_rc_network() -> None:
    data = _complete("RCNetwork")
    time = data["time"]
    exact = 1.0 - np.exp(-time / (100.0 * 1e-3))
    err = np.abs(data["v_c"] - exact).max()
    _check("RCNetwork", "v_c == analytic RC charging", err < 1e-4,
           f"max|err|={err:.3e}")


CHECKS = (
    check_spring_damper,
    check_pi_controller,
    check_bouncing_ball,
    check_feedthrough,
    check_rc_network,
    check_mixed_types,
    check_vector_io,
    check_parameterized,
    check_thermostat,
    check_stiff_chemical,
    check_dc_motor,
    check_onnx_policy,
    check_units_annotated,
)


def main() -> None:
    for check in CHECKS:
        check()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED:")
        for failure in FAILURES:
            print(f"  {failure}")
        sys.exit(1)
    print("\nevery reference matches an independently derived solution")


if __name__ == "__main__":
    main()
