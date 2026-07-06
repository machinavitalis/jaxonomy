# SPDX-License-Identifier: MIT

"""T-017 — JIT compile-time benchmark suite.

Run directly:

    python benchmarks/compile_time.py            # measure + write baseline JSON
    python benchmarks/compile_time.py --check    # compare vs baseline, exit 1

The ``--check`` mode compares each case's median compile time against
``benchmarks/compile_time_baseline.json`` and fails if any case exceeds
``REGRESSION_THRESHOLD`` (default 1.5x).  Threshold is permissive because
sub-second compiles on CI runners have substantial jitter.

Each iteration calls ``jax.clear_caches()`` so JAX's in-memory trace cache
does not turn the second iteration into a near-zero "compile".  All cases
use ``jaxonomy.testing.Benchmark`` to split compile from sim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jaxonomy  # noqa: E402
from jaxonomy import logging as _jx_logging  # noqa: E402

_jx_logging.set_log_level("WARNING")  # suppress per-step INFO spam

from jaxonomy import (  # noqa: E402
    DiagramBuilder,
    LeafSystem,
    SimulatorOptions,
    StateMachineBuilder,
    simulate_batch,
)
from jaxonomy.library import (  # noqa: E402
    Adder, Clock, Constant, Gain, Integrator, PID, StateMachine,
)
from jaxonomy.models import BouncingBall  # noqa: E402
from jaxonomy.testing.util import Benchmark  # noqa: E402


N_REPEAT = 3
REGRESSION_THRESHOLD = 1.5
BASELINE_PATH = Path(__file__).parent / "compile_time_baseline.json"


# ── case builders ───────────────────────────────────────────────────────────


class _ScalarDecay(LeafSystem):
    """dy/dt = -a * y."""
    def __init__(self, a=1.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_continuous_state_output(name="y")

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


class _SHO(LeafSystem):
    """2-D harmonic oscillator."""
    def __init__(self, omega=2.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("omega", omega)
        self.declare_continuous_state(default_value=jnp.array([1.0, 0.0]), ode=self._ode)
        self.declare_continuous_state_output(name="state")

    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        return jnp.array([v, -(params["omega"] ** 2) * x])


def _build_pid_plant():
    b = DiagramBuilder()
    ref = b.add(Constant(value=1.0, name="ref"))
    err = b.add(Adder(2, operators="+-", name="err"))
    pid = b.add(PID(kp=1.0, ki=0.5, kd=0.1, n=50.0, name="pid"))
    plant = b.add(Integrator(initial_state=0.0, name="plant"))
    b.connect(ref.output_ports[0], err.input_ports[0])
    b.connect(plant.output_ports[0], err.input_ports[1])
    b.connect(err.output_ports[0], pid.input_ports[0])
    b.connect(pid.output_ports[0], plant.input_ports[0])
    return b.build(name="pid_plant"), 4.0


def _build_rc_acausal():
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(ev, name="c1", C=1.0,
                        initial_voltage=0.0, initial_voltage_fixed=True)
    ref1 = elec.Ground(ev, name="ref1")
    ad.connect(v1, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", v1, "n")
    ad.connect(v1, "n", ref1, "p")
    lpf = AcausalCompiler(ev, ad)()
    builder = DiagramBuilder()
    builder.add(lpf)
    return builder.build(name="rc_acausal"), 5.0


def _build_bouncing_ball():
    model = BouncingBall(g=9.81, e=0.8)
    ctx = model.create_context().with_continuous_state(jnp.array([5.0, 0.0]))
    return model, 3.0, ctx


def _build_state_machine():
    smb = StateMachineBuilder()
    idle = smb.add_state("idle")
    running = smb.add_state("running")
    done = smb.add_state("done")
    smb.set_initial_state(idle)
    smb.add_transition(idle, running, guard="x > 0.5", actions=["y = 1.0"])
    smb.add_transition(running, done, guard="x > 1.5", actions=["y = 2.0"])
    sm_default = smb.build(name="three_state_sm_template")
    sm = StateMachine(
        sm_data=sm_default._sm,
        inputs=list(sm_default._input_names),
        outputs=list(sm_default._output_names),
        dt=0.05, time_mode="discrete",
        name="three_state_sm", accelerate_with_jax=False,
    )
    builder = DiagramBuilder()
    clk = builder.add(Clock(name="clk"))
    sm_block = builder.add(sm)
    builder.connect(clk.output_ports[0], sm_block.input_ports[0])
    return builder.build(name="sm_root"), 3.0


def _build_decay_for_batch():
    b = DiagramBuilder()
    g = b.add(Gain(gain=-1.0, name="g"))
    intg = b.add(Integrator(initial_state=1.0, name="intg"))
    b.connect(g.output_ports[0], intg.input_ports[0])
    b.connect(intg.output_ports[0], g.input_ports[0])
    return b.build(name="decay_batch")


# ── runners ─────────────────────────────────────────────────────────────────


def _clear_jax_caches():
    try:
        jax.clear_caches()
    except Exception:  # noqa: BLE001 — older JAX builds
        pass


def _measure_simple(system, tf, context=None):
    """Return (compile_time_seconds, sim_time_seconds) for one cold run."""
    _clear_jax_caches()
    bench = Benchmark(system, context=context, sim_stop_time=tf)
    cs = bench.time_compile_and_sim(N=1)
    s = bench.time_sim(N=1)
    return max(cs - s, 0.0), s


def _measure_batch_decay(tf=5.0, n=10):
    sys = _build_decay_for_batch()
    g_values = -jnp.linspace(0.5, 2.5, n)
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5", max_major_steps=200,
    )
    port = sys["intg"].output_ports[0]

    def _run():
        return simulate_batch(
            sys, t_span=(0.0, tf),
            param_batches={"g.gain": g_values},
            options=opts, recorded_signals={"y": port},
        )

    _clear_jax_caches()
    t0 = time.perf_counter(); _run(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(); warm = time.perf_counter() - t0
    return max(cold - warm, 0.0), warm


CASES: dict[str, Callable[[], tuple[float, float]]] = {
    "scalar_exponential_decay": lambda: _measure_simple(_ScalarDecay(a=1.0), 5.0),
    "harmonic_oscillator":      lambda: _measure_simple(_SHO(omega=2.0), 5.0),
    "pid_first_order_plant":    lambda: _measure_simple(*_build_pid_plant()),
    "rc_acausal_dae":           lambda: _measure_simple(*_build_rc_acausal()),
    "bouncing_ball_zc":         lambda: (
        lambda s, tf, c: _measure_simple(s, tf, context=c)
    )(*_build_bouncing_ball()),
    "state_machine_three_state": lambda: _measure_simple(*_build_state_machine()),
    "simulate_batch_decay_n10":   lambda: _measure_batch_decay(tf=5.0, n=10),
    "simulate_batch_decay_n100":  lambda: _measure_batch_decay(tf=5.0, n=100),
    "simulate_batch_decay_n1000": lambda: _measure_batch_decay(tf=5.0, n=1000),
}


# ── orchestration ───────────────────────────────────────────────────────────


def _hardware_info():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxonomy_version": jaxonomy.__version__,
        "jax_default_backend": jax.default_backend(),
    }


def run_all_cases():
    results = {}
    for name, fn in CASES.items():
        compile_times, sim_times = [], []
        skipped = None
        for i in range(N_REPEAT):
            try:
                ct, st = fn()
            except Exception as e:  # noqa: BLE001 — honest fallback
                skipped = f"{type(e).__name__}: {e}"
                break
            compile_times.append(ct)
            sim_times.append(st)
            print(f"  [{name}] run {i + 1}/{N_REPEAT}  "
                  f"compile={ct:.3f}s  sim={st:.3f}s", flush=True)
        if skipped is not None:
            print(f"  [{name}] SKIPPED — {skipped}", flush=True)
            results[name] = {"skipped": True, "reason": skipped}
            continue
        results[name] = {
            "compile_time_median_s": float(statistics.median(compile_times)),
            "sim_time_median_s":     float(statistics.median(sim_times)),
            "compile_times_s":       [float(x) for x in compile_times],
            "n_repeat": N_REPEAT,
        }
    return results


def _check_against_baseline(measured, baseline_path):
    if not baseline_path.exists():
        print(f"baseline file {baseline_path} missing — cannot --check")
        return False
    with baseline_path.open() as f:
        baseline = json.load(f)
    base_cases = baseline.get("cases", {})
    ok = True
    print(f"\n## Compile-time regression check  (threshold {REGRESSION_THRESHOLD:.2f}x)\n")
    print(f"{'case':<35} {'baseline':>10} {'measured':>10} {'ratio':>8} status")
    print("-" * 80)
    for name, case in measured.items():
        if case.get("skipped"):
            print(f"{name:<35} {'-':>10} {'-':>10} {'-':>8} SKIPPED")
            continue
        if name not in base_cases or base_cases[name].get("skipped"):
            print(f"{name:<35} {'-':>10} "
                  f"{case['compile_time_median_s']:>10.3f} {'-':>8} NEW")
            continue
        base = base_cases[name]["compile_time_median_s"]
        meas = case["compile_time_median_s"]
        ratio = meas / base if base > 0 else float("inf")
        status = "OK" if ratio <= REGRESSION_THRESHOLD else "REGRESSION"
        if status == "REGRESSION":
            ok = False
        print(f"{name:<35} {base:>10.3f} {meas:>10.3f} {ratio:>8.2f} {status}")
    return ok


def _emit_github_summary(measured):
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if not sp:
        return
    lines = ["## T-017 compile-time benchmark", "",
             "| case | compile (s) | sim (s) |", "| --- | --- | --- |"]
    for name, case in measured.items():
        if case.get("skipped"):
            lines.append(f"| {name} | SKIPPED | {case['reason']} |")
        else:
            lines.append(
                f"| {name} | {case['compile_time_median_s']:.3f} | "
                f"{case['sim_time_median_s']:.3f} |"
            )
    Path(sp).open("a").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare against baseline; exit 1 on regression")
    parser.add_argument("--out", type=Path, default=BASELINE_PATH,
                        help="path to write baseline JSON (ignored with --check)")
    args = parser.parse_args()

    print(f"Running compile-time benchmark suite\n  N_REPEAT = {N_REPEAT}"
          f"\n  cases    = {len(CASES)}\n")
    measured = run_all_cases()
    payload = {
        "hardware": _hardware_info(),
        "regression_threshold": REGRESSION_THRESHOLD,
        "cases": measured,
    }

    if args.check:
        ok = _check_against_baseline(measured, BASELINE_PATH)
        _emit_github_summary(measured)
        return 0 if ok else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    print(f"\nWrote baseline to {args.out}")
    _emit_github_summary(measured)
    return 0


if __name__ == "__main__":
    sys.exit(main())
