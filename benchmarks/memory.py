# SPDX-License-Identifier: MIT

"""T-018 — memory footprint benchmark suite.

Run directly:

    python benchmarks/memory.py            # measure + write baseline JSON
    python benchmarks/memory.py --check    # compare vs baseline, exit 1

The ``--check`` mode compares each case's median peak resident-set size
against ``benchmarks/memory_baseline.json`` and fails if any case exceeds
``REGRESSION_THRESHOLD`` (default 1.5x).

# Why subprocess + ru_maxrss (not tracemalloc)

JAX/XLA allocate device buffers (and the XLA C++ heap) outside the
Python heap, so ``tracemalloc`` only catches a sliver of the real
footprint — typically <5 % of what ``simulate_batch`` actually consumes.
We use ``resource.getrusage(RUSAGE_SELF).ru_maxrss``, the OS-reported
peak resident-set size for the process.

But ``ru_maxrss`` is *peak since process start*, not a delta — once a
workload spikes the RSS, the watermark stays high for any subsequent
measurement in the same process. To get a clean per-workload number we
fork a subprocess for each measurement and read its post-mortem peak.
This also isolates JAX's process-global state so e.g. the compile-cache
peak from case A doesn't pollute case B.

# Platform note: ru_maxrss units

* Linux: kilobytes
* macOS: bytes

We detect ``sys.platform`` and convert to MB consistently.

The 7 compile cases mirror T-017's ``compile_time.py`` so the two
benchmarks track the same representative surface.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


N_REPEAT = 3
REGRESSION_THRESHOLD = 1.5
BASELINE_PATH = Path(__file__).parent / "memory_baseline.json"


def _ru_maxrss_to_mb(ru_maxrss: int) -> float:
    """Convert ``ru_maxrss`` to MB across Linux (KB) and macOS (bytes)."""
    if sys.platform == "darwin":
        return ru_maxrss / (1024 * 1024)
    return ru_maxrss / 1024  # Linux + most BSDs report KB


# ── workload scripts (run in subprocess, print "PEAK_MB:<value>") ──────────

# Each workload is a self-contained script. It imports jaxonomy, runs the
# workload, then prints its post-workload ru_maxrss in MB on the LAST line
# as ``PEAK_MB:<float>``. The parent process parses that line.

_PROLOGUE = r"""
import sys, os, resource, gc
sys.path.insert(0, %r)
import jax
import jax.numpy as jnp
import jaxonomy
from jaxonomy import logging as _jx_logging
_jx_logging.set_log_level("WARNING")
"""

_EPILOGUE = r"""
gc.collect()
_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    _peak_mb = _ru / (1024 * 1024)
else:
    _peak_mb = _ru / 1024
print(f"PEAK_MB:{_peak_mb:.3f}")
"""


def _run_subprocess(workload_body: str) -> float:
    """Run ``workload_body`` in a fresh Python; return peak RSS in MB."""
    script = _PROLOGUE % str(_REPO_ROOT) + workload_body + _EPILOGUE
    out = subprocess.check_output(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    text = out.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith("PEAK_MB:"):
            return float(line.split(":", 1)[1])
    raise RuntimeError(
        f"workload did not print PEAK_MB; output was:\n{text}"
    )


# ── batch-decay (per-element) ──────────────────────────────────────────────

_BATCH_DECAY_BODY = r"""
from jaxonomy import DiagramBuilder, SimulatorOptions, simulate_batch
from jaxonomy.library import Gain, Integrator
b = DiagramBuilder()
g = b.add(Gain(gain=-1.0, name="g"))
intg = b.add(Integrator(initial_state=1.0, name="intg"))
b.connect(g.output_ports[0], intg.input_ports[0])
b.connect(intg.output_ports[0], g.input_ports[0])
sys_ = b.build(name="decay_batch")
N = %d
g_values = -jnp.linspace(0.5, 2.5, N)
opts = SimulatorOptions(
    math_backend="jax", ode_solver_method="dopri5", max_major_steps=200,
)
port = sys_["intg"].output_ports[0]
res = simulate_batch(
    sys_, t_span=(0.0, 5.0),
    param_batches={"g.gain": g_values},
    options=opts, recorded_signals={"y": port},
)
del res
gc.collect()
"""


def _measure_batch_decay(n: int) -> float:
    return _run_subprocess(_BATCH_DECAY_BODY % n)


# ── shared workload snippets ───────────────────────────────────────────────

_SHO_CLS = """
from jaxonomy import LeafSystem
class _SHO(LeafSystem):
    def __init__(self, omega=2.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("omega", omega)
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 0.0]), ode=self._ode,
        )
        self.declare_continuous_state_output(name="state")
    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        return jnp.array([v, -(params["omega"] ** 2) * x])
"""

_SCALAR_DECAY_CLS = """
from jaxonomy import LeafSystem
class _ScalarDecay(LeafSystem):
    def __init__(self, a=1.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_continuous_state_output(name="y")
    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state
"""

_LONG_HORIZON_BODY = _SHO_CLS + r"""
from jaxonomy import SimulatorOptions
from jaxonomy.simulation import simulate
sho = _SHO(omega=2.0)
ctx = sho.create_context()
opts = SimulatorOptions(
    math_backend="jax", ode_solver_method="dopri5", max_major_steps=200_000,
)
res = simulate(
    sho, ctx, t_span=(0.0, %f), options=opts,
    recorded_signals={"y": sho.output_ports[0]},
)
del res
gc.collect()
"""


def _measure_long_horizon(t_end: float) -> float:
    return _run_subprocess(_LONG_HORIZON_BODY % t_end)


# ── compile-time peak (mirrors T-017's 7 cases) ────────────────────────────

# Each of these scripts builds + cold-compiles a system once via Benchmark.
# We don't separate compile-only vs sim — the peak watermark catches the
# worst point in the pipeline (which for these workloads is XLA lowering).

_COMPILE_CASE_BODIES = {
    "scalar_exponential_decay": _SCALAR_DECAY_CLS + r"""
from jaxonomy.testing.util import Benchmark
sys_ = _ScalarDecay(a=1.0)
Benchmark(sys_, sim_stop_time=5.0).time_compile_and_sim(N=1)
""",
    "harmonic_oscillator": _SHO_CLS + r"""
from jaxonomy.testing.util import Benchmark
sys_ = _SHO(omega=2.0)
Benchmark(sys_, sim_stop_time=5.0).time_compile_and_sim(N=1)
""",
    "pid_first_order_plant": r"""
from jaxonomy import DiagramBuilder
from jaxonomy.library import Adder, Constant, Integrator, PID
from jaxonomy.testing.util import Benchmark
b = DiagramBuilder()
ref = b.add(Constant(value=1.0, name="ref"))
err = b.add(Adder(2, operators="+-", name="err"))
pid = b.add(PID(kp=1.0, ki=0.5, kd=0.1, n=50.0, name="pid"))
plant = b.add(Integrator(initial_state=0.0, name="plant"))
b.connect(ref.output_ports[0], err.input_ports[0])
b.connect(plant.output_ports[0], err.input_ports[1])
b.connect(err.output_ports[0], pid.input_ports[0])
b.connect(pid.output_ports[0], plant.input_ports[0])
sys_ = b.build(name="pid_plant")
bench = Benchmark(sys_, sim_stop_time=4.0)
bench.time_compile_and_sim(N=1)
""",
    "rc_acausal_dae": r"""
from jaxonomy import DiagramBuilder
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.testing.util import Benchmark
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
b = DiagramBuilder()
b.add(lpf)
sys_ = b.build(name="rc_acausal")
bench = Benchmark(sys_, sim_stop_time=5.0)
bench.time_compile_and_sim(N=1)
""",
    "bouncing_ball_zc": r"""
from jaxonomy.models import BouncingBall
from jaxonomy.testing.util import Benchmark
sys_ = BouncingBall(g=9.81, e=0.8)
ctx = sys_.create_context().with_continuous_state(jnp.array([5.0, 0.0]))
bench = Benchmark(sys_, context=ctx, sim_stop_time=3.0)
bench.time_compile_and_sim(N=1)
""",
    "state_machine_three_state": r"""
from jaxonomy import DiagramBuilder, StateMachineBuilder
from jaxonomy.library import Clock, StateMachine
from jaxonomy.testing.util import Benchmark
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
b = DiagramBuilder()
clk = b.add(Clock(name="clk"))
sm_block = b.add(sm)
b.connect(clk.output_ports[0], sm_block.input_ports[0])
sys_ = b.build(name="sm_root")
bench = Benchmark(sys_, sim_stop_time=3.0)
bench.time_compile_and_sim(N=1)
""",
    "simulate_batch_decay_n10": _BATCH_DECAY_BODY % 10,
}


def _measure_compile_case(name: str) -> float:
    return _run_subprocess(_COMPILE_CASE_BODIES[name])


# ── orchestration ──────────────────────────────────────────────────────────


def _hardware_info():
    import jax  # local import: parent process needs no JAX init
    import jaxonomy
    try:
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        ram_gb = round(ram_bytes / 1e9, 1)
    except (AttributeError, ValueError, OSError):
        ram_gb = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxonomy_version": jaxonomy.__version__,
        "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "kilobytes",
        "ram_gb_total": ram_gb,
    }


def _median_of_n(fn, *args, n=N_REPEAT):
    samples = []
    for i in range(n):
        try:
            samples.append(fn(*args))
        except subprocess.CalledProcessError as e:
            return None, f"subprocess failed (rc={e.returncode}): "\
                         f"{e.output.decode('utf-8', errors='replace')[-300:]}"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
    return statistics.median(samples), samples


def _linear_slope(xs, ys):
    """Simple least-squares slope (y = a + b·x)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else float("nan")


def run_all():
    results = {}

    # 1. per-element batch memory at N=1, 10, 100
    print("# Per-element batch memory")
    batch_sizes = [1, 10, 100]
    batch_peaks = {}
    for n in batch_sizes:
        med, samples = _median_of_n(_measure_batch_decay, n)
        if med is None:
            print(f"  [batch_n={n}] SKIPPED — {samples}")
            results[f"batch_n{n}"] = {"skipped": True, "reason": samples}
        else:
            print(f"  [batch_n={n}] median={med:.1f} MB  samples={samples}")
            results[f"batch_n{n}"] = {
                "peak_mb_median": float(med),
                "peak_mb_samples": [float(x) for x in samples],
                "n_repeat": N_REPEAT,
            }
            batch_peaks[n] = med
    if len(batch_peaks) >= 2:
        slope = _linear_slope(
            list(batch_peaks.keys()), list(batch_peaks.values()),
        )
        print(f"  → mb_per_element ≈ {slope:.3f} MB/elt (linear fit)")
        results["batch_mb_per_element"] = float(slope)

    # 2. long-horizon: 10 s, 100 s, 1000 s
    print("\n# Long-horizon growth (harmonic oscillator)")
    horizons = [10.0, 100.0, 1000.0]
    horizon_peaks = {}
    for tf in horizons:
        med, samples = _median_of_n(_measure_long_horizon, tf)
        key = f"long_horizon_t{int(tf)}"
        if med is None:
            print(f"  [t_end={tf}] SKIPPED — {samples}")
            results[key] = {"skipped": True, "reason": samples}
        else:
            print(f"  [t_end={tf}] median={med:.1f} MB  samples={samples}")
            results[key] = {
                "peak_mb_median": float(med),
                "peak_mb_samples": [float(x) for x in samples],
                "n_repeat": N_REPEAT,
            }
            horizon_peaks[tf] = med
    if len(horizon_peaks) >= 2:
        slope = _linear_slope(
            list(horizon_peaks.keys()), list(horizon_peaks.values()),
        )
        print(f"  → mb_per_sim_second ≈ {slope:.4f} MB/s")
        results["long_horizon_mb_per_sim_second"] = float(slope)

    # 3. cold-compile peak across the 7 T-017 cases
    print("\n# Cold-compile peak (mirrors T-017 cases)")
    for name in _COMPILE_CASE_BODIES:
        med, samples = _median_of_n(_measure_compile_case, name)
        key = f"compile_{name}"
        if med is None:
            print(f"  [{name}] SKIPPED — {samples}")
            results[key] = {"skipped": True, "reason": samples}
        else:
            print(f"  [{name}] median={med:.1f} MB  samples={samples}")
            results[key] = {
                "peak_mb_median": float(med),
                "peak_mb_samples": [float(x) for x in samples],
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
    print(f"\n## Memory regression check  (threshold {REGRESSION_THRESHOLD:.2f}x)\n")
    print(f"{'case':<40} {'baseline':>10} {'measured':>10} {'ratio':>8} status")
    print("-" * 85)
    for name, case in measured.items():
        if not isinstance(case, dict):
            continue  # scalar derived metrics (mb_per_element / mb_per_sim_second)
        if case.get("skipped"):
            print(f"{name:<40} {'-':>10} {'-':>10} {'-':>8} SKIPPED")
            continue
        if name not in base_cases or not isinstance(base_cases[name], dict) \
                or base_cases[name].get("skipped"):
            print(f"{name:<40} {'-':>10} "
                  f"{case['peak_mb_median']:>10.1f} {'-':>8} NEW")
            continue
        base = base_cases[name]["peak_mb_median"]
        meas = case["peak_mb_median"]
        ratio = meas / base if base > 0 else float("inf")
        status = "OK" if ratio <= REGRESSION_THRESHOLD else "REGRESSION"
        if status == "REGRESSION":
            ok = False
        print(f"{name:<40} {base:>10.1f} {meas:>10.1f} {ratio:>8.2f} {status}")
    return ok


def _emit_github_summary(measured):
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if not sp:
        return
    lines = ["## T-018 memory benchmark", "",
             "| case | peak (MB) |", "| --- | --- |"]
    for name, case in measured.items():
        if not isinstance(case, dict):
            lines.append(f"| {name} | {case:.3f} |")
            continue
        if case.get("skipped"):
            lines.append(f"| {name} | SKIPPED |")
        else:
            lines.append(f"| {name} | {case['peak_mb_median']:.1f} |")
    Path(sp).open("a").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare against baseline; exit 1 on regression")
    parser.add_argument("--out", type=Path, default=BASELINE_PATH,
                        help="path to write baseline JSON (ignored with --check)")
    args = parser.parse_args()

    print(f"Running memory benchmark suite\n  N_REPEAT = {N_REPEAT}"
          f"\n  ru_maxrss_unit = "
          f"{'bytes (macOS)' if sys.platform == 'darwin' else 'KB (Linux)'}\n")
    measured = run_all()
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
