# SPDX-License-Identifier: MIT

"""T-019 — Public benchmark suite with published numbers.

Run directly:

    python benchmarks/public.py                       # measure + write baseline JSON
    python benchmarks/public.py --check               # compare vs baseline, exit 1
    python benchmarks/public.py --device gpu          # run on a CUDA GPU (T-019a)
    python benchmarks/public.py --device gpu \\
        --update-baseline gpu_t4                      # populate the gpu_t4 column

Five CPU-runnable problems: cartpole throughput, quadruped at N=1000
parallel envs (simplified 4-leg, 8 states, simulate_batch),
articulated-quadruped throughput (real 12-DoF MJX body, T-019b),
spring-damper system-ID convergence, and linearization vs analytic
Jacobians.  Numbers committed to public_baseline.json with hardware
fingerprint; --check fails on +50% regression.  Competitor comparison
context in docs/benchmarks.md.

The baseline JSON carries per-device columns under each case (e.g.
``cases.cartpole_throughput.cpu``, ``cases.cartpole_throughput.gpu_t4``).
Older flat shape (case-level metric keys) is still read by ``--check``
for backwards compatibility — see ``_resolve_baseline_case``.

T-019a contract for populating GPU columns lives in
``docs/benchmarks.md`` ("GPU runner contract").
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── T-019a device selection ────────────────────────────────────────────────
# Honoured before `import jax` so JAX's lazy backend init picks the right
# platform.  Both the CLI flag (``--device gpu``) and the env var
# ``JAXONOMY_BENCH_DEVICE`` route through this code path; the CLI flag wins
# by re-exporting the env var early in main().  See docs/benchmarks.md.
_DEVICE_ENV_VAR = "JAXONOMY_BENCH_DEVICE"
_DEVICE_REQUESTED = os.environ.get(_DEVICE_ENV_VAR, "").strip().lower() or None
if _DEVICE_REQUESTED in {"gpu", "cuda"}:
    # JAX accepts "cuda" or "gpu" for NVIDIA; "gpu" is the documented alias.
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
elif _DEVICE_REQUESTED == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jaxonomy  # noqa: E402
from jaxonomy import logging as _jx_logging  # noqa: E402

_jx_logging.set_log_level("WARNING")

from jaxonomy import (  # noqa: E402
    DiagramBuilder,
    LeafSystem,
    SimulatorOptions,
    linearize,
    simulate,
    simulate_batch,
)
from jaxonomy.models import CartPole  # noqa: E402

N_REPEAT = 3
REGRESSION_THRESHOLD = 1.5
BASELINE_PATH = Path(__file__).parent / "public_baseline.json"

# Quadruped batch size — drop to N_QUADRUPED_FALLBACK on slow hosts /
# low memory.  See docs/benchmarks.md for the GPU-scaled story.
N_QUADRUPED = 1000
N_QUADRUPED_FALLBACK = 100


# ── shared helpers ─────────────────────────────────────────────────────────


def _clear_jax_caches():
    try:
        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass


# ── 1. Cartpole throughput ─────────────────────────────────────────────────


def _measure_cartpole_throughput(t_end=10.0):
    """Return dict with compile_s, sim_s, wall_per_simsec."""
    _clear_jax_caches()
    plant = CartPole(
        x0=jnp.array([0.0, 0.1, 0.0, 0.0]),  # x, theta, dot_x, dot_theta
        m_c=1.0, m_p=0.1, L=0.5, g=9.81,
        full_state_output=True, name="cartpole",
    )
    plant.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = plant.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        max_major_steps=1000, return_context=False,
    )
    rec = {"x": plant.output_ports[0]}

    def _run():
        return simulate(
            plant, ctx, t_span=(0.0, t_end),
            options=opts, recorded_signals=rec,
        )

    t0 = time.perf_counter(); _run(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(); warm = time.perf_counter() - t0
    return {
        "compile_s": float(max(cold - warm, 0.0)),
        "sim_s": float(warm),
        "wall_per_simsec": float(warm / t_end),
        "t_end_s": t_end,
    }


# ── 2. Quadruped throughput at N=1000 parallel envs ────────────────────────


class _QuadrupedSingleJoint(LeafSystem):
    """Simplified quadruped: 4 independent damped pendulum legs (8 states).

    ODE per leg: theta_dot = omega; omega_dot = -(g/L) sin(theta) - b*omega.
    Not a multi-body articulated quadruped — see docs/benchmarks.md.
    """

    def __init__(self, x0=None, g=9.81, L=0.3, b=0.5, **kwargs):
        super().__init__(**kwargs)
        if x0 is None:
            x0 = jnp.array([0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, 0.0])
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("b", b)
        self.declare_continuous_state(default_value=x0, ode=self._ode)
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        x = state.continuous_state
        g, L, b = params["g"], params["L"], params["b"]
        # 4 legs, two states each
        theta = x[0::2]
        omega = x[1::2]
        omega_dot = -(g / L) * jnp.sin(theta) - b * omega
        # interleave back: [θ̇₀, ω̇₀, θ̇₁, ω̇₁, …]
        return jnp.stack([omega, omega_dot], axis=-1).reshape(-1)


def _measure_quadruped_throughput(n_env=N_QUADRUPED, t_end=5.0):
    """N independent quadruped sims via simulate_batch."""
    _clear_jax_caches()
    db = DiagramBuilder()
    db.add(_QuadrupedSingleJoint(name="quad"))
    sys = db.build(name="quadruped_root")
    # Sweep over per-env damping so each element is genuinely distinct.
    b_values = jnp.linspace(0.3, 0.7, n_env)
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        max_major_steps=500, return_context=False,
    )
    rec = {"x": sys["quad"].output_ports[0]}

    def _run():
        return simulate_batch(
            sys, t_span=(0.0, t_end),
            param_batches={"quad.b": b_values},
            options=opts, recorded_signals=rec,
        )

    t0 = time.perf_counter(); _run(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(); warm = time.perf_counter() - t0
    env_secs = n_env * t_end
    return {
        "n_env": int(n_env),
        "t_end_s": t_end,
        "compile_s": float(max(cold - warm, 0.0)),
        "sim_s": float(warm),
        "env_seconds_per_wall_second": float(env_secs / warm)
        if warm > 0 else float("inf"),
    }


# ── 2b. Articulated-quadruped throughput (T-019b, MJX 12-DoF body) ─────────


# Hand-authored MJCF, 12-DoF (3 per leg × 4 legs) + free-joint trunk.
# Lives next to the other MuJoCo example assets — no mesh dependencies, so
# the model loads cleanly without any external download step.
_ARTICULATED_QUADRUPED_XML = (
    _REPO_ROOT / "docs" / "examples" / "mujoco" / "assets"
    / "articulated_quadruped.xml"
)


def _measure_articulated_quadruped_throughput(t_end=5.0):
    """Single-env continuous-time MJX rollout of an articulated quadruped.

    Honest scope: this measures *physics throughput on a multi-body
    articulated body with floor contact*, **not** locomotion / RL env-step
    rate (no controller, no learned policy).  Single-env (N=1) — the
    batched discrete-mode MJX variant is the companion benchmark
    ``articulated_quadruped_batched_throughput`` (T-019b-followup-batched).
    """
    # Imported lazily — mujoco / mjx are heavy and not always installed.
    from jaxonomy.library.mujoco import MJX

    _clear_jax_caches()
    db = DiagramBuilder()
    block = MJX(
        file_name=str(_ARTICULATED_QUADRUPED_XML),
        dt=None,  # continuous-time, jaxonomy's adaptive solver
        key_frame_0="stand",
        name="quad",
    )
    # nu may be 0 (no actuators) — fix_value still needs a vector of shape (nu,)
    block.input_ports[0].fix_value(jnp.zeros(block.nu))
    db.add(block)
    sys = db.build(name="articulated_quadruped_root")

    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        max_major_steps=1000, return_context=False,
    )
    ctx = sys.create_context()

    def _run():
        return simulate(sys, ctx, t_span=(0.0, t_end), options=opts)

    t0 = time.perf_counter(); _run(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(); warm = time.perf_counter() - t0
    return {
        "n_env": 1,
        "t_end_s": t_end,
        "nq": int(block.nq),
        "nv": int(block.nv),
        "nu": int(block.nu),
        "compile_s": float(max(cold - warm, 0.0)),
        "sim_s": float(warm),
        "wall_per_simsec": float(warm / t_end) if t_end > 0 else float("inf"),
        "env_seconds_per_wall_second": float(t_end / warm)
        if warm > 0 else float("inf"),
    }


# ── 2c. Articulated-quadruped batched throughput (T-019b-followup-batched) ─

# Batched-MJX timestep: matches the underlying MJX block's discrete-mode
# integrator dt.  Smaller dt → more steps → more wall time per env-second.
_ARTICULATED_QUADRUPED_BATCH_DT = 0.002

# T-019b-followup-batched: pick a batch size that fits in <16 GB RAM
# since CI hosts are tight; GPU runs can lift this much higher.
N_QUADRUPED_BATCHED = 64
N_QUADRUPED_BATCHED_FALLBACK = 16


def _measure_articulated_quadruped_batched_throughput(
    n_env=N_QUADRUPED_BATCHED, t_end=1.0,
):
    """Batched discrete-mode MJX rollout of the articulated quadruped.

    Companion to ``_measure_articulated_quadruped_throughput``: this is
    the closer cousin to the RL ``env.step`` rate metric.  Each batch
    element is an independent rollout of the same 12-DoF body driven by
    a per-batch ``Constant`` control source — the hand-authored XML has
    no actuators (``nu == 0``), so the constant value is dynamics-inert
    and serves only as a dynamic-parameter handle for ``simulate_batch``
    (``ctrl.value``).  Unlocked by the contact-array dtype fix in
    ``MJX._step_cache_cb`` (T-019b-followup-batched).
    """
    # Imported lazily — mujoco / mjx are heavy and not always installed.
    from jaxonomy.library.mujoco import MJX
    from jaxonomy.library.primitives import Constant

    _clear_jax_caches()
    db = DiagramBuilder()
    mjx_blk = MJX(
        file_name=str(_ARTICULATED_QUADRUPED_XML),
        dt=_ARTICULATED_QUADRUPED_BATCH_DT,  # discrete-mode, MJX solver
        key_frame_0="stand",
        name="quad",
    )
    # ``Constant`` requires a non-empty value; when nu == 0 we still
    # allocate shape (1,) but skip the connect.
    nu_eff = max(mjx_blk.nu, 1)
    ctrl_blk = Constant(value=jnp.zeros(nu_eff), name="ctrl")
    db.add(mjx_blk)
    db.add(ctrl_blk)
    if mjx_blk.nu > 0:
        db.connect(ctrl_blk.output_ports[0], mjx_blk.input_ports[0])
    else:
        mjx_blk.input_ports[0].fix_value(jnp.zeros(0))
    sys = db.build(name="articulated_quadruped_batched_root")

    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        max_major_steps=int(t_end / _ARTICULATED_QUADRUPED_BATCH_DT * 1.5) + 10,
        return_context=False,
    )
    rec = {"qpos": mjx_blk.output_ports[0]}
    batch_vals = jnp.tile(jnp.zeros(nu_eff)[None, :], (n_env, 1))

    def _run():
        return simulate_batch(
            sys, t_span=(0.0, t_end),
            param_batches={"ctrl.value": batch_vals},
            options=opts, recorded_signals=rec,
        )

    t0 = time.perf_counter(); _run(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(); warm = time.perf_counter() - t0
    env_secs = n_env * t_end
    return {
        "n_env": int(n_env),
        "t_end_s": float(t_end),
        "dt": float(_ARTICULATED_QUADRUPED_BATCH_DT),
        "nq": int(mjx_blk.nq),
        "nv": int(mjx_blk.nv),
        "nu": int(mjx_blk.nu),
        "compile_s": float(max(cold - warm, 0.0)),
        "sim_s": float(warm),
        "env_seconds_per_wall_second": float(env_secs / warm)
        if warm > 0 else float("inf"),
    }


# ── 3. System-identification convergence (spring-damper) ────────────────────


class _SpringDamper(LeafSystem):
    """m * x'' + b * x' + k * x = u, output = x."""

    def __init__(self, m=1.0, b=0.5, k=2.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("m", m)
        self.declare_dynamic_parameter("b", b)
        self.declare_dynamic_parameter("k", k)
        self.declare_input_port(name="u")
        self.declare_continuous_state(
            default_value=jnp.array([0.0, 0.0]), ode=self._ode,
        )
        self.declare_output_port(
            lambda t, s, *u, **p: s.continuous_state[0],
            name="x", requires_inputs=False,
        )

    def _ode(self, time, state, *inputs, **params):
        x, v = state.continuous_state
        (u,) = inputs
        u_scalar = jnp.asarray(u).reshape(())
        m, b, k = params["m"], params["b"], params["k"]
        return jnp.array([v, (u_scalar - b * v - k * x) / m])


def _measure_sysid_convergence(seed=0, max_iter=50, target_mse=1e-4):
    """Fit (b, k) of a spring-damper to noisy synthetic data via L-BFGS-B.

    Mass is anchored — three-param identifiability is a separate study.
    Convergence target: parameter MSE below ``target_mse``.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0.0, 5.0, 200)

    # synthetic ground truth ────────────────────────────────────────
    m_true, b_true, k_true = 1.0, 0.5, 2.0
    plant_true = _SpringDamper(m=m_true, b=b_true, k=k_true, name="plant")
    plant_true.input_ports[0].fix_value(jnp.array([1.0]))  # step input
    ctx_true = plant_true.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        max_major_steps=400, return_context=False,
    )
    rec = {"x": plant_true.output_ports[0]}
    res_true = simulate(
        plant_true, ctx_true, t_span=(0.0, 5.0),
        options=opts, recorded_signals=rec,
    )
    # interpolate truth onto the data grid + add noise
    t_true = np.asarray(res_true.time)
    x_true = np.asarray(res_true.outputs["x"])
    y_data = np.interp(t_eval, t_true, x_true) \
        + 0.01 * rng.standard_normal(len(t_eval))

    # build a closure that evaluates loss for given (b, k) ──────────
    def _simulate_with(b_v, k_v):
        plant = _SpringDamper(m=m_true, b=float(b_v), k=float(k_v), name="plant")
        plant.input_ports[0].fix_value(jnp.array([1.0]))
        ctx = plant.create_context()
        res = simulate(
            plant, ctx, t_span=(0.0, 5.0),
            options=opts, recorded_signals={"x": plant.output_ports[0]},
        )
        return np.asarray(res.time), np.asarray(res.outputs["x"])

    def _loss(theta):
        b_v, k_v = float(theta[0]), float(theta[1])
        t_sim, x_sim = _simulate_with(b_v, k_v)
        x_interp = np.interp(t_eval, t_sim, x_sim)
        return float(np.mean((x_interp - y_data) ** 2))

    theta0 = np.array([1.5, 0.5])  # bad init: 3x b_true, 0.25x k_true
    t0 = time.perf_counter()
    opt_res = minimize(
        _loss, theta0, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-7},
    )
    wall = time.perf_counter() - t0

    fitted = opt_res.x
    truth = np.array([b_true, k_true])
    param_mse = float(np.mean((fitted - truth) ** 2))
    return {
        "n_iter": int(opt_res.nit),
        "n_fev": int(opt_res.nfev),
        "wall_s": float(wall),
        "final_loss": float(opt_res.fun),
        "param_mse": param_mse,
        "fitted_b": float(fitted[0]),
        "fitted_k": float(fitted[1]),
        "true_b": float(b_true),
        "true_k": float(k_true),
        "converged": bool(param_mse < target_mse),
        "target_mse": float(target_mse),
        "method": "L-BFGS-B",
    }


# ── 4. Linearization vs analytic baseline ─────────────────────────────────


class _NonlinearOscillator(LeafSystem):
    """``x'' = -k*x - b*x' - alpha*x**3 + u``; linearize at origin drops the
    cubic and yields A=[[0,1],[-k,-b]], B=[[0],[1]], C=[[1,0]], D=[[0]]."""

    def __init__(self, k=2.0, b=0.5, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("k", k)
        self.declare_dynamic_parameter("b", b)
        self.declare_dynamic_parameter("alpha", alpha)
        self.declare_input_port(name="u")
        self.declare_continuous_state(
            default_value=jnp.array([0.0, 0.0]), ode=self._ode,
        )
        self.declare_output_port(
            lambda t, s, *u, **p: s.continuous_state[0],
            name="x", requires_inputs=False,
        )

    def _ode(self, time, state, *inputs, **params):
        x, v = state.continuous_state
        (u,) = inputs
        u_scalar = jnp.asarray(u).reshape(())
        k, b, a = params["k"], params["b"], params["alpha"]
        return jnp.array([v, u_scalar - k * x - b * v - a * x ** 3])


def _measure_linearization(k=2.0, b=0.5, alpha=1.0):
    """Time + numerical-accuracy comparison vs analytic (A, B, C, D)."""
    _clear_jax_caches()
    plant = _NonlinearOscillator(k=k, b=b, alpha=alpha, name="osc")
    plant.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = plant.create_context()

    # warm jit cache once
    lin = linearize(plant, ctx)
    jax.block_until_ready(lin.A)

    def _one():
        t0 = time.perf_counter()
        ls = linearize(plant, ctx)
        jax.block_until_ready(ls.A)
        return time.perf_counter() - t0, ls

    times = []
    for _ in range(N_REPEAT):
        t, ls = _one()
        times.append(t)

    refs = (
        np.array([[0.0, 1.0], [-k, -b]]),  # A
        np.array([[0.0], [1.0]]),           # B
        np.array([[1.0, 0.0]]),             # C
        np.array([[0.0]]),                  # D
    )
    got = (np.asarray(ls.A), np.asarray(ls.B), np.asarray(ls.C), np.asarray(ls.D))
    err = max(float(np.max(np.abs(g - r))) for g, r in zip(got, refs))
    return {
        "linearize_warm_median_s": float(statistics.median(times)),
        "linearize_warm_samples_s": [float(x) for x in times],
        "max_abs_error_vs_analytic": err,
        "matches_analytic_within_1e_6": bool(err < 1e-6),
        "n_states": 2, "n_inputs": 1, "n_outputs": 1,
    }


# ── orchestration ──────────────────────────────────────────────────────────


def _hardware_info():
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
        "jax_default_backend": jax.default_backend(),
        "ram_gb_total": ram_gb,
    }


def _safe_run(label, fn):
    """Wrap fn in try/except — return (result_or_None, skip_reason_or_None)."""
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def run_all_cases(quadruped_n=N_QUADRUPED):
    results = {}

    # 1. Cartpole — median of N_REPEAT cold compiles
    print("\n# Cartpole throughput (single env, t_end=10s)")
    samples = []
    for i in range(N_REPEAT):
        r, err = _safe_run("cartpole", _measure_cartpole_throughput)
        if err:
            print(f"  [cartpole] SKIPPED — {err}")
            results["cartpole_throughput"] = {"skipped": True, "reason": err}
            break
        print(f"  run {i + 1}/{N_REPEAT}  compile={r['compile_s']:.3f}s  "
              f"sim={r['sim_s']:.4f}s  wall/simsec={r['wall_per_simsec']:.4e}")
        samples.append(r)
    if samples:
        med = lambda k: float(statistics.median([s[k] for s in samples]))
        results["cartpole_throughput"] = {
            "compile_s_median": med("compile_s"),
            "sim_s_median": med("sim_s"),
            "wall_per_simsec_median": med("wall_per_simsec"),
            "t_end_s": samples[0]["t_end_s"], "n_repeat": len(samples),
        }

    # 2. Quadruped — try N_QUADRUPED, fall back to N_QUADRUPED_FALLBACK
    print(f"\n# Quadruped throughput (N={quadruped_n} parallel envs, t_end=5s)")
    r, err = _safe_run(
        "quadruped", lambda: _measure_quadruped_throughput(n_env=quadruped_n))
    if err and quadruped_n > N_QUADRUPED_FALLBACK:
        print(f"  N={quadruped_n} failed ({err}); retrying N={N_QUADRUPED_FALLBACK}")
        r, err = _safe_run(
            "quadruped",
            lambda: _measure_quadruped_throughput(n_env=N_QUADRUPED_FALLBACK))
    if err:
        print(f"  [quadruped] SKIPPED — {err}")
        results["quadruped_throughput"] = {"skipped": True, "reason": err}
    else:
        print(f"  N={r['n_env']}  compile={r['compile_s']:.3f}s  "
              f"sim={r['sim_s']:.3f}s  "
              f"env-s/wall-s={r['env_seconds_per_wall_second']:.1f}")
        results["quadruped_throughput"] = r

    # 2b. Articulated quadruped (T-019b) — single env, real 12-DoF MJX body.
    print("\n# Articulated quadruped throughput "
          "(N=1 MJX continuous-time, t_end=5s)")
    r, err = _safe_run(
        "articulated_quadruped",
        lambda: _measure_articulated_quadruped_throughput(t_end=5.0))
    if err:
        print(f"  [articulated_quadruped] SKIPPED — {err}")
        results["articulated_quadruped_throughput"] = {
            "skipped": True, "reason": err}
    else:
        print(f"  N={r['n_env']}  nq={r['nq']} nv={r['nv']} nu={r['nu']}  "
              f"compile={r['compile_s']:.3f}s  sim={r['sim_s']:.3f}s  "
              f"wall/simsec={r['wall_per_simsec']:.4f}  "
              f"env-s/wall-s={r['env_seconds_per_wall_second']:.3f}")
        results["articulated_quadruped_throughput"] = r

    # 2c. Articulated quadruped batched (T-019b-followup-batched) —
    # discrete-mode MJX over N parallel rollouts via simulate_batch.
    # Falls back to a smaller batch size on memory-tight hosts.
    print(f"\n# Articulated quadruped batched throughput "
          f"(N={N_QUADRUPED_BATCHED} discrete-mode MJX, t_end=1s)")
    r, err = _safe_run(
        "articulated_quadruped_batched",
        lambda: _measure_articulated_quadruped_batched_throughput(
            n_env=N_QUADRUPED_BATCHED, t_end=1.0))
    if err:
        print(f"  N={N_QUADRUPED_BATCHED} failed ({err}); retrying "
              f"N={N_QUADRUPED_BATCHED_FALLBACK}")
        r, err = _safe_run(
            "articulated_quadruped_batched",
            lambda: _measure_articulated_quadruped_batched_throughput(
                n_env=N_QUADRUPED_BATCHED_FALLBACK, t_end=1.0))
    if err:
        print(f"  [articulated_quadruped_batched] SKIPPED — {err}")
        results["articulated_quadruped_batched_throughput"] = {
            "skipped": True, "reason": err}
    else:
        print(f"  N={r['n_env']}  nq={r['nq']} nv={r['nv']} dt={r['dt']}  "
              f"compile={r['compile_s']:.3f}s  sim={r['sim_s']:.3f}s  "
              f"env-s/wall-s={r['env_seconds_per_wall_second']:.2f}")
        results["articulated_quadruped_batched_throughput"] = r

    # 3. System-ID convergence — single fixed-seed run (deterministic)
    print("\n# System-ID convergence (spring-damper, fit (b, k))")
    r, err = _safe_run("sysid", lambda: _measure_sysid_convergence(seed=0))
    if err:
        print(f"  [sysid] SKIPPED — {err}")
        results["sysid_convergence"] = {"skipped": True, "reason": err}
    else:
        print(f"  iters={r['n_iter']}  fev={r['n_fev']}  "
              f"wall={r['wall_s']:.2f}s  param_mse={r['param_mse']:.2e}  "
              f"converged={r['converged']}")
        results["sysid_convergence"] = r

    # 4. Linearization vs analytic
    print("\n# Linearization vs analytic (nonlinear oscillator at x=0)")
    r, err = _safe_run("linearize", _measure_linearization)
    if err:
        print(f"  [linearize] SKIPPED — {err}")
        results["linearization_vs_analytic"] = {"skipped": True, "reason": err}
    else:
        print(f"  warm median={r['linearize_warm_median_s']:.4f}s  "
              f"max_err={r['max_abs_error_vs_analytic']:.2e}  "
              f"matches_analytic={r['matches_analytic_within_1e_6']}")
        results["linearization_vs_analytic"] = r

    return results


# ── --check regression gate (median-vs-baseline, +50% threshold) ────────────


_CHECK_KEYS = {
    "cartpole_throughput": "wall_per_simsec_median",
    "quadruped_throughput": "sim_s",
    "articulated_quadruped_throughput": "sim_s",
    "articulated_quadruped_batched_throughput": "sim_s",
    "sysid_convergence": "wall_s",
    "linearization_vs_analytic": "linearize_warm_median_s",
}


def _resolve_baseline_case(base_cases, name, device_key):
    """Return the baseline dict for ``name`` under ``device_key``.

    Backwards compatibility: a baseline produced before the per-device
    schema landed has metric keys directly on ``base_cases[name]``.  In
    that case we fall back to the flat dict iff ``device_key == "cpu"``
    (which is what the legacy single-device baseline implicitly was);
    otherwise we return ``None`` so the caller can mark the case
    NEW/MISSING rather than silently fail.
    """
    case = base_cases.get(name)
    if not isinstance(case, dict):
        return None
    if device_key in case and isinstance(case[device_key], dict):
        return case[device_key]
    # Legacy flat layout: only honour for cpu.  We detect "flat" by the
    # absence of any of the recognised device keys.
    known_devices = {"cpu", "gpu_t4", "gpu_a100", "gpu_h100", "gpu", "tpu"}
    if device_key == "cpu" and not (known_devices & set(case.keys())):
        return case
    return None


def _check_against_baseline(measured, baseline_path, device_key="cpu"):
    if not baseline_path.exists():
        print(f"baseline file {baseline_path} missing — cannot --check")
        return False
    with baseline_path.open() as f:
        baseline = json.load(f)
    base_cases = baseline.get("cases", {})
    ok = True
    print(f"\n## Public benchmark regression check  "
          f"(device={device_key}, threshold {REGRESSION_THRESHOLD:.2f}x)\n")
    print(f"{'case':<35} {'metric':<28} {'baseline':>12} {'measured':>12} "
          f"{'ratio':>8} status")
    print("-" * 105)
    for name, metric_key in _CHECK_KEYS.items():
        case = measured.get(name, {})
        if case.get("skipped"):
            print(f"{name:<35} {metric_key:<28} {'-':>12} {'-':>12} "
                  f"{'-':>8} SKIPPED")
            continue
        base_case = _resolve_baseline_case(base_cases, name, device_key)
        if base_case is None or base_case.get("skipped"):
            # No baseline for this (case, device) — treat as NEW, not a
            # regression.  This is what makes per-device columns
            # backwards-compatible: missing GPU column ⇒ skip cleanly.
            print(f"{name:<35} {metric_key:<28} {'-':>12} "
                  f"{case.get(metric_key, float('nan')):>12.4g} "
                  f"{'-':>8} NEW")
            continue
        base = base_case.get(metric_key)
        if base is None:
            print(f"{name:<35} {metric_key:<28} {'-':>12} "
                  f"{case.get(metric_key, float('nan')):>12.4g} "
                  f"{'-':>8} NEW")
            continue
        meas = case.get(metric_key, 0.0)
        ratio = meas / base if base > 0 else float("inf")
        status = "OK" if ratio <= REGRESSION_THRESHOLD else "REGRESSION"
        if status == "REGRESSION":
            ok = False
        print(f"{name:<35} {metric_key:<28} {base:>12.4g} {meas:>12.4g} "
              f"{ratio:>8.2f} {status}")
    return ok


def _emit_github_summary(measured):
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if not sp:
        return
    lines = ["## T-019 public benchmark suite", "",
             "| problem | headline metric | value |",
             "| --- | --- | --- |"]
    fmt = {
        "cartpole_throughput": (
            "wall-s / sim-s",
            lambda c: f"{c.get('wall_per_simsec_median', float('nan')):.3e}",
        ),
        "quadruped_throughput": (
            "env-s / wall-s",
            lambda c: f"{c.get('env_seconds_per_wall_second', 0):.1f} "
                      f"(N={c.get('n_env')})",
        ),
        "articulated_quadruped_throughput": (
            "wall-s / sim-s",
            lambda c: f"{c.get('wall_per_simsec', 0):.4f} "
                      f"(N={c.get('n_env')}, nq={c.get('nq')}/nv={c.get('nv')})",
        ),
        "articulated_quadruped_batched_throughput": (
            "env-s / wall-s",
            lambda c: f"{c.get('env_seconds_per_wall_second', 0):.2f} "
                      f"(N={c.get('n_env')}, dt={c.get('dt')})",
        ),
        "sysid_convergence": (
            "iter / wall / param_mse",
            lambda c: f"{c.get('n_iter')} / {c.get('wall_s', 0):.2f}s / "
                      f"{c.get('param_mse', 0):.2e}",
        ),
        "linearization_vs_analytic": (
            "warm s / err vs analytic",
            lambda c: f"{c.get('linearize_warm_median_s', 0):.4f} / "
                      f"{c.get('max_abs_error_vs_analytic', 0):.2e}",
        ),
    }
    for name, (metric, fn) in fmt.items():
        c = measured.get(name, {})
        val = "SKIPPED" if c.get("skipped") else fn(c)
        lines.append(f"| {name} | {metric} | {val} |")
    Path(sp).open("a").write("\n".join(lines) + "\n")


def _migrate_flat_to_per_device(cases):
    """Wrap a legacy flat-shape ``cases`` dict into the per-device schema
    under the ``cpu`` key.  Idempotent — already-nested cases pass through.
    """
    known_devices = {"cpu", "gpu_t4", "gpu_a100", "gpu_h100", "gpu", "tpu"}
    out = {}
    for name, case in cases.items():
        if isinstance(case, dict) and (known_devices & set(case.keys())):
            out[name] = case  # already per-device
        else:
            out[name] = {"cpu": case}
    return out


def _merge_into_baseline(existing_payload, measured, device_key, hardware_info):
    """Merge a freshly-measured per-case dict into the per-device baseline.

    - Other devices' columns are preserved.
    - ``hardware`` becomes a per-device map; legacy flat ``hardware`` is
      migrated under ``cpu`` if no device key conflict.
    """
    # Migrate cases.
    cases = _migrate_flat_to_per_device(existing_payload.get("cases", {}))
    for name, case in measured.items():
        cases.setdefault(name, {})
        cases[name][device_key] = case

    # Migrate hardware: support either a flat dict (old) or a per-device map (new).
    hw = existing_payload.get("hardware", {})
    if isinstance(hw, dict) and "platform" in hw and not any(
        k in hw for k in ("cpu", "gpu", "gpu_t4", "gpu_a100", "gpu_h100", "tpu")
    ):
        hw = {"cpu": hw}
    if not isinstance(hw, dict):
        hw = {}
    hw[device_key] = hardware_info

    return {
        "hardware": hw,
        "regression_threshold": existing_payload.get(
            "regression_threshold", REGRESSION_THRESHOLD
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare against baseline; exit 1 on regression")
    parser.add_argument("--out", type=Path, default=BASELINE_PATH,
                        help="path to write baseline JSON (ignored with --check)")
    parser.add_argument("--quadruped-n", type=int, default=N_QUADRUPED,
                        help=f"quadruped batch size (default {N_QUADRUPED}; "
                             f"falls back to {N_QUADRUPED_FALLBACK} on OOM)")
    parser.add_argument(
        "--device", choices=["cpu", "gpu", "cuda"], default=None,
        help="JAX device platform to run on (default: env JAXONOMY_BENCH_DEVICE "
             "or JAX's default backend).  Note: this flag must agree with the "
             "platform JAX has already initialised; if you mix CLI flag and "
             "env var, set the env var.",
    )
    parser.add_argument(
        "--update-baseline", default=None, metavar="DEVICE_KEY",
        help="When writing the baseline, store under this device column "
             "(e.g. cpu, gpu_t4, gpu_a100, gpu_h100).  Defaults to cpu when "
             "--device is unset, gpu_unknown otherwise.  Other device columns "
             "in the existing baseline are preserved.",
    )
    args = parser.parse_args()

    # If --device was passed, the env var may not have been set in time — warn
    # but proceed.  The recommended path is to set JAXONOMY_BENCH_DEVICE in the
    # environment (or use --device which re-exports it for child processes).
    if args.device and _DEVICE_REQUESTED is None:
        print(
            f"warning: --device {args.device} was passed but JAX was already "
            f"initialised on backend '{jax.default_backend()}'. For reliable "
            f"device selection, prefer `JAXONOMY_BENCH_DEVICE={args.device} "
            f"python benchmarks/public.py ...`."
        )

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"Running public benchmark suite (T-019)\n  N_REPEAT = {N_REPEAT}"
          f"\n  quadruped_n = {args.quadruped_n}\n"
          f"  jax_default_backend = {backend}\n"
          f"  jax_devices = {devices}\n")

    # Resolve the device-column key for both --check and baseline write.
    requested_device = args.device or _DEVICE_REQUESTED or backend
    if requested_device in {"gpu", "cuda"}:
        # Without a hardware-id probe we can't tell T4/A100/H100 apart at
        # runtime.  Default to "gpu" generic; user can override via
        # --update-baseline gpu_t4 / gpu_a100 / gpu_h100.
        default_device_key = "gpu"
    else:
        default_device_key = "cpu" if requested_device == "cpu" else requested_device
    device_key = args.update_baseline or default_device_key

    measured = run_all_cases(quadruped_n=args.quadruped_n)

    if args.check:
        ok = _check_against_baseline(measured, BASELINE_PATH, device_key=device_key)
        _emit_github_summary(measured)
        return 0 if ok else 1

    # Merge into existing baseline so other-device columns are preserved.
    if args.out.exists():
        try:
            with args.out.open() as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        existing = {}

    payload = _merge_into_baseline(
        existing, measured, device_key=device_key,
        hardware_info=_hardware_info(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    print(f"\nWrote baseline to {args.out}  (device column: {device_key!r})")
    _emit_github_summary(measured)
    return 0


if __name__ == "__main__":
    sys.exit(main())
