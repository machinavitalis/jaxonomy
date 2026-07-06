# SPDX-License-Identifier: MIT

"""T-101: jaxonomy.uq smoke tests.

Covers the five required cases:

1. ``test_sample_parameters_shape`` — IID sampling shape + bounds.
2. ``test_latin_hypercube_uniform_distribution`` — LHS strata coverage.
3. ``test_sobol_known_function`` — Ishigami benchmark vs analytic indices.
4. ``test_morris_recovers_relative_importance`` — Morris ranking on Ishigami.
5. ``test_sobol_with_simulate_batch`` — wired through ``simulate_batch``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions
from jaxonomy.library import Gain, Integrator
from jaxonomy.testing.markers import skip_if_not_jax
from jaxonomy.uq import (
    LogNormal,
    Normal,
    Triangular,
    Uniform,
    latin_hypercube_sample,
    morris_screening,
    sample_parameters,
    sobol_indices,
)

skip_if_not_jax()
pytestmark = pytest.mark.minimal


OPTS = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="dopri5",
    rtol=1e-8,
    atol=1e-10,
    max_major_steps=200,
)


# ---------------------------------------------------------------------------
# 1. sample_parameters
# ---------------------------------------------------------------------------

def test_sample_parameters_shape():
    dists = {
        "a": Uniform(0.0, 1.0),
        "b": Normal(0.0, 1.0),
        "c": LogNormal(0.0, 0.5),
    }
    n = 64
    out = sample_parameters(dists, n, jax.random.PRNGKey(0))
    assert set(out.keys()) == {"a", "b", "c"}
    for v in out.values():
        assert v.shape == (n,)

    # Uniform stays in bounds.
    assert float(jnp.min(out["a"])) >= 0.0
    assert float(jnp.max(out["a"])) <= 1.0
    # LogNormal is strictly positive.
    assert float(jnp.min(out["c"])) > 0.0


def test_sample_parameters_triangular_in_bounds():
    dists = {"x": Triangular(low=-1.0, mode=0.0, high=2.0)}
    samples = sample_parameters(dists, 256, jax.random.PRNGKey(1))["x"]
    assert float(jnp.min(samples)) >= -1.0 - 1e-12
    assert float(jnp.max(samples)) <= 2.0 + 1e-12


# ---------------------------------------------------------------------------
# 2. Latin Hypercube
# ---------------------------------------------------------------------------

def test_latin_hypercube_uniform_distribution():
    """LHS should have at least one sample per stratum on each axis."""
    n = 50
    dists = {f"x{i}": Uniform(0.0, 1.0) for i in range(3)}
    out = latin_hypercube_sample(dists, n, jax.random.PRNGKey(2))
    for v in out.values():
        v_np = np.asarray(v)
        # Every stratum [k/n, (k+1)/n] should contain exactly one sample.
        bins = np.floor(v_np * n).astype(int).clip(0, n - 1)
        assert len(np.unique(bins)) == n, (
            f"LHS produced duplicate strata: {len(np.unique(bins))}/{n}"
        )

    # Compare LHS vs IID discrepancy on the same N: LHS should fill the
    # cube more uniformly (median pairwise stratum count == 1, not >= 2).
    iid = sample_parameters(dists, n, jax.random.PRNGKey(3))
    iid_bins = np.floor(np.asarray(iid["x0"]) * n).astype(int).clip(0, n - 1)
    # IID nearly always has duplicates (birthday-paradox-style).
    assert len(np.unique(iid_bins)) < n


# ---------------------------------------------------------------------------
# Ishigami test function (Sobol benchmark)
# ---------------------------------------------------------------------------

ISH_A = 7.0
ISH_B = 0.1


def _ishigami(p):
    x1, x2, x3 = p["x1"], p["x2"], p["x3"]
    return jnp.sin(x1) + ISH_A * jnp.sin(x2) ** 2 + ISH_B * (x3 ** 4) * jnp.sin(x1)


def _ishigami_analytic_indices() -> dict[str, dict[str, float]]:
    """Closed-form first- and total-order indices for the Ishigami function
    on ``[-pi, pi]^3``."""
    a, b = ISH_A, ISH_B
    pi = np.pi
    V1 = 0.5 * (1.0 + b * pi**4 / 5.0) ** 2
    V2 = a**2 / 8.0
    V3 = 0.0
    V13 = (b**2 * pi**8) * (1.0 / 18.0 - 1.0 / 50.0)
    V = V1 + V2 + V13
    return {
        "x1": {"first_order": V1 / V, "total_order": (V1 + V13) / V},
        "x2": {"first_order": V2 / V, "total_order": V2 / V},
        "x3": {"first_order": V3 / V, "total_order": V13 / V},
    }


# ---------------------------------------------------------------------------
# 3. Sobol on Ishigami
# ---------------------------------------------------------------------------

def test_sobol_known_function():
    pi = float(np.pi)
    dists = {
        "x1": Uniform(-pi, pi),
        "x2": Uniform(-pi, pi),
        "x3": Uniform(-pi, pi),
    }
    res = sobol_indices(
        diagram=None,
        t_span=None,
        distributions=dists,
        qoi_fn=_ishigami,
        n_samples=4096,
        key=jax.random.PRNGKey(0),
    )
    expected = _ishigami_analytic_indices()

    # First-order: x1 and x2 dominate; x3 first-order ~ 0.
    assert res["x1"]["first_order"] == pytest.approx(
        expected["x1"]["first_order"], abs=0.05
    )
    assert res["x2"]["first_order"] == pytest.approx(
        expected["x2"]["first_order"], abs=0.05
    )
    assert abs(res["x3"]["first_order"]) < 0.05

    # Total-order: x3 picks up interaction with x1.
    assert res["x3"]["total_order"] > 0.10  # analytic ~0.244
    assert res["x1"]["total_order"] > res["x1"]["first_order"]
    assert res["x2"]["total_order"] == pytest.approx(
        expected["x2"]["total_order"], abs=0.05
    )


# ---------------------------------------------------------------------------
# 4. Morris on Ishigami
# ---------------------------------------------------------------------------

def test_morris_recovers_relative_importance():
    pi = float(np.pi)
    dists = {
        "x1": Uniform(-pi, pi),
        "x2": Uniform(-pi, pi),
        "x3": Uniform(-pi, pi),
    }
    res = morris_screening(
        diagram=None,
        t_span=None,
        distributions=dists,
        qoi_fn=_ishigami,
        n_trajectories=40,
        levels=4,
        key=jax.random.PRNGKey(0),
    )
    mu_star = {k: v["mu_star"] for k, v in res.items()}
    # x2 is the highest-magnitude main effect (mu_star largest), x3 smallest.
    assert mu_star["x2"] > mu_star["x1"]
    assert mu_star["x1"] > mu_star["x3"]
    # sigma is a non-linearity proxy; all should be > 0 on Ishigami.
    for v in res.values():
        assert v["sigma"] > 0.0


# ---------------------------------------------------------------------------
# 5. Sobol through simulate_batch (rate-of-decay parametrised ODE)
# ---------------------------------------------------------------------------

def _build_decay_diagram():
    """y' = -k*y, y(0)=1.  Rate ``k`` is dynamic via the gain block."""
    b = DiagramBuilder()
    g = b.add(Gain(gain=-1.0, name="g"))
    intg = b.add(Integrator(initial_state=1.0, name="intg"))
    b.connect(g.output_ports[0], intg.input_ports[0])
    b.connect(intg.output_ports[0], g.input_ports[0])
    return b.build(name="decay")


def test_sobol_with_simulate_batch():
    """A parametrised exponential decay where only the rate matters."""
    diagram = _build_decay_diagram()
    # Two parameters: g.gain (= -k, the rate) and a dummy (initial_state of
    # the integrator) that is held effectively constant — first-order index
    # of g.gain should dominate.
    dists = {
        "g.gain": Uniform(-2.0, -0.5),
        "intg.initial_state": Uniform(0.99, 1.01),
    }

    def qoi_fn(res):
        return jnp.asarray(res.outputs["y"][:, -1])

    recorded = {"y": diagram["intg"].output_ports[0]}

    res = sobol_indices(
        diagram=diagram,
        t_span=(0.0, 2.0),
        distributions=dists,
        qoi_fn=qoi_fn,
        n_samples=128,
        options=OPTS,
        recorded_signals=recorded,
        key=jax.random.PRNGKey(0),
    )
    # The rate parameter dominates: first-order index large, the dummy's small.
    assert res["g.gain"]["first_order"] > 0.7
    assert res["intg.initial_state"]["first_order"] < 0.1


# ---------------------------------------------------------------------------
# Smoke: top-level import
# ---------------------------------------------------------------------------

def test_uq_module_exposed():
    assert hasattr(jaxonomy, "uq")
    assert callable(jaxonomy.uq.sobol_indices)
    assert callable(jaxonomy.uq.morris_screening)
    assert callable(jaxonomy.uq.sample_parameters)


# ---------------------------------------------------------------------------
# T-101-followup-fused-saltelli: bit-exactness fused vs unfused
# ---------------------------------------------------------------------------

def test_sobol_fused_matches_unfused():
    """The fused (single simulate_batch) and unfused (d+2 calls) paths must
    return numerically identical Sobol indices — the math is the same, only
    batching differs."""
    pi = float(np.pi)
    dists = {
        "x1": Uniform(-pi, pi),
        "x2": Uniform(-pi, pi),
        "x3": Uniform(-pi, pi),
    }
    key = jax.random.PRNGKey(7)
    res_unfused = sobol_indices(
        diagram=None, t_span=None, distributions=dists,
        qoi_fn=_ishigami, n_samples=256, key=key, fused=False,
    )
    res_fused = sobol_indices(
        diagram=None, t_span=None, distributions=dists,
        qoi_fn=_ishigami, n_samples=256, key=key, fused=True,
    )
    for name in dists:
        assert res_fused[name]["first_order"] == pytest.approx(
            res_unfused[name]["first_order"], rel=1e-12, abs=1e-12,
        )
        assert res_fused[name]["total_order"] == pytest.approx(
            res_unfused[name]["total_order"], rel=1e-12, abs=1e-12,
        )


# ---------------------------------------------------------------------------
# T-101-followup-vectorised-morris: bit-exactness vectorised vs Python loop
# ---------------------------------------------------------------------------

def test_morris_vectorised_matches_loop():
    """The vectorised (fused=True) elementary-effect aggregation must match
    the legacy Python-loop path bit-for-bit."""
    pi = float(np.pi)
    dists = {
        "x1": Uniform(-pi, pi),
        "x2": Uniform(-pi, pi),
        "x3": Uniform(-pi, pi),
    }
    key = jax.random.PRNGKey(11)
    res_loop = morris_screening(
        diagram=None, t_span=None, distributions=dists, qoi_fn=_ishigami,
        n_trajectories=20, levels=4, key=key, fused=False,
    )
    res_vec = morris_screening(
        diagram=None, t_span=None, distributions=dists, qoi_fn=_ishigami,
        n_trajectories=20, levels=4, key=key, fused=True,
    )
    for name in dists:
        assert res_vec[name]["mu_star"] == pytest.approx(
            res_loop[name]["mu_star"], rel=1e-12, abs=1e-12,
        )
        assert res_vec[name]["sigma"] == pytest.approx(
            res_loop[name]["sigma"], rel=1e-12, abs=1e-12,
        )


# ---------------------------------------------------------------------------
# Smoke benchmark (slow, prints timing — does not assert ratio)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sobol_fused_faster_at_large_d():
    """Synthetic d=10 surrogate: report fused vs unfused wall time.

    No assertion on speedup ratio (CI noise) — the print output goes into
    the log and confirms the perf direction.  In analytic mode the win is
    smaller than in simulation mode (no per-matrix JIT compile to amortise),
    but the single-call qoi_fn invocation still trims overhead.
    """
    import time
    d = 10
    dists = {f"x{i}": Uniform(-1.0, 1.0) for i in range(d)}

    def qoi(p):
        # Cheap nonlinear surrogate — sum of sin(i * x_i).
        out = jnp.zeros_like(p["x0"])
        for i, k in enumerate(p):
            out = out + jnp.sin((i + 1) * p[k])
        return out

    key = jax.random.PRNGKey(0)
    n = 256

    # Warm up (JAX trace).
    _ = sobol_indices(
        diagram=None, t_span=None, distributions=dists, qoi_fn=qoi,
        n_samples=8, key=key, fused=True,
    )
    _ = sobol_indices(
        diagram=None, t_span=None, distributions=dists, qoi_fn=qoi,
        n_samples=8, key=key, fused=False,
    )

    t0 = time.perf_counter()
    sobol_indices(
        diagram=None, t_span=None, distributions=dists, qoi_fn=qoi,
        n_samples=n, key=key, fused=False,
    )
    t_unfused = time.perf_counter() - t0

    t0 = time.perf_counter()
    sobol_indices(
        diagram=None, t_span=None, distributions=dists, qoi_fn=qoi,
        n_samples=n, key=key, fused=True,
    )
    t_fused = time.perf_counter() - t0

    print(
        f"\n[sobol_indices d={d} N={n}] unfused={t_unfused*1e3:.2f} ms "
        f"fused={t_fused*1e3:.2f} ms speedup={t_unfused/t_fused:.2f}x"
    )
