# SPDX-License-Identifier: MIT

"""What the persistent JIT cache, kernel reuse and an outer ``jit`` save.

Run directly:

    python benchmarks/jit_cache.py            # print the tables
    python benchmarks/jit_cache.py --json out.json

This is the measured run behind ``docs/jit_cache.md``.  Three questions, two
models (a 6-block PID loop and a 159-block chain of 23 coupled oscillators):

1. *A later process.*  Time to build the model and finish the first
   ``simulate`` in a fresh process: without the cache, while writing it, and
   in a later process that reads it and uses other parameter and initial
   values.
2. *More runs in the same process.*  The same call again; a new parameter
   value set through the context; a new parameter value set with
   ``Diagram.with_parameters`` (a copy of the diagram per value, which runs
   on the kernel of its original), without the disk cache and with it.
3. *Repeated gradients.*  ``jax.grad`` through ``simulate`` called bare and
   wrapped in ``jax.jit``: first call and later calls.

Every number comes from its own Python process (the cache only matters
across processes, and JAX's in-memory caches would otherwise leak between
measurements).  Reported values are medians over ``--runs`` processes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------- models

def pid_loop(kp=2.0, x0=0.0):
    """Discrete PID around a leaky integrator plant: 6 blocks."""
    from jaxonomy import DiagramBuilder
    from jaxonomy.library import Adder, Constant, Gain, Integrator, PIDDiscrete

    b = DiagramBuilder()
    ref = b.add(Constant(1.0, name="ref"))
    err = b.add(Adder(2, operators="+-", name="err"))
    pid = b.add(PIDDiscrete(kp=kp, ki=1.0, kd=0.05, dt=0.01, name="pid"))
    total = b.add(Adder(2, name="sum"))
    leak = b.add(Gain(-0.5, name="leak"))
    plant = b.add(Integrator(initial_state=x0, name="plant"))
    b.connect(ref.output_ports[0], err.input_ports[0])
    b.connect(plant.output_ports[0], err.input_ports[1])
    b.connect(err.output_ports[0], pid.input_ports[0])
    b.connect(pid.output_ports[0], total.input_ports[0])
    b.connect(plant.output_ports[0], leak.input_ports[0])
    b.connect(leak.output_ports[0], total.input_ports[1])
    b.connect(total.output_ports[0], plant.input_ports[0])
    return b.build(name="pid_loop"), (0.0, 2.0), "leak"


def oscillator_chain(k=4.0, x0=0.7, n=23):
    """n masses coupled by springs, built from primitive blocks: 159 blocks."""
    import math

    from jaxonomy import DiagramBuilder
    from jaxonomy.library import Adder, Gain, Integrator

    b = DiagramBuilder()
    xs, vs, adders = [], [], []
    for i in range(n):
        v = b.add(Integrator(initial_state=0.0, name=f"v{i}"))
        x = b.add(Integrator(initial_state=math.sin(x0 * i), name=f"x{i}"))
        b.connect(v.output_ports[0], x.input_ports[0])
        a = b.add(Adder(2 + (i > 0) + (i < n - 1), name=f"f{i}"))
        b.connect(a.output_ports[0], v.input_ports[0])
        xs.append(x), vs.append(v), adders.append(a)
    for i in range(n):
        port = 0
        g = b.add(Gain(-2.0 * k, name=f"gs{i}"))
        b.connect(xs[i].output_ports[0], g.input_ports[0])
        b.connect(g.output_ports[0], adders[i].input_ports[port])
        port += 1
        g = b.add(Gain(-0.2, name=f"gd{i}"))
        b.connect(vs[i].output_ports[0], g.input_ports[0])
        b.connect(g.output_ports[0], adders[i].input_ports[port])
        port += 1
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                g = b.add(Gain(k, name=f"gc{i}_{j}"))
                b.connect(xs[j].output_ports[0], g.input_ports[0])
                b.connect(g.output_ports[0], adders[i].input_ports[port])
                port += 1
    return b.build(name="oscillator_chain"), (0.0, 10.0), "gs0"


# name -> (builder, first values, other values)
MODELS = {
    "pid_loop": (pid_loop, dict(kp=2.0), dict(kp=3.1, x0=0.2)),
    "oscillator_chain": (oscillator_chain, dict(k=4.0), dict(k=5.5, x0=0.9)),
}


# ---------------------------------------------------------------------- child

def _child(model: str, mode: str, cache_dir: str) -> dict:
    import jax
    import jax.monitoring

    import jaxonomy
    from jaxonomy import logging as jx_logging

    jx_logging.set_log_level("ERROR")
    compiles = [0]

    def on_event(event, duration, **kwargs):
        if event.endswith("backend_compile_duration"):
            compiles[0] += 1

    jax.monitoring.register_event_duration_secs_listener(on_event)
    if cache_dir:
        jaxonomy.enable_persistent_jit_cache(cache_dir)

    build, first, other = MODELS[model]

    def timed(fn):
        c0 = compiles[0]
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        return time.perf_counter() - t, compiles[0] - c0

    def cache_files():
        return os.listdir(cache_dir) if cache_dir and os.path.isdir(cache_dir) else []

    if mode in ("first", "other"):
        before = len(cache_files())
        t = time.perf_counter()
        diagram, t_span, _ = build(**(first if mode == "first" else other))
        context = diagram.create_context()
        t_build = time.perf_counter() - t
        t_sim, _ = timed(lambda: jaxonomy.simulate(diagram, context, t_span).context.continuous_state)
        files = cache_files()
        return {"blocks": len(diagram.nodes), "build_s": t_build, "simulate_s": t_sim,
                "new_entries": len(files) - before,
                "cache_kb": sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files) / 1e3}

    diagram, t_span, block = build(**first)
    context = diagram.create_context()

    if mode == "sweep":
        def run(d, ctx):
            return timed(lambda: jaxonomy.simulate(d, ctx, t_span).context.continuous_state)

        out = {"first": run(diagram, context), "again": run(diagram, context)}
        shifted = [x * 1.01 + 0.001 for x in context.continuous_state]
        out["other_state"] = run(diagram, context.with_continuous_state(shifted))
        sid = diagram[block].system_id
        gain = float(context[sid].parameters["gain"])
        out["context"] = [
            run(diagram, context.with_subcontext(sid, context[sid].with_parameter("gain", gain * f)))
            for f in (1.1, 1.2, 1.3, 1.4)]
        out["diagram"] = []
        before = len(cache_files())
        for f in (1.1, 1.2, 1.3, 1.4):
            def copy_and_run(f=f):  # the whole step per value: copy, context, simulate
                d2 = diagram.with_parameters({f"{block}.gain": gain * f})
                return jaxonomy.simulate(d2, d2.create_context(), t_span).context.continuous_state
            out["diagram"].append(timed(copy_and_run))
        out["diagram_new_entries"] = len(cache_files()) - before
        return out

    if mode in ("grad_bare", "grad_jit"):
        import jax.numpy as jnp
        from jaxonomy import SimulatorOptions

        options = SimulatorOptions(enable_autodiff=True, max_major_steps=200)

        def loss(x):
            result = jaxonomy.simulate(diagram, context.with_continuous_state(x), t_span,
                                       options=options)
            return sum(jnp.sum(leaf ** 2) for leaf in result.context.continuous_state)

        x0 = [jnp.asarray(leaf) for leaf in context.continuous_state]
        grad = jax.grad(loss) if mode == "grad_bare" else jax.jit(jax.grad(loss))
        return {"calls": [timed(lambda: grad(x0)) for _ in range(4)]}

    raise SystemExit(f"unknown mode {mode}")


def _run_child(model: str, mode: str, cache_dir: str = "") -> dict:
    p = subprocess.run([sys.executable, __file__, "--child", model, mode, cache_dir],
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------- driver

def _median(values):
    return statistics.median(values)


def measure(runs: int) -> dict:
    import jax

    res = {"environment": {"platform": platform.platform(), "machine": platform.machine(),
                           "python": platform.python_version(), "jax": jax.__version__,
                           "runs": runs},
           "models": {}}
    for model in MODELS:
        later = {"no_cache": [], "writing": [], "reading_other_values": []}
        for _ in range(runs):
            with tempfile.TemporaryDirectory() as tmp:
                cache = os.path.join(tmp, "jit")
                later["no_cache"].append(_run_child(model, "first"))
                later["writing"].append(_run_child(model, "first", cache))
                later["reading_other_values"].append(_run_child(model, "other", cache))
        sweeps = [_run_child(model, "sweep") for _ in range(runs)]
        with tempfile.TemporaryDirectory() as tmp:
            cached_sweep = _run_child(model, "sweep", os.path.join(tmp, "jit"))
        grads = {m: [_run_child(model, m) for _ in range(runs)] for m in ("grad_bare", "grad_jit")}
        rows = {}
        for key, rs in later.items():
            rows[key] = {"build_s": _median(r["build_s"] for r in rs),
                         "simulate_s": _median(r["simulate_s"] for r in rs),
                         "new_entries": rs[-1]["new_entries"], "cache_kb": rs[-1]["cache_kb"]}
        sweep = {
            "first_s": _median(s["first"][0] for s in sweeps),
            "again_s": _median(s["again"][0] for s in sweeps),
            "context_s": _median(t for s in sweeps for t, _ in s["context"]),
            "context_compiles": max(c for s in sweeps for _, c in s["context"]),
            "diagram_s": _median(t for s in sweeps for t, _ in s["diagram"]),
            "diagram_compiles": max(c for s in sweeps for _, c in s["diagram"]),
            "diagram_cached_s": _median(t for t, _ in cached_sweep["diagram"]),
            "diagram_cache_entries_per_value": cached_sweep["diagram_new_entries"]
            / len(cached_sweep["diagram"]),
        }
        grad = {m: {"first_s": _median(g["calls"][0][0] for g in gs),
                    "first_compiles": gs[-1]["calls"][0][1],
                    "later_s": _median(t for g in gs for t, _ in g["calls"][1:]),
                    "later_compiles": max(c for g in gs for _, c in g["calls"][1:])}
                for m, gs in grads.items()}
        res["models"][model] = {"blocks": later["no_cache"][-1]["blocks"], "later_process": rows,
                                "same_process": sweep, "gradients": grad}
    return res


def report(res: dict) -> str:
    env = res["environment"]
    ms = lambda s: f"{1e3 * s:.0f} ms" if s >= 0.0095 else f"{1e3 * s:.1f} ms"
    out = [f"{env['platform']}, Python {env['python']}, JAX {env['jax']}; "
           f"medians of {env['runs']} processes", ""]
    out += ["A later process (build + first simulate)", "",
            "| model | no cache | writing the cache | later process, other values | cache size |",
            "| --- | --- | --- | --- | --- |"]
    for name, m in res["models"].items():
        r = m["later_process"]
        tot = lambda k: r[k]["build_s"] + r[k]["simulate_s"]
        out.append(f"| {name} ({m['blocks']} blocks) | {ms(tot('no_cache'))} | {ms(tot('writing'))} | "
                   f"{ms(tot('reading_other_values'))} "
                   f"({r['reading_other_values']['new_entries']} new entries) | "
                   f"{r['writing']['cache_kb']:.0f} kB |")
    out += ["", "More runs in the same process", "",
            "| model | first simulate | same call again | new value through the context | "
            "new value through `Diagram.with_parameters` | the same with the disk cache on |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name, m in res["models"].items():
        s = m["same_process"]
        out.append(f"| {name} | {ms(s['first_s'])} | {ms(s['again_s'])} | {ms(s['context_s'])} "
                   f"({s['context_compiles']} compiles) | {ms(s['diagram_s'])} "
                   f"({s['diagram_compiles']} compiles per value) | {ms(s['diagram_cached_s'])} "
                   f"({s['diagram_cache_entries_per_value']:g} new cache entries per value) |")
    out += ["", "Repeated gradients through simulate", "",
            "| model | bare: first call | bare: later calls | jitted: first call | "
            "jitted: later calls |", "| --- | --- | --- | --- | --- |"]
    for name, m in res["models"].items():
        g = m["gradients"]
        out.append(f"| {name} | {ms(g['grad_bare']['first_s'])} "
                   f"({g['grad_bare']['first_compiles']} programs) | {ms(g['grad_bare']['later_s'])} | "
                   f"{ms(g['grad_jit']['first_s'])} ({g['grad_jit']['first_compiles']} program) | "
                   f"{ms(g['grad_jit']['later_s'])} |")
    return "\n".join(out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(_child(sys.argv[2], sys.argv[3], sys.argv[4])))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=5, help="processes per measurement (median)")
    ap.add_argument("--json", default=None, help="also write the raw results here")
    args = ap.parse_args()
    res = measure(args.runs)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    print(report(res))


if __name__ == "__main__":
    main()
