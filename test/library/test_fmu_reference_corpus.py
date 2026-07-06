# SPDX-License-Identifier: MIT
"""
T-026a — stress harness for the FMI Reference-FMUs corpus.

Drives jaxonomy's :class:`~jaxonomy.library.ModelicaFMU` block against the
official reference FMUs published by the Modelica Association,
exercising:

* both FMI **2.0** and **3.0** co-simulation interfaces
* multiple physical domains: continuous mechanical (BouncingBall),
  stiff continuous (VanDerPol), linear discrete (Dahlquist), pure
  discrete (Stair), state-space arrays (StateSpace), and mixed-type
  Feedthrough (Float32/64, Int8/16/32/64, UInt8/16/32/64, Boolean,
  Enumeration)
* scalar and **array** I/O (StateSpace's ``u``/``y`` are 1-D vectors)
* the per-port type-dispatched ``doStep`` path added in T-026a

The reference-FMU release ships *x86_64*-only darwin binaries, so on
Apple Silicon we have to rebuild from source. The corpus location is
discovered via the ``JAXONOMY_FMU_CORPUS`` environment variable; when
it is unset (or the layout is wrong), the whole module is skipped.

Build instructions for the corpus, on macOS-arm64::

    git clone --depth 1 --branch v0.0.39 \\
        https://github.com/modelica/Reference-FMUs.git
    cd Reference-FMUs
    # FMI 2 (the build maps aarch64 -> the existing darwin64/ output dir):
    cmake -B build-fmi2 -DFMI_VERSION=2 -DFMI_ARCHITECTURE=aarch64 \\
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build-fmi2 --config Release
    # FMI 3 (uses the proper aarch64-darwin/ platform string):
    cmake -B build-fmi3 -DFMI_VERSION=3 -DFMI_ARCHITECTURE=aarch64 \\
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build-fmi3 --config Release
    # Then point the harness at:
    mkdir -p ~/.fmu-corpus/{2.0,3.0}
    cp build-fmi2/fmus/*.fmu ~/.fmu-corpus/2.0/
    cp build-fmi3/fmus/*.fmu ~/.fmu-corpus/3.0/
    export JAXONOMY_FMU_CORPUS=~/.fmu-corpus

(Note: the FMI 2 build needs a tiny patch to its ``CMakeLists.txt`` to
let ``aarch64`` map to ``darwin64``; fmpy looks up FMI 2 binaries by
``binaries/darwin64/<id>.dylib`` regardless of ABI.)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Constant, ModelicaFMU


pytestmark = pytest.mark.slow


def _corpus_root() -> Path | None:
    raw = os.environ.get("JAXONOMY_FMU_CORPUS")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


CORPUS = _corpus_root()
SKIP_REASON = (
    "JAXONOMY_FMU_CORPUS not set or invalid; see this module's docstring "
    "for build instructions for the FMI Reference-FMU corpus."
)
pytestmark_skip = pytest.mark.skipif(CORPUS is None, reason=SKIP_REASON)


def _fmu(version: str, name: str) -> Path:
    """Resolve the FMU path under the configured corpus root."""
    assert CORPUS is not None
    return CORPUS / version / f"{name}.fmu"


def _has(version: str, name: str) -> bool:
    return CORPUS is not None and _fmu(version, name).is_file()


# ── 1. Continuous + discrete physics, no inputs ───────────────────────


@pytestmark_skip
@pytest.mark.parametrize(
    "version,model,outputs,checks",
    [
        # Mechanical bouncing ball — h decreases under gravity until
        # the floor event, v swings sign at impact.
        ("2.0", "BouncingBall", {"h", "v"},
         lambda out: out["h"][0] == pytest.approx(1.0, abs=1e-9)
                     and out["h"].min() < 0.5),
        ("3.0", "BouncingBall", {"h", "v"},
         # FMI 3 BouncingBall also exposes ``h_ft`` which is an *alias*
         # of ``h`` (same value reference). T-026a fix avoids the
         # dict-by-vr collapse that previously raised "duplicate field
         # name 'h_ft'" — both ports must register and return the
         # alias's underlying value.
         lambda out: "h_ft" in out
                     and np.allclose(out["h_ft"], out["h"])),

        # Linear scalar Dahlquist — exact analytic decay.
        ("2.0", "Dahlquist", {"x"},
         lambda out: out["x"][0] == pytest.approx(1.0, abs=1e-9)
                     and out["x"][-1] < out["x"][0]),
        ("3.0", "Dahlquist", {"x"},
         lambda out: out["x"][-1] < out["x"][0]),

        # Stiff Van der Pol oscillator — the trace must oscillate (not
        # be a constant).
        ("2.0", "VanDerPol", {"x0", "x1"},
         lambda out: float(np.std(out["x0"])) > 0.1),
        ("3.0", "VanDerPol", {"x0", "x1"},
         lambda out: float(np.std(out["x0"])) > 0.1),

        # Pure discrete staircase counter — strictly non-decreasing.
        ("2.0", "Stair", {"counter"},
         lambda out: bool(np.all(np.diff(out["counter"]) >= 0))),
        ("3.0", "Stair", {"counter"},
         lambda out: bool(np.all(np.diff(out["counter"]) >= 0))),
    ],
)
def test_reference_fmu_input_free(version, model, outputs, checks):
    """Run a reference FMU that needs no external inputs and verify the
    headline behaviour holds. This exercises the type-dispatched
    ``getXxx`` reads on the output side for every numeric width that
    each model uses."""
    if not _has(version, model):
        pytest.skip(f"corpus missing {version}/{model}.fmu")
    bld = jaxonomy.DiagramBuilder()
    blk = bld.add(ModelicaFMU(file_name=str(_fmu(version, model)),
                              dt=0.01, name="b"))
    diagram = bld.build()
    ctx = diagram.create_context()
    rec = {p.name: p for p in blk.output_ports}
    assert outputs.issubset(rec.keys()), (
        f"missing expected outputs on {version}/{model}: "
        f"{outputs - set(rec)}"
    )
    res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), recorded_signals=rec)
    out = {n: np.asarray(v) for n, v in res.outputs.items()}
    assert checks(out), (
        f"physics check failed for {version}/{model}; "
        f"sample last values: { {k: v[-1] for k, v in out.items()} }"
    )


# ── 2. Mixed-type co-simulation: FMI 3 Feedthrough ────────────────────


@pytestmark_skip
def test_fmi3_feedthrough_mixed_types_round_trip():
    """Feedthrough echoes its 14 numeric inputs (Float32/64, Int8…64,
    UInt8…64, Boolean, Enumeration) to outputs of the same type. We
    drive each input with a constant of the matching dtype and verify
    the output port reads back the same value, casted to that type's
    range. This is the load-bearing test for T-026a's per-port type
    dispatch in ``exec_step``."""
    if not _has("3.0", "Feedthrough"):
        pytest.skip("corpus missing 3.0/Feedthrough.fmu")
    import jax.numpy as jnp

    bld = jaxonomy.DiagramBuilder()
    blk = bld.add(ModelicaFMU(file_name=str(_fmu("3.0", "Feedthrough")),
                              dt=0.1, name="ft"))

    # Drive each input with a typed constant, recording our expected
    # echo value.
    expected = {}
    for i, port in enumerate(blk.input_ports):
        var = blk.fmu_input_vars[i]
        if var.type == "Boolean":
            value = True
            const = Constant(jnp.bool_(value), name=f"c{i}")
        elif var.type.startswith("UInt"):
            value = 42
            const = Constant(jnp.uint32(value), name=f"c{i}")
        elif var.type in ("Int8", "Int16", "Int32", "Int64", "Enumeration"):
            value = -7 if var.type != "Enumeration" else 1
            const = Constant(jnp.int64(value), name=f"c{i}")
        elif var.type == "Float32":
            value = 1.5
            const = Constant(jnp.float32(value), name=f"c{i}")
        else:
            value = 2.5
            const = Constant(jnp.float64(value), name=f"c{i}")
        c = bld.add(const)
        bld.connect(c.output_ports[0], port)
        # Map the input port name to its echo output name.
        echo_name = var.name.replace("_input", "_output")
        expected[echo_name] = value

    diagram = bld.build()
    ctx = diagram.create_context()
    rec = {p.name: p for p in blk.output_ports if p.name in expected}
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), recorded_signals=rec)

    for name, want in expected.items():
        got = np.asarray(res.outputs[name])[-1]
        if isinstance(want, bool):
            assert bool(got) == want, f"{name}: {got} != {want}"
        else:
            assert got == pytest.approx(want, rel=1e-5, abs=1e-6), (
                f"{name}: {got} != {want}"
            )


# ── 3. Array I/O: FMI 3 StateSpace ────────────────────────────────────


@pytestmark_skip
def test_fmi3_state_space_array_io():
    """The StateSpace reference FMU has ``u`` and ``y`` of shape (3,) —
    this is the array-I/O path of T-026a. We drive ``u`` with a
    constant 3-vector and verify the output trace has the right shape
    (T, 3) and that ``y`` advances away from its initial state."""
    if not _has("3.0", "StateSpace"):
        pytest.skip("corpus missing 3.0/StateSpace.fmu")
    import jax.numpy as jnp

    bld = jaxonomy.DiagramBuilder()
    blk = bld.add(ModelicaFMU(file_name=str(_fmu("3.0", "StateSpace")),
                              dt=0.05, name="ss"))
    # Sanity: the FMU exposes the expected array I/O shape.
    assert blk.fmu_input_vars[0].shape == (3,)
    assert blk.fmu_output_vars[0].shape == (3,)

    u = bld.add(Constant(jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64),
                         name="u_drive"))
    bld.connect(u.output_ports[0], blk.input_ports[0])

    diagram = bld.build()
    ctx = diagram.create_context()
    rec = {p.name: p for p in blk.output_ports}
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), recorded_signals=rec)

    y = np.asarray(res.outputs["y"])
    assert y.shape[1:] == (3,), f"expected shape (T, 3), got {y.shape}"
    # The state evolves: at least one component should have moved
    # noticeably from the initial sample.
    assert float(np.max(np.abs(y[-1] - y[0]))) > 1e-3, (
        f"y did not evolve: y[0]={y[0]}, y[-1]={y[-1]}"
    )


# ── 4. Diagnostic: scheduledExecution-only FMU is rejected cleanly ────


@pytestmark_skip
def test_clocks_scheduled_execution_rejected_with_clear_error():
    """The FMI 3 Clocks FMU only ships a scheduledExecution interface —
    a different stepping protocol that the co-simulation block
    deliberately does not support. We verify the block raises a
    diagnostic error rather than crashing on a None attribute."""
    if not _has("3.0", "Clocks"):
        pytest.skip("corpus missing 3.0/Clocks.fmu")
    from jaxonomy.framework import BlockInitializationError
    with pytest.raises(BlockInitializationError, match="co-simulation"):
        ModelicaFMU(file_name=str(_fmu("3.0", "Clocks")),
                    dt=0.05, name="clk")
