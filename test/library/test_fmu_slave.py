# SPDX-License-Identifier: MIT
"""
In-process ``JaxonomyDiagramSlave`` tests (no FMU build).

Covers the FMU-export conformance/DX sweep:

- outputs primed at initialization (FMI 2.0 §4.2.4 — readable after
  ``exit_initialization_mode``, before the first ``do_step``);
- exported diagram input ports honored (wrapped onto injected
  Constant blocks instead of being silently ignored);
- vector Constant blocks flatten to per-element FMI inputs;
- ``EXPOSE_INITIAL_STATES`` FMI parameters applied at initialization;
- embedded jaxonomy logging quieted (and opt-outs);
- ``REUSE_SIMULATOR`` persistent-kernel path matches the fresh
  per-segment ``simulate`` path.

The binary round-trip counterparts live in
``test_fmu_export_binary.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pythonfmu = pytest.importorskip("pythonfmu")
from pythonfmu import Fmi2Causality, Fmi2Variability  # noqa: E402

import jaxonomy  # noqa: E402
from jaxonomy.library import Constant, Gain, Integrator  # noqa: E402
from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave  # noqa: E402


def _make_slave(cls, tmp_path):
    return cls(instance_name="x", resources=str(tmp_path))


# ── diagram factories ─────────────────────────────────────────────────


def _build_const_only():
    bld = jaxonomy.DiagramBuilder()
    c = bld.add(Constant(7.5, name="seven_pt_five"))
    bld.export_output(c.output_ports[0], name="out")
    return bld.build()


def _build_setpoint_gain():
    bld = jaxonomy.DiagramBuilder()
    sp = bld.add(Constant(0.0, name="setpoint"))
    g = bld.add(Gain(2.0, name="g"))
    bld.connect(sp.output_ports[0], g.input_ports[0])
    bld.export_output(g.output_ports[0], name="out")
    return bld.build()


def _build_exported_input():
    bld = jaxonomy.DiagramBuilder()
    g = bld.add(Gain(2.0, name="g"))
    bld.export_input(g.input_ports[0], name="u")
    bld.export_output(g.output_ports[0], name="y")
    return bld.build()


def _build_integrator():
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(1.0, name="rate"))
    integ = bld.add(Integrator(0.5, name="integ"))
    bld.connect(src.output_ports[0], integ.input_ports[0])
    bld.export_output(integ.output_ports[0], name="x")
    return bld.build()


class ConstSlave(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build_const_only)


class SetpointSlave(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build_setpoint_gain)


class ExportedInputSlave(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build_exported_input)


class IntegratorSlave(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build_integrator)
    EXPOSE_INITIAL_STATES = {"x0": "integ"}


# ── 1. Outputs primed at initialization ───────────────────────────────


def test_outputs_primed_at_exit_initialization_mode(tmp_path):
    """getReal on an output right after exitInitializationMode must
    return the diagram's t=0 value, not the 0.0 registration
    placeholder (FMI 2.0 §4.2.4)."""
    slave = _make_slave(ConstSlave, tmp_path)
    assert slave._values["out"] == 0.0  # placeholder before init
    slave.exit_initialization_mode()
    assert slave._values["out"] == pytest.approx(7.5, abs=1e-12)


def test_priming_applies_initialization_mode_inputs(tmp_path):
    """Inputs the master writes during initialization mode must be
    reflected in the primed outputs."""
    slave = _make_slave(SetpointSlave, tmp_path)
    slave._values["setpoint"] = 3.0
    slave.exit_initialization_mode()
    assert slave._values["out"] == pytest.approx(6.0, abs=1e-12)


def test_zero_step_size_do_step_still_refreshes_outputs(tmp_path):
    """The historical doStep(step_size=0.0) workaround keeps working."""
    slave = _make_slave(ConstSlave, tmp_path)
    assert slave.do_step(0.0, 0.0) is True
    assert slave._values["out"] == pytest.approx(7.5, abs=1e-12)


# ── 2. Exported diagram input ports ───────────────────────────────────


def test_exported_input_port_drives_diagram(tmp_path):
    """setReal on an exported diagram input port must reach the
    diagram (previously it silently did nothing)."""
    slave = _make_slave(ExportedInputSlave, tmp_path)
    var_names = {v.name for v in slave.vars.values()}
    assert "u" in var_names
    u_var = next(v for v in slave.vars.values() if v.name == "u")
    assert u_var.causality == Fmi2Causality.input

    t = 0.0
    for u, expected_y in [(4.0, 8.0), (-1.5, -3.0)]:
        slave._values["u"] = u
        assert slave.do_step(t, 0.01) is True
        assert slave._values["y"] == pytest.approx(expected_y, abs=1e-9), (
            f"u={u}: exported input port ignored, y={slave._values['y']}"
        )
        t += 0.01


def test_exported_input_port_primed_at_initialization(tmp_path):
    slave = _make_slave(ExportedInputSlave, tmp_path)
    slave._values["u"] = 2.5
    slave.exit_initialization_mode()
    assert slave._values["y"] == pytest.approx(5.0, abs=1e-9)


def test_exported_input_name_collision_raises(tmp_path):
    """An exported input port whose name collides with another FMI
    variable fails loudly at construction instead of silently dropping
    the input."""
    def _build():
        bld = jaxonomy.DiagramBuilder()
        g = bld.add(Gain(2.0, name="g"))
        bld.export_input(g.input_ports[0], name="dup")
        bld.export_output(g.output_ports[0], name="dup")
        return bld.build()

    class Coll(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build)

    with pytest.raises(RuntimeError, match="collides"):
        _make_slave(Coll, tmp_path)


# ── 3. Vector Constant inputs flatten to elements ─────────────────────


def test_vector_constant_flattens_to_element_inputs(tmp_path):
    """A vector-valued Constant registers one FMI Real per element
    (``name[i]``) and element writes land in the context (previously a
    vector Constant crashed registration with float()-on-array)."""
    def _build():
        bld = jaxonomy.DiagramBuilder()
        bld.add(Constant(np.array([1.0, 2.0]), name="vec"))
        c = bld.add(Constant(0.0, name="scalar_out"))
        bld.export_output(c.output_ports[0], name="out")
        return bld.build()

    class Vec(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build)

    slave = _make_slave(Vec, tmp_path)
    var_names = {v.name for v in slave.vars.values()}
    assert {"vec[0]", "vec[1]"}.issubset(var_names)
    assert slave._values["vec[0]"] == pytest.approx(1.0)
    assert slave._values["vec[1]"] == pytest.approx(2.0)

    slave._values["vec[0]"] = 4.0
    slave._values["vec[1]"] = 5.0
    assert slave.do_step(0.0, 0.01) is True
    vec_id = next(l.system_id for l in slave._diagram.leaf_systems
                  if l.name == "vec")
    applied = np.asarray(slave._ctx[vec_id].parameters["value"])
    assert np.allclose(applied, [4.0, 5.0]), applied


# ── 4. EXPOSE_INITIAL_STATES ──────────────────────────────────────────


def test_expose_initial_states_registers_fixed_parameter(tmp_path):
    slave = _make_slave(IntegratorSlave, tmp_path)
    x0_var = next(v for v in slave.vars.values() if v.name == "x0")
    assert x0_var.causality == Fmi2Causality.parameter
    assert x0_var.variability == Fmi2Variability.fixed
    # Start value mirrors the diagram's built-in initial state.
    assert slave._values["x0"] == pytest.approx(0.5, abs=1e-12)


def test_expose_initial_states_applied_at_initialization(tmp_path):
    slave = _make_slave(IntegratorSlave, tmp_path)
    slave._values["x0"] = 2.0
    slave.exit_initialization_mode()
    # Applied to the context and visible in the primed output ...
    assert slave._values["x"] == pytest.approx(2.0, abs=1e-9)
    # ... and integration continues from there: dx/dt = 1.
    assert slave.do_step(0.0, 0.1) is True
    assert slave._values["x"] == pytest.approx(2.1, abs=1e-6)


def test_expose_initial_states_default_is_diagram_initial_state(tmp_path):
    """Untouched x0 parameters leave the built-in initial state."""
    slave = _make_slave(IntegratorSlave, tmp_path)
    slave.exit_initialization_mode()
    assert slave._values["x"] == pytest.approx(0.5, abs=1e-9)


def test_expose_initial_states_unknown_block_raises(tmp_path):
    class Bad(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(_build_integrator)
        EXPOSE_INITIAL_STATES = {"x0": "no_such_block"}

    with pytest.raises(RuntimeError, match="no leaf system named"):
        _make_slave(Bad, tmp_path)


# ── 5. Logging scoped to the slave's embedded calls ───────────────────


class _NoisySlave(JaxonomyDiagramSlave):
    """Logs at INFO from inside the slave's embedded-call scope."""
    DIAGRAM_FACTORY = staticmethod(_build_const_only)

    def read_outputs(self, ctx):
        logging.getLogger("jaxonomy").info("noisy per-step message")
        return super().read_outputs(ctx)


@pytest.fixture
def jaxonomy_log_capture():
    logger = logging.getLogger("jaxonomy")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield logger, records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def test_embedded_calls_quiet_by_default(tmp_path, jaxonomy_log_capture,
                                         monkeypatch):
    """Per-step sub-ERROR logging inside the slave is suppressed even
    when the process-wide jaxonomy logger is chatty — without touching
    the logger outside the slave's calls."""
    logger, records = jaxonomy_log_capture
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    slave = _make_slave(_NoisySlave, tmp_path)
    assert slave.do_step(0.0, 0.01) is True
    assert all(r.levelno >= logging.ERROR for r in records), (
        f"embedded calls leaked sub-ERROR log records: "
        f"{[r.getMessage() for r in records]}"
    )
    # The pre-existing level is restored once the call returns.
    assert logger.level == logging.DEBUG
    logger.info("outside the slave")
    assert any(r.getMessage() == "outside the slave" for r in records)


def test_log_level_env_var_opts_out(tmp_path, jaxonomy_log_capture,
                                    monkeypatch):
    """A user-set LOG_LEVEL env var wins: the slave leaves the logger
    alone and the per-step message passes through."""
    _logger, records = jaxonomy_log_capture
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    slave = _make_slave(_NoisySlave, tmp_path)
    assert slave.do_step(0.0, 0.01) is True
    assert any(r.getMessage() == "noisy per-step message" for r in records)


def test_log_level_none_opts_out(tmp_path, jaxonomy_log_capture,
                                 monkeypatch):
    _logger, records = jaxonomy_log_capture
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    class Quietless(_NoisySlave):
        LOG_LEVEL = None

    slave = _make_slave(Quietless, tmp_path)
    assert slave.do_step(0.0, 0.01) is True
    assert any(r.getMessage() == "noisy per-step message" for r in records)


# ── 6. REUSE_SIMULATOR persistent kernel ──────────────────────────────


def test_reuse_simulator_matches_fresh_simulate(tmp_path):
    """The persistent-kernel path must be numerically equivalent to
    one fresh simulate() call per segment, including across input
    changes, and must actually reuse the kernel."""
    class Fresh(SetpointSlave):
        REUSE_SIMULATOR = False

    reuse = _make_slave(SetpointSlave, tmp_path)
    fresh = _make_slave(Fresh, tmp_path)
    assert SetpointSlave.REUSE_SIMULATOR is True  # default

    t = 0.0
    setpoints = [5.0, -3.0, 1.5, 1.5, 0.25]
    for sp in setpoints:
        for slave in (reuse, fresh):
            slave._values["setpoint"] = sp
            assert slave.do_step(t, 0.01) is True
        assert reuse._values["out"] == pytest.approx(
            fresh._values["out"], abs=1e-12
        )
        t += 0.01

    assert reuse._kernel is not None
    assert fresh._kernel is None


def test_reuse_simulator_stateful_integration(tmp_path):
    """State carries across kernel steps: the integrator accumulates
    exactly like the fresh-simulate path."""
    class Fresh(IntegratorSlave):
        REUSE_SIMULATOR = False

    reuse = _make_slave(IntegratorSlave, tmp_path)
    fresh = _make_slave(Fresh, tmp_path)
    for slave in (reuse, fresh):
        slave.exit_initialization_mode()

    t = 0.0
    for _ in range(5):
        for slave in (reuse, fresh):
            assert slave.do_step(t, 0.1) is True
        assert reuse._values["x"] == pytest.approx(
            fresh._values["x"], abs=1e-9
        )
        t += 0.1
    # dx/dt = 1 from x0 = 0.5 over 0.5s.
    assert reuse._values["x"] == pytest.approx(1.0, abs=1e-6)


def test_reuse_simulator_kernel_object_is_stable(tmp_path):
    """Subsequent equal-sized steps reuse the same jitted kernel."""
    slave = _make_slave(IntegratorSlave, tmp_path)
    assert slave.do_step(0.0, 0.1) is True
    kernel_first = slave._kernel
    assert kernel_first is not None
    assert slave.do_step(0.1, 0.1) is True
    assert slave.do_step(0.2, 0.05) is True  # smaller step: no rebuild
    assert slave._kernel is kernel_first
    assert slave.do_step(0.25, 0.5) is True  # larger step: rebuild
    assert slave._kernel is not kernel_first
