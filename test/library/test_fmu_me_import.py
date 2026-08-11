# SPDX-License-Identifier: MIT
"""Model-exchange FMU import against the FMI Reference-FMU corpus.

A co-simulation import samples the FMU on a fixed communication grid, so
its accuracy is capped by that step size. Model exchange hands the
derivatives to jaxonomy's solver instead, which is both more accurate and
the only interface some tools export at all (OpenModelica emits model
exchange, and its own importer accepts nothing else).

These tests exercise :class:`~jaxonomy.library.ModelicaFMUME` against the
official Reference FMUs in both FMI 2.0 and FMI 3.0:

* ``Dahlquist``   — one continuous state, no events, analytic solution
* ``VanDerPol``   — two states, stiff limit cycle, no events
* ``BouncingBall``— one event indicator: zero-crossing localization plus
  a state reset applied through the FMU's own event iteration

The corpus location comes from ``JAXONOMY_FMU_CORPUS`` (the same variable
``test_fmu_reference_corpus.py`` uses); the module skips when it is unset.
CI builds the corpus, so this runs there on every push. Build it locally
with::

    git clone --depth 1 --branch v0.0.39 \\
        https://github.com/modelica/Reference-FMUs.git
    cd Reference-FMUs
    MODELS="BouncingBall VanDerPol Dahlquist Stair Feedthrough"
    cmake -B b2 -DFMI_VERSION=2 -DCMAKE_BUILD_TYPE=Release
    cmake --build b2 --target $MODELS
    cmake -B b3 -DFMI_VERSION=3 -DCMAKE_BUILD_TYPE=Release
    cmake --build b3 --target $MODELS StateSpace
    mkdir -p ~/.fmu-corpus/2.0 ~/.fmu-corpus/3.0
    cp b2/fmus/*.fmu ~/.fmu-corpus/2.0/ && cp b3/fmus/*.fmu ~/.fmu-corpus/3.0/
    export JAXONOMY_FMU_CORPUS=~/.fmu-corpus
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from jaxonomy import DiagramBuilder, simulate, SimulatorOptions
from jaxonomy.library import ModelicaFMU, ModelicaFMUME


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
requires_corpus = pytest.mark.skipif(CORPUS is None, reason=SKIP_REASON)

FMI_VERSIONS = ["2.0", "3.0"]


def _fmu(version: str, model: str) -> str:
    path = CORPUS / version / f"{model}.fmu"
    if not path.is_file():
        pytest.skip(f"{model}.fmu missing from the {version} corpus")
    return str(path)


def _run(version, model, t_end, options=None, **block_kwargs):
    builder = DiagramBuilder()
    block = builder.add(
        ModelicaFMUME(_fmu(version, model), name=model, **block_kwargs)
    )
    diagram = builder.build()
    recorded = {port.name: port for port in block.output_ports}
    results = simulate(
        diagram,
        diagram.create_context(),
        (0.0, t_end),
        options=options,
        recorded_signals=recorded,
    )
    time = np.asarray(results.time)
    outputs = {k: np.asarray(v).squeeze() for k, v in results.outputs.items()}
    return time, outputs


@requires_corpus
@pytest.mark.parametrize("version", FMI_VERSIONS)
def test_dahlquist_matches_analytic_solution(version):
    """der(x) = -k*x with k = 1, x0 = 1 has the closed form exp(-t).

    The point of the assertion is the *scale*: a co-simulation import of
    the same model is limited by its communication step, while here the
    error tracks solver tolerance.
    """
    time, outputs = _run(version, "Dahlquist", 5.0)
    x = outputs["x"]
    assert np.abs(x - np.exp(-time)).max() < 1e-5


@requires_corpus
@pytest.mark.parametrize("version", FMI_VERSIONS)
def test_vanderpol_limit_cycle_stays_bounded(version):
    """The mu = 1 limit cycle settles near |x0| = 2; a broken derivative
    or state sync diverges instead of orbiting."""
    _, outputs = _run(version, "VanDerPol", 20.0)
    x0 = outputs["x0"]
    assert 1.5 < np.abs(x0).max() < 3.0


@requires_corpus
@pytest.mark.parametrize("version", FMI_VERSIONS)
def test_bouncing_ball_events_reset_the_state(version):
    """The ball must bounce, not fall through the floor.

    Height stays non-negative only if the event indicator becomes a
    zero-crossing *and* the reset map runs the FMU's event iteration and
    reads back the reinitialized continuous state. Skipping the initial
    event iteration on FMI 3.0 (which leaves the FMU in event mode after
    initialization, unlike FMI 2.0) put the ball at h = -8.4.
    """
    time, outputs = _run(version, "BouncingBall", 3.0)
    h = outputs["h"]
    assert h.min() > -1e-3, f"ball fell through the floor: h_min={h.min()}"
    assert h.max() == pytest.approx(1.0, abs=1e-3)
    rebounds = int(np.sum((h[:-1] < 0.05) & (np.diff(h) > 0)))
    assert rebounds >= 3, f"expected repeated rebounds, saw {rebounds}"


@requires_corpus
def test_fmi2_and_fmi3_bouncing_ball_agree():
    """The two builds of one reference model must produce the same
    trajectory. This is the regression guard for version-specific event
    handling: FMI 2.0 needs an explicit enterEventMode after
    initialization and FMI 3.0 does not."""
    common = SimulatorOptions(max_major_step_length=0.01)
    time2, out2 = _run("2.0", "BouncingBall", 3.0, options=common)
    time3, out3 = _run("3.0", "BouncingBall", 3.0, options=common)

    def _bounce_times(time, h):
        # Comparing sample-by-sample would measure interpolation error
        # around the discontinuities rather than agreement; the bounce
        # instants are the physically meaningful signature.
        below = h < 1e-6
        starts = np.flatnonzero(below[1:] & ~below[:-1]) + 1
        return time[starts]

    b2 = _bounce_times(time2, out2["h"])
    b3 = _bounce_times(time3, out3["h"])
    assert len(b2) == len(b3), f"bounce counts differ: {len(b2)} vs {len(b3)}"
    assert len(b2) >= 3, f"expected repeated bounces, saw {len(b2)}"
    assert np.abs(b2 - b3).max() < 1e-2


@requires_corpus
def test_co_simulation_only_fmu_is_rejected_with_guidance(tmp_path):
    """A CS-only FMU must fail at construction naming the other block,
    not fail obscurely at the first derivative evaluation."""
    import shutil
    import zipfile

    source = _fmu("2.0", "VanDerPol")
    stripped = tmp_path / "CsOnly.fmu"
    shutil.copy(source, stripped)
    # Rewriting the descriptor is enough: the block dispatches on the
    # presence of the ModelExchange element.
    with zipfile.ZipFile(stripped) as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    xml = entries["modelDescription.xml"].decode("utf-8")
    if "<ModelExchange" not in xml:
        pytest.skip("reference FMU has no ModelExchange element to remove")
    start = xml.index("<ModelExchange")
    end = xml.index(">", xml.index("</ModelExchange>")) + 1
    entries["modelDescription.xml"] = (xml[:start] + xml[end:]).encode("utf-8")
    with zipfile.ZipFile(stripped, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    from jaxonomy.framework import BlockInitializationError

    with pytest.raises(BlockInitializationError, match="ModelicaFMU"):
        ModelicaFMUME(str(stripped), name="cs_only")


@requires_corpus
def test_model_exchange_beats_co_simulation_on_the_same_model():
    """Both interfaces of one reference FMU, same reference trajectory.

    Dahlquist has a closed form, so this compares each import path
    against the truth rather than against each other. Co-simulation is
    limited by its communication step; model exchange is not.
    """
    dt = 0.05
    time_me, out_me = _run(
        "2.0", "Dahlquist", 5.0,
        options=SimulatorOptions(max_major_step_length=dt),
    )
    me_error = np.abs(out_me["x"] - np.exp(-time_me)).max()

    builder = DiagramBuilder()
    block = builder.add(
        ModelicaFMU(_fmu("2.0", "Dahlquist"), dt=dt, name="cs")
    )
    diagram = builder.build()
    results = simulate(
        diagram,
        diagram.create_context(),
        (0.0, 5.0),
        options=SimulatorOptions(max_major_step_length=dt),
        recorded_signals={"x": block.output_ports[0]},
    )
    cs_error = np.abs(
        np.asarray(results.outputs["x"]).squeeze()
        - np.exp(-np.asarray(results.time))
    ).max()

    assert me_error < cs_error / 10, (
        f"model exchange {me_error:.3e} vs co-simulation {cs_error:.3e} "
        f"at the same communication step"
    )
