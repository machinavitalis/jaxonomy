# SPDX-License-Identifier: MIT

"""Index reduction picks the same differential states regardless of hash seed.

Regression for a consumer finding (T-044 phase 2): for symmetric
constraint-coupled variable pairs — the planar pendulum's ``(vx, x)`` vs
``(vy, y)`` under the holonomic constraint ``x² + y² = L²`` — the choice of
which pair survives into ``sed.x`` as differential states depended on
``PYTHONHASHSEED``.  T-002a made the state-vector *ordering* deterministic,
but the DiagramProcessing input path still built its variable lists with
``list(set(...))``, and the Pantelides bipartite graph inserted edges by
iterating sets — both hash-ordered, and both feeding the rref pivot choice
in dummy-derivative selection.

Anything that references a specific compiled differential state by physical
name (e.g. ``NeuralDAEBlock(targets=[(pendulum, "vx")])``) is broken by this
non-determinism, so the partition must be reproducible across processes.
"""

import os
import subprocess
import sys

from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

_CHILD = r"""
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal.component_library.planar import PlanarPendulum

ev = EqnEnv()
ad = AcausalDiagram()
p = PlanarPendulum(ev, name="pend", m=1.0, L=1.0)
ad.comps[p] = None
sys_ = AcausalCompiler(ev, ad)(leaf_backend="jax")
print("x=" + ",".join(sorted(str(s) for s in sys_.sed.x)))
print("y=" + ",".join(sorted(str(s) for s in sys_.sed.y)))
"""


def _compile_partition(hashseed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, f"child (seed {hashseed}) failed:\n{out.stderr}"
    return "\n".join(
        line for line in out.stdout.splitlines() if line.startswith(("x=", "y="))
    )


def test_pendulum_state_partition_is_hash_seed_independent():
    # Seeds 0 and 2 produced different partitions ((vx,x) vs (vy,y)) before
    # the fix; assert fresh processes now agree.
    partitions = {seed: _compile_partition(seed) for seed in (0, 2)}
    assert partitions[0] == partitions[2], (
        "differential/algebraic partition varies with PYTHONHASHSEED:\n"
        f"seed 0:\n{partitions[0]}\nseed 2:\n{partitions[2]}"
    )
    # The differential pair must be one coherent (velocity, position) pair,
    # not a hash-dependent mixture.
    assert partitions[0].splitlines()[0] in (
        "x=pend_vx(t),pend_x(t)",
        "x=pend_vy(t),pend_y(t)",
    )
