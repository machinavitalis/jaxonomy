# SPDX-License-Identifier: MIT

"""Differentiable Predictive Control (DPC) helpers.

This module ships a *minimum-viable* closed-loop rollout + cost-composition
+ training loop that supports the canonical DPC pattern:

1. **Rollout.** Build a closed-loop trajectory of a plant under a
   parameterised policy ``u = policy(params, x, ref)``. Implemented via
   fixed-step RK4 on a user-supplied ``plant_ode(time, x, u) -> dx/dt``
   so the entire rollout is JAX-traceable end-to-end. See
   :class:`ClosedLoopRollout`.
2. **Cost composition.** Stage cost + terminal cost + a list of
   :class:`Penalty` terms (soft / barrier flavours). See
   :func:`dpc_loss`.
3. **Training.** Optax-driven gradient descent on the policy parameters.
   See :func:`train_policy`.

**Diagram integration (T-040-followup, shipped 2026-05-20).** The policy
can also be authored as a real jaxonomy ``LeafSystem`` (:class:`PolicyBlock`)
and run inside a feedback ``Diagram`` under :func:`jaxonomy.simulate` — the
same solver stack, event handling, and recording as every other model.
``jax.grad`` of a downstream cost flows through ``simulate`` into the
policy's parameters (validated AD-vs-FD in
``test/control/test_t_040_diagram.py``). See :class:`PolicyBlock`,
:class:`PlantBlock`, :func:`build_closed_loop`, and
:func:`simulate_closed_loop`. The function-level :class:`ClosedLoopRollout`
remains the lightweight path for batched receding-horizon training where a
hand-rolled RK4 over a bare ``plant_ode`` is preferred.

**Still out of scope** (intentionally, per the T-040 anti-scope note):

- A symbolic ``Variable`` / ``Constraint`` DSL — penalties are plain
  Python callables (:class:`Penalty`).
- The Neuromancer two-tank reference-tracking tutorial — a downstream
  ``docs/examples`` notebook, tracked under ``T-040-followup``.
- A batched ``simulate_batch``-over-``Diagram`` rollout that drops into
  :func:`dpc_loss` / :func:`train_policy` directly (the function-level
  :class:`ClosedLoopRollout` covers batched training today).

The :func:`train_policy` callable works end-to-end on a 1-D linear plant;
see ``test/control/test_t_040_dpc_smoke.py``.
"""

from ._rollout import ClosedLoopRollout
from ._loss import Penalty, dpc_loss
from ._train import train_policy
from ._diagram import (
    PolicyBlock,
    PlantBlock,
    build_closed_loop,
    simulate_closed_loop,
    ClosedLoopRunner,
)

__all__ = [
    "ClosedLoopRollout",
    "Penalty",
    "dpc_loss",
    "train_policy",
    "PolicyBlock",
    "PlantBlock",
    "build_closed_loop",
    "simulate_closed_loop",
    "ClosedLoopRunner",
]
