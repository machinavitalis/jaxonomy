# Scope: PINNs and PDE surrogates

**TL;DR — classical physics-informed neural networks (PINNs) for PDEs are out
of scope for Jaxonomy.** Jaxonomy is a simulation engine for systems governed
by ODEs and DAEs evolving in *time*; it has no spatial discretization, no
collocation-point sampling, and no PDE residual machinery, and we do not plan
to add them.

## What "PINN" means here

The term is used for two quite different things. Only one of them belongs in
Jaxonomy:

| | Classical PDE PINN | Physics-informed *dynamics* learning |
|---|---|---|
| Governing equations | PDEs over space(-time): Burgers, Navier–Stokes, heat equation | ODEs / DAEs over time: mechanics, circuits, thermal networks, chemistry |
| Unknown | A neural field `u(x, t)` trained to satisfy the PDE residual at collocation points | Parameters and/or a neural correction term inside a *simulated* model |
| Core machinery | Spatial sampling, residual losses, boundary/initial-condition penalties | A differentiable time-stepping simulator |
| In Jaxonomy? | **No** | **Yes — this is a core capability** |

## Out of scope (use these instead)

If you want to train `u(x, t)` against a PDE residual — surrogate models for
fluid fields, heat maps over a plate, wave propagation — use a library built
for it:

- [DeepXDE](https://github.com/lululxvi/deepxde) — the reference PINN library
  (PDEs, IDEs, fractional PDEs; TensorFlow/PyTorch/JAX backends).
- [NVIDIA PhysicsNeMo (formerly Modulus)](https://developer.nvidia.com/physicsnemo)
  — industrial-scale physics-ML, including PINNs and neural operators.
- [Neuromancer](https://github.com/pnnl/neuromancer) — differentiable
  programming for constrained optimization and physics-informed system
  identification in PyTorch.

A spatially discretized PDE (method of lines) *can* be simulated in Jaxonomy —
a finite-volume battery-electrode model or a discretized heat rod is just a
large ODE/DAE system — but Jaxonomy does not own the discretization, and we
will not add collocation/residual training utilities for neural fields.

## In scope (what Jaxonomy does instead)

Physics-informed learning where the physics enters through a **differentiable
simulation in time**:

- **Universal differential equations (UDE)** — a neural term inside an ODE
  right-hand side, trained end-to-end through `simulate` (see the
  [UDE + symbolic regression example](../examples/ude_and_sr_lotka_volterra.ipynb)).
- **Neural DAE** — a neural correction inside an *acausal, constrained* DAE
  (`NeuralDAEBlock`; see the
  [constrained pendulum drag-recovery example](../examples/neural_dae_pendulum_drag.ipynb)).
  The index-reduction pipeline (Pantelides) runs unchanged with the neural
  term in place.
- **Neural ODE blocks** — `MLP` (Equinox) and imported `PyTorch` /
  `TensorFlow` / `ONNX` networks as blocks inside a diagram.
- **SINDy** — sparse symbolic regression of dynamics from data (`Sindy`
  block).
- **Differentiable parameter estimation** — `fit_parameters`, lookup-table
  fitting, and the whole autodiff/optimization workflow.

The dividing line: **if the "physics" constraint is enforced by simulating a
dynamical system forward in time, it belongs here; if it is enforced by a
residual loss over a spatial domain, it does not.**
