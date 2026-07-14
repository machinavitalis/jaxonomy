# Scope: reduced-order modeling and surrogates

**TL;DR — reduced-order modeling (ROM) and data-driven surrogates *of
dynamical systems* are in scope for Jaxonomy and live in
`jaxonomy.library.rom`.** Given a full-order model (an ODE/DAE diagram or
snapshot data from one), Jaxonomy can build a cheaper reduced model that is
still a first-class, differentiable, simulatable block. What is **out of
scope** is the same thing as for PINNs — surrogates of *spatial PDE fields*
(neural operators, `u(x, t)` collocation); see [`pinn.md`](pinn.md).

## What "ROM" covers here

A reduced-order model approximates a high-dimensional or expensive dynamical
system with a low-order one that is fast to simulate. Jaxonomy supports the
three families that matter for control and simulation engineers, plus
statistical surrogates for the input→output map:

| Family | Methods | When to reach for it |
|---|---|---|
| **Linear MOR** | balanced truncation (`balred`), minimal realization (`minreal`), modal truncation, singular-perturbation residualization, Krylov/IRKA *(planned)* | You have (or can linearize to) an LTI model and want a smaller LTI with a certified error bound |
| **Projection ROM** | POD–Galerkin, Petrov–Galerkin/LSPG, DEIM hyper-reduction | You have the *equations* of a large nonlinear ODE/DAE (e.g. a method-of-lines PDE) and snapshots, and want an intrusive, physics-preserving reduction |
| **Data-driven operator ROM** | DMD, DMDc, ERA, eDMD / Koopman | You have *data* (snapshots), maybe no equations, and want a linear predictor — including a lifted-linear (Koopman) model you can drop straight into linear MPC/LQR |
| **Statistical surrogates** | Gaussian process / kriging, polynomial chaos (PCE), RBF response surface | You want a cheap, optionally uncertainty-aware surrogate of an input→output map (design maps, UQ, calibration) |

## Choosing a method

- **Do you have the model equations, or only data?** Equations → linear MOR
  (if linear) or POD–Galerkin/DEIM (if nonlinear). Data only → DMD/DMDc,
  ERA, or eDMD/Koopman.
- **Linear or nonlinear?** Linear and you want a guaranteed error →
  balanced truncation (a priori H∞ bound). Nonlinear with equations →
  POD–Galerkin, and add DEIM so the per-step cost stops scaling with the
  full state dimension. Nonlinear with only data → eDMD/Koopman with a
  lifting dictionary.
- **Is the reduced model for a controller?** Koopman/DMDc produce a *linear*
  reduced model in (possibly lifted) coordinates — a good basis for linear
  MPC / LQR-style control (design in lifted coordinates, de-lift with `C`).
  A lifted model wants a *terminal-cost* MPC rather than a hard
  terminal-equality one; the `rom_dmdc_koopman_mpc` example shows the pattern.
- **Is it an input→output map, not a trajectory?** Use a statistical
  surrogate (GP/PCE/RBF). PCE additionally yields analytic Sobol indices and
  moments, so it doubles as an accelerated-UQ path into `jaxonomy.uq`.

## In scope — what Jaxonomy does

Every reducer returns a first-class Jaxonomy object: linear MOR returns a
reduced `LinearizedSystem`/`LTISystem`; projection and operator ROMs return a
differentiable, `jit`/`vmap`-able `LeafSystem` you can compose in a diagram
and drive through `jaxonomy.simulate`; statistical surrogates are feedthrough
blocks. ROM quality metrics (relative trajectory error, retained energy,
projection error, held-out cross-validation) live alongside the reducers.

## Out of scope (use these instead)

- **Spatial PDE field surrogates / neural operators** (`u(x, t)` over a
  domain, Fourier/DeepONet operators) — see [`pinn.md`](pinn.md). A
  spatially *discretized* PDE (method of lines) is a large ODE and *can* be
  reduced with POD–Galerkin/DEIM here; Jaxonomy just does not own the
  discretization or train neural fields.
- **Mesh generation, CFD/FEA solvers.** Bring your own high-fidelity solver;
  Jaxonomy reduces the resulting dynamical system or its snapshots.

The dividing line is the same as for PINNs: **if the reduced object is a
dynamical system evolving in time (or a map you sample), it belongs here; if
it is a neural field trained against a spatial PDE residual, it does not.**
