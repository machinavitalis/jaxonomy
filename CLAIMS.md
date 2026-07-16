# Claims

**Internal document — not published.** Maps every claim made in a shippable surface
(`README.md`, `docs/**`, `examples/**`, `benchmarks/**`, root-level
`*.ipynb`) to the specific evidence that substantiates it. The full
list of shippable surfaces and the discipline that governs them lives
in `AGENTS/RULES.md` — this file is the evidence ledger that rule
relies on.

## The rule

Two layers of discipline govern shippable-surface claims:

- **New or modified claims** get a row here: point it at an evidence
  file, run the evidence, confirm green. A row with no green evidence
  means the claim comes down.
- **Pre-existing claims** not yet in the table are covered by the
  [adversarial-review pass in AGENTS/RULES.md](AGENTS/RULES.md) until a
  shippable-surface edit touches them — that edit is the moment to
  register the claim with a row. Backfill is opportunistic.

The pre-publication checklist below applies to *registered* claims.

This is the discipline that separates a real engineering project from a
vibe-coded one. Readers don't have to trust us; they can check.

This file is public, alongside its inverse `KNOWN_GAPS.md`: CLAIMS records
what we've proven and the evidence behind it; KNOWN_GAPS records what we
haven't proven yet. Both ship.

---

## Format

| ID | Claim (as written) | Where it appears | Evidence file | Status |
|---|---|---|---|---|
| C-NNN | [exact wording of the claim] | [path or URL] | [path to test/benchmark/log] | green / yellow / red |

**Status meaning:**

- **green** — evidence file exists, runs in CI, passes. Claim is
  publishable.
- **yellow** — evidence file exists but isn't yet in CI, or runs
  locally but not yet committed. Acceptable for short windows only.
- **red** — no evidence yet, evidence is failing, or claim drifted
  from what evidence shows. **Claim must come down within 24 hours.**

---

## Active claims

| ID | Claim (as written) | Where it appears | Evidence file | Status |
|---|---|---|---|---|
| C-002 | DPC training reduces the two-tank tracking loss ~1570x (59.97 → 0.038) and bottom-level tracking RMS 12.8x (1.271 m → 0.099 m), trained policy settles within 24 mm of a 0.6 m setpoint. | `docs/examples/dpc_two_tank_reference_tracking.ipynb` (training + result cells) | same notebook — executed outputs (cells run live, no checkpoint) | green |
| C-003 | `jax.grad` of a terminal cost flows through `simulate_closed_loop` and matches central finite differences to relative error ~5.9e-6. | `docs/examples/dpc_two_tank_reference_tracking.ipynb` (AD-vs-FD cell) | same notebook cell + `test/control/test_t_040_diagram.py::test_gradient_through_simulate_matches_fd` | green |
| C-004 | The trained DPC policy recovers the analytic steady-state command u* = (c2/kp)√r (0.461 vs 0.452 at r=0.6; 0.407 vs 0.369 at r=0.4). | `docs/examples/dpc_two_tank_reference_tracking.ipynb` (steady-state validation cell) | same notebook — executed outputs | green |
| C-005 | Exported FMUs pass the official `fmpy.validate_fmu` checker with zero findings; CI additionally runs INTO-CPS VDMCheck2 on every generated FMU. | `KNOWN_GAPS.md` (FMU support) | `test/library/test_t_026c_fmu_official_validation.py` (fmpy gate runs everywhere; VDMCheck2 via the `fmu-validators` CI job) | green |
| C-006 | Balanced truncation (`balred`) and minimal realization (`minreal`) match `python-control`'s implementations, and the reduced model respects the a priori H∞ error bound ‖G−Gᵣ‖∞ ≤ 2·Σ(truncated Hankel singular values). | `README.md` (Reduced-order modeling), `docs/scope/rom.md` | `test/library/test_rom_linear_mor.py` (control cross-validation + error-bound tests) | green |
| C-007 | POD–Galerkin reduces a linear method-of-lines heat equation and DEIM hyper-reduces a nonlinear (Fisher–KPP) model, with reduced-vs-full trajectory relative L2 error < 1e-2 (heat, r=5) and < 5e-2 (DEIM, r=m=8, n=60), evaluating the nonlinearity at only m points. | `README.md` (Reduced-order modeling), `docs/scope/rom.md` | `test/library/test_rom_pod.py` | green |
| C-008 | DMD, DMDc, and ERA recover the eigenvalues / (A,B) / impulse response of known linear systems from snapshots, and the DMD/Koopman predictor blocks simulate in `jaxonomy.simulate`; eDMD with a polynomial dictionary beats plain DMD on a nonlinear map. | `README.md` (Reduced-order modeling), `docs/scope/rom.md` | `test/library/test_rom_dmd.py`, `test/library/test_rom_koopman.py` | green |
| C-009 | Gaussian-process, polynomial-chaos, and RBF surrogate blocks fit and evaluate as differentiable Jaxonomy blocks: GP means match `scikit-learn`, PCE analytic mean/variance and Sobol indices match Monte-Carlo, and RBF interpolates its training data exactly. | `README.md` (Data-driven modeling) | `test/library/test_rom_surrogates.py` | green |
| C-010 | `multi_event_time_gradient` computes the saltation gradient dt_e/dp for every firing along a hybrid trajectory by propagating forward sensitivities through reset maps; per-firing values match central finite differences within 5% on a multi-bounce restitution problem (the correctly-zero first-bounce sensitivity is recovered exactly), with PyTree parameters and `jax.grad` composability. | `CHANGELOG.md` ([3.0.0] Added, `multi_event_time_gradient`) | `test/autodiff/test_t_125_followup_multi_event_saltation_bug.py` (plus `test_t_125_followup_multi_event.py`, `test_t_125_followup_multi_events_batched.py`) | green |
| C-011 | Reverse-mode BDF-DAE adjoint gradients w.r.t. acausal parameters that enter through an algebraic constraint (e.g. `Insulator.R`) match central finite differences within 5% on single-cell and two-cell thermal networks. | `CHANGELOG.md` ([3.0.0] Fixed, T-113-followup-dae-adjoint-sign-bug) | `test/acausal/test_t_113_followup_dae_adjoint_sign_bug.py` | green |
| C-012 | `implicit_solver(solver, residual)` makes a `lax.while_loop`-based iterative solve reverse-mode differentiable via the implicit function theorem: the wrapped scalar/vector/PyTree solvers match analytic gradients, jit and vmap, and `jax.grad` through `simulate` with a wrapped solver matches finite differences (the unwrapped solver demonstrably fails to reverse-differentiate). | `CHANGELOG.md` ([3.1.0] Added, T-131) | `test/optimization/test_t_131_implicit_solver.py` | green |
| C-013 | `declare_continuous_state(project=fn)` keeps a quaternion on the unit sphere where the unprojected state drifts, applies after every step under fixed-step `rk4` and at major-step boundaries under adaptive solvers, is differentiable, and composes with `substeps=`. | `CHANGELOG.md` ([3.1.0] Added, T-132) | `test/simulation/test_t_132_state_projection.py` | green |
| C-014 | `declare_continuous_state(substeps=N)` stabilizes a stiff block that blows up single-rate under fixed-step `rk4`, supports multiple fast groups, and is jit-, vmap-, and reverse-AD-compatible with gradients matching finite differences. | `CHANGELOG.md` ([3.1.0] Added, T-133) | `test/simulation/test_t_133_multirate_substepping.py` | green |
| C-015 | `SimulatorOptions(dae_initial_projection=True)` recovers state resets on compiled acausal DAEs that otherwise NaN on the first implicit step; `AcausalSystem.continuous_state_layout()` names each state row (differential vs algebraic); `project_constraints` gradients are FD-verified in both the default value-only and the implicit (IFT, `lax.custom_root`) modes and warn on non-convergence; the BDF-DAE backward sweep zeroes the algebraic rows of the initial-state cotangent so reset-then-integrate gradients match finite differences on the differential rows. | `CHANGELOG.md` ([3.1.0] Added + Fixed, DAE projection / algebraic initial states) | `test/simulation/test_dae_initial_projection.py`, `test/simulation/test_dae_projection_gradients.py` | green |
| C-016 | `JaxonomyDiagramSlave` FMU export honors exported diagram input ports as real FMI inputs, primes outputs during `exitInitializationMode` (no 0.0 placeholders before the first `doStep`), applies `EXPOSE_INITIAL_STATES` FMI parameters at initialization, and the default-on cached-kernel `REUSE_SIMULATOR` path matches a fresh `simulate` bit-identically. | `CHANGELOG.md` ([3.1.0] Added, FMU export) | `test/library/test_fmu_slave.py`, `test/library/test_fmu_export_binary.py` | green |
| C-017 | When the BDF corrector/step loop goes non-finite, the solver emits a `UserWarning` with the failure time, collapsed step size, and offending state rows (negative time = reverse-time adjoint solve); healthy runs stay silent, and forward results are bit-identical with and without the autodiff path. | `CHANGELOG.md` ([3.1.0] Added, BDF non-finite abort diagnostic) | `test/simulation/test_bdf_nonfinite_diagnostics.py` | green |
| C-018 | A declared weak (non-fixed) acausal state IC that the consistent-initialization solve overrides by a materially different value emits a `UserWarning` naming the symbol, the declared value, the value actually used, and the `ic_fixed=True` remedy. | `CHANGELOG.md` ([3.1.0] Fixed, acausal IC warnings) | `test/acausal/test_acausal_ic_warnings_and_opentank.py` | green |
| C-019 | Recording-buffer overflow degrades to uniform decimation — `results.time` starts at t0 and spans the whole trajectory at reduced resolution with memory bounded by `buffer_length` — and warns once ("recorded at reduced resolution"), including on the `simulate_batch` kernel/vmap paths. | `CHANGELOG.md` ([3.1.0] Changed, T-138) | `test/simulation/test_t_138_overflow_decimation.py`, `test/simulation/test_buffer_overflow_warning.py` | green |
| C-020 | A NEUROMANCER-trained DPC policy exported through ONNX (`ONNXJax` + `ZeroOrderHold`) runs in a Jaxonomy closed loop with max state deviation 3.84e-08 vs the reference over 400 closed-loop steps, and a jit+vmap 512-sample parameter-robustness sweep evaluates in ~7 ms steady-state. | `docs/examples/pinn_across_stacks_part_1_policy_export.ipynb` | same notebook — executed outputs | green |
| C-021 | A physics-structured neural flow correction inside a compiled acausal DAE (2 differential + 18 algebraic states) trains end-to-end through the implicit BDF solve, recovering the injected fault with 1.43% mean relative error and 0.996 correlation; AD matches central finite differences to ~5e-5 relative. | `docs/examples/pinn_across_stacks_part_2_neural_dae.ipynb` | same notebook — executed outputs (offline cross-check: `docs/examples/media/pinn_across_stacks_part_2_publication_offline.py`) | green |
| C-022 | The same plant exported as an FMI 2.0 co-simulation FMU (`fmpy.validate_fmu`: zero findings) and driven in lockstep by the PyTorch policy matches the in-process simulation to max deviation 1.783e-08 m over the horizon. | `docs/examples/pinn_across_stacks_part_3_fmi_cosim.ipynb` | same notebook — executed outputs | green |

*Pre-existing claims are covered by the adversarial-review pass in
`AGENTS/RULES.md`; rows get added here as shippable-surface edits
backfill them.*

Backfill backlog — high-value claims to register first:

- Differentiability claims in `README.md` → tests under `test/autodiff/`
- Solver correctness claims → `test/integrators/` and `validation/`
- Acausal compiler claims → `test/acausal/`
- Cross-tool agreement with `python-control` → tests guarded by
  `pytest.importorskip("control")`
- Block library count ("150+ library blocks" in `README.md`) → audit
  `jaxonomy/library/` against `jaxonomy.library.__all__`. Live count
  is 151 LeafSystem/Diagram subclasses — keep the
  README number conservative relative to the live count.

---

## Pre-publication checklist (for *registered* claims)

Before a shippable-surface change ships, for the subset of claims that
have rows here:

1. Every new claim added by the change has a row in this file.
2. Every existing *registered* claim the change touches has been
   re-verified (evidence re-run green).
3. No row is in red status.
4. Yellow rows have a written plan to reach green within a defined
   window.
5. The corresponding `KNOWN_GAPS.md` entries are updated for anything
   no longer claimed.

If any of the above fails on a registered row, the surface does not
ship. Un-registered pre-existing claims fall back to the
adversarial-review pass in `AGENTS/RULES.md`.

---

## Why this file exists

Two reasons:

1. **Self-discipline.** The failure mode is that claims drift from
   reality without anyone noticing. This file is the forcing function.
   If a row can't be filled, the claim can't be made.

2. **External readiness.** When the conversation gets serious with a
   downstream consumer, an evaluator, or a partner, the question "how
   do you know that's true?" will be asked. This file is the answer
   and the artifact that demonstrates the engineering discipline
   behind it.
