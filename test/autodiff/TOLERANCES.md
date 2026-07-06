# Gradient-correctness tolerance policy (T-001)

This policy is the source of truth for how AD-vs-FD gradient agreement is
evaluated in `test/autodiff/`. The numbers are encoded in `tolerances.py` and
consumed by `assert_grad_matches_fd` in `_framework.py`. Any change here must
update `tolerances.py` in the same commit.

## Matrix

| Solver | dtype   | rtol  | atol  | FD step | sim_rtol | sim_atol |
|--------|---------|-------|-------|---------|----------|----------|
| dopri5 | float32 | 5e-3  | 1e-4  | 3e-3    | 1e-6     | 1e-8     |
| dopri5 | float64 | 5e-4  | 1e-5  | 1e-5    | 1e-8     | 1e-10    |
| bdf    | float32 | 1e-2  | 5e-4  | 3e-3    | 1e-5     | 1e-7     |
| bdf    | float64 | 5e-3  | 1e-4  | 1e-5    | 1e-6     | 1e-8     |
| rk4    | float32 | 5e-3  | 1e-4  | 3e-3    | 1e-6     | 1e-8     |
| rk4    | float64 | 1e-3  | 1e-5  | 1e-5    | 1e-8     | 1e-10    |

Stateless (feedthrough/reduce/source with no simulation) checks:

| dtype   | rtol | atol | FD step |
|---------|------|------|---------|
| float32 | 1e-3 | 1e-5 | 3e-3    |
| float64 | 1e-6 | 1e-8 | 1e-5    |

## Rationale

- **FD step selection.** Central-difference truncation error is `O(ε²)` and
  rounding error scales as `O(η / ε)` where `η` is machine epsilon. The optimal
  `ε ≈ η^(1/3)` is ~`6e-6` for float64 and ~`4e-3` for float32. Values in the
  table are chosen just above these optima so FD itself is not the limiting
  factor. Larger FD step on float32 also avoids underflow on sub-unit inputs.
- **BDF looser than Dopri5.** BDF on DAEs (mass-matrix systems) has worse
  adjoint accuracy at the same solver tolerances — confirmed by the
  `test_acausal_spring_mass_ic_grad` comment in the existing suite ("~1 % error
  vs analytic at default tolerance"). The policy reflects this.
- **float32 looser than float64.** A float32 simulation cannot produce a
  gradient with more than ~`1e-4` relative accuracy regardless of solver
  tolerance. The float32 row documents that ceiling explicitly rather than
  silently accepting lower-precision AD.
- **RK4 is fixed-step.** `rtol`/`atol` are accepted for API symmetry but
  ignored by the solver; accuracy is controlled by `max_minor_step_size`. The
  framework sets `max_minor_step_size=0.01` for all RK4 gradient tests, which
  gives ~1e-8 local truncation error on the short horizons used here. For
  longer simulations or stiffer dynamics RK4 will need a smaller step; the
  tolerance row here reflects the 0.01-step regime only.

## Failure-message format

`assert_grad_matches_fd` raises `GradientMismatch` with:
- block / scenario name
- solver and dtype
- `input[i][element]`: AD and FD values
- `abs_err`, `rel_err`, and the `rtol/atol` that were violated
- any `extra` dict the caller attached (e.g. parameter sampled by hypothesis)

## Changing a tolerance

Loosening a tolerance to fix a flaky test is a **red flag, not a fix**. Before
relaxing any number above:

1. Confirm the AD/FD mismatch is inside the expected solver/adjoint accuracy
   envelope (solve the same problem at tighter `sim_rtol` — if the gradient
   stabilises, the test just needs tighter solver settings).
2. Document the investigation in the commit message.
3. Only then update both this file and `tolerances.py`.
