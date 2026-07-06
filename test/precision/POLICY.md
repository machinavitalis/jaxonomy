# Numerical precision policy (T-005)

This document is the source of truth for Jaxonomy's floating-point
precision behaviour.  The tests in this directory enforce the policy.

## The default: float64

Jaxonomy enables `JAX_ENABLE_X64` during package import
(`jaxonomy/_init.py:12`).  Consequently:

- `jnp.asarray(1.0)` → `float64` (not float32, which is JAX's default
  without x64 enabled).
- Continuous state arrays are `float64` unless the user constructs them
  from `np.float32` inputs.
- Solver tolerances (`rtol`, `atol`) default to values that assume
  float64 arithmetic (`rtol=1e-6`, `atol=1e-8`); these are below the
  float32 precision floor and would produce meaningless "tight"
  simulations if x64 is disabled.
- Integer time uses `int64` (picosecond resolution), independent of
  the float dtype choice.

## Opting into float32

Precision downgrade is a **supported but not default** configuration.
To run in float32:

```bash
JAX_ENABLE_X64=false python your_script.py
```

Or in-process **before importing jaxonomy**:

```python
import os
os.environ["JAX_ENABLE_X64"] = "false"
import jaxonomy  # must come after the env var
```

Once jaxonomy is imported with x64 enabled, switching mid-process is
not supported — JAX caches the dtype decision globally.

When running in float32, you **must** loosen solver tolerances:

| Solver | rtol (f32) | atol (f32) |
|--------|-----------|------------|
| rk4    | n/a (fixed-step)  | n/a        |
| dopri5 | ≥ 1e-5    | ≥ 1e-7     |
| bdf    | ≥ 1e-4    | ≥ 1e-6     |

Values tighter than these are below the float32 epsilon (~1.19e-7)
and the solver will either stall or return garbage.

## Error bounds per solver at default tolerance

These are documented absolute-error bounds on the final state of a
well-conditioned benchmark simulation (10-second scalar exponential
decay, `dx/dt = -x`, `x(0) = 1`).  Measured empirically; see
`test_precision_bounds.py`.

| Solver | dtype   | rtol   | atol   | max |Δx(T)| |
|--------|---------|--------|--------|---------------|
| rk4    | float64 | n/a    | n/a    | < 1e-10       |
| dopri5 | float64 | 1e-6   | 1e-8   | < 1e-7        |
| dopri5 | float64 | 1e-10  | 1e-12  | < 1e-11       |
| bdf    | float64 | 1e-6   | 1e-8   | < 1e-5        |
| bdf    | float64 | 1e-10  | 1e-12  | < 1e-9        |
| dopri5 | float32 | 1e-5   | 1e-7   | < 1e-5        |

Stiffer and higher-dimensional systems scale these bounds; the figures
above are for a 1-D well-conditioned test and should be treated as a
best-case floor, not a guarantee for arbitrary user systems.

## Module-boundary dtype discipline

Blocks and simulation code should:

- Use `jnp.asarray(x)` (not `jnp.array(x, dtype=jnp.float32)`) when
  accepting user input, so the dtype respects the ambient x64 setting.
- Declare continuous-state default values as plain Python scalars or
  `jnp.array([...])` (no explicit dtype) unless the block's physics
  demands a specific width.
- Avoid `np.float32` / `np.float64` casts in ODE right-hand sides.
  Promotion from the state array's dtype is automatic and correct.
- When a block *must* pin a dtype (e.g. `library/data_source.py`
  reading CSV data always into `float64`), document the reason inline.

## Edge cases exercised by the test suite

- **Stiff oscillator** (`test_stiff_system_bdf`): Van der Pol with
  µ = 100 over 10 s.  BDF must handle this; Dopri5 would stall.
  Checks that the trajectory matches a reference solution within
  `2e-3`.
- **Near-singular Jacobian** (`test_near_singular_pendulum`): pendulum
  driven close to the upright equilibrium.  Solver accuracy should
  degrade gracefully; this test checks that the simulator completes
  without NaN and reports a final state within 10× nominal error of
  a reference.
- **Long-horizon oscillator**
  (`test_long_horizon_energy_drift_float64`): 10 000 periods of a
  simple harmonic oscillator under Dopri5 at default tolerances.
  Relative energy drift must stay below `1e-4`.
- **Float32 regression** (`test_float32_loose_tolerance`): documents
  what float32 + loose-tolerance behaviour actually looks like;
  guards against a silent precision regression.

## Changing this policy

Precision is a load-bearing choice for many downstream users.  Do not
tighten the default tolerances or narrow the dtype envelope without:

1. Running all four tests in `test/precision/` across every solver.
2. Confirming the change does not regress T-001's gradient suite or
   T-004's conservation suite.
3. Documenting the motivating investigation in the commit message.
