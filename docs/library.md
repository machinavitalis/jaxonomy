# Block library

**Notes on imported neural models** (`ONNX` / `ONNXJax` / `PyTorch` / `TensorFlow`):

- **x64 at import.** `import jaxonomy` enables JAX 64-bit mode
  (`jax_enable_x64`) process-wide, so float32 artifacts see float64 inputs
  and silently compute in different arithmetic than they were trained and
  validated in. Cast explicitly at block boundaries:
  `cast_outputs_to_dtype="float32"` on the block, `x.astype(jnp.float32)`
  on upstream signals.
- **Discrete-time policies need a `ZeroOrderHold`.** A sample-and-hold
  controller exported from a discrete-time training loop
  (torch/NEUROMANCER-style) is otherwise re-evaluated at every ODE solver
  stage, silently destroying step-for-step parity with the exporting
  framework. Follow the policy block with `ZeroOrderHold(dt=ts)` and pin
  the step grid with `SimulatorOptions(max_major_step_length=ts,
  max_minor_step_size=ts)` — with that, closed-loop parity is ~4e-8 over
  400 steps on the two-tank benchmark.

## ::: jaxonomy.library
