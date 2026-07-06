# SPDX-License-Identifier: MIT
"""
JAX-traceable ONNX inference block (T-023a).

Where :class:`ONNX` (T-023) wraps an ONNX model in
``jax.pure_callback`` — host-side onnxruntime, no defined VJP — this
block runs the ONNX graph through ``jaxonnxruntime``, a JAX-native
ONNX executor.  Inference is a regular JAX computation, so:

  - ``jax.grad`` returns a real gradient through the model.
  - ``jax.jit`` compiles the inference into the surrounding jit.
  - ``jax.vmap`` vectorises naturally.
  - The block plays well with autodiff-driven optimisation,
    end-to-end RL gradient flow, etc.

Op coverage is whatever ``jaxonnxruntime`` implements.  Models that
use unsupported ops fail to ``prepare`` with an explicit error
message; consider falling back to the host-callback :class:`ONNX`
block in that case.

Usage::

    from jaxonomy.library import ONNXJax

    blk = ONNXJax(file_name="model.onnx", num_inputs=1, num_outputs=1)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..framework import LeafSystem, parameters
from ..framework.system_base import UpstreamEvalError
from ..lazy_loader import LazyLoader
from ..logging import logger


onnx = LazyLoader("onnx", globals(), "onnx")
jort_backend = LazyLoader(
    "jort_backend", globals(), "jaxonnxruntime.backend",
)


__all__ = ["ONNXJax"]


class ONNXJax(LeafSystem):
    """JAX-traceable ONNX inference (T-023a).

    Args:
        file_name: Path to the ``.onnx`` model file.
        num_inputs: Number of input tensors the model expects.
        num_outputs: Number of output tensors the model produces.
        cast_outputs_to_dtype: Optional ``jnp`` dtype name to cast every
            output to (``"float32"`` / ``"float64"`` / ...).
        name: Optional block name.

    Differentiability: end-to-end via ``jaxonnxruntime``'s JAX
    primitive implementations.  Op coverage failure shows up at
    initialize() time as a clear ``RuntimeError`` from
    ``jaxonnxruntime``.

    For models that use ops outside ``jaxonnxruntime``'s coverage,
    fall back to :class:`ONNX` — same constructor signature, runs via
    ``onnxruntime`` host callback (no autodiff).
    """

    @parameters(
        static=[
            "file_name", "num_inputs", "num_outputs", "cast_outputs_to_dtype",
        ]
    )
    def __init__(
        self,
        file_name: str,
        num_inputs: int = 1,
        num_outputs: int = 1,
        cast_outputs_to_dtype=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._num_inputs = int(num_inputs)
        self._num_outputs = int(num_outputs)

        for _ in range(self._num_inputs):
            self.declare_input_port()

        def _make_output_callback(idx):
            def _cb(time, state, *inputs, **params):
                outs = self._evaluate(time, state, *inputs, **params)
                return outs[idx]
            return _cb

        for i in range(self._num_outputs):
            self.declare_output_port(
                _make_output_callback(i), requires_inputs=True,
            )

    def initialize(
        self,
        file_name: str,
        num_inputs: int = 1,
        num_outputs: int = 1,
        cast_outputs_to_dtype=None,
    ):
        if num_inputs != self._num_inputs:
            raise ValueError(
                f"ONNXJax: num_inputs cannot be changed after construction "
                f"({self._num_inputs} → {num_inputs})."
            )
        if num_outputs != self._num_outputs:
            raise ValueError(
                f"ONNXJax: num_outputs cannot be changed after construction "
                f"({self._num_outputs} → {num_outputs})."
            )

        self._dtype_output = (
            getattr(jnp, cast_outputs_to_dtype)
            if cast_outputs_to_dtype is not None
            else None
        )

        # Load model + prepare a JAX-callable executor.
        model = onnx.load(file_name)
        try:
            self._rep = jort_backend.Backend.prepare(model)
        except Exception as e:
            raise RuntimeError(
                f"ONNXJax.initialize: jaxonnxruntime failed to prepare "
                f"model {file_name!r}: {e}.  This usually means the "
                "model uses ops that jaxonnxruntime hasn't implemented "
                "yet.  Fall back to ONNX (host-callback) if you don't "
                "need gradients."
            ) from e

        # Cache input/output names for ordering.
        self._input_names = [i.name for i in model.graph.input]
        # Filter to real inputs only (initializer-shadowed inputs are not user-supplied)
        initializer_names = {init.name for init in model.graph.initializer}
        self._input_names = [n for n in self._input_names if n not in initializer_names]
        self._output_names = [o.name for o in model.graph.output]

        if len(self._input_names) != self._num_inputs:
            raise ValueError(
                f"ONNXJax: model has {len(self._input_names)} inputs "
                f"but the block declared num_inputs={self._num_inputs}.  "
                f"Model inputs: {self._input_names}"
            )
        if len(self._output_names) != self._num_outputs:
            raise ValueError(
                f"ONNXJax: model has {len(self._output_names)} outputs "
                f"but the block declared num_outputs={self._num_outputs}.  "
                f"Model outputs: {self._output_names}"
            )

    # ── shape inference & runtime ─────────────────────────────────────────

    def initialize_static_data(self, context):
        try:
            inputs = self.collect_inputs(context)
            outs = self._run(inputs)
            self._result_type = [
                jax.ShapeDtypeStruct(o.shape, o.dtype) for o in outs
            ]
        except UpstreamEvalError:
            logger.debug(
                "ONNXJax.initialize_static_data: UpstreamEvalError, "
                "deferring shape inference until root context is built."
            )
        return super().initialize_static_data(context)

    def _run(self, inputs):
        outs = self._rep.run(list(inputs))
        if self._dtype_output is not None:
            outs = [jnp.asarray(o, self._dtype_output) for o in outs]
        else:
            outs = [jnp.asarray(o) for o in outs]
        return outs

    def _evaluate(self, time, state, *inputs, **params):
        # Run inline (not via pure_callback) — jaxonnxruntime emits
        # JAX primitives directly.
        return self._run(inputs)
