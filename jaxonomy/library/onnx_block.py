# SPDX-License-Identifier: MIT
"""
ONNX inference block (T-023).

Wraps a pre-trained model saved as ``.onnx`` and exposes it as a
:class:`LeafSystem` block.  Inference runs through ``onnxruntime`` via
:func:`jax.pure_callback` — gradients through the block are not
defined under reverse-mode autodiff (the spec calls this "best-effort";
users who need gradients should hold-the-line in JAX with an
``onnx2jax``-style conversion path, tracked as T-023a).

Example::

    from jaxonomy.library import ONNX

    model = ONNX(
        file_name="my_model.onnx",
        num_inputs=1,
        num_outputs=1,
    )

The model's input names map to the block's input ports in declaration
order; same for outputs.  Pre/post-processing (e.g. normalisation,
softmax) belong as separate Jaxonomy blocks upstream / downstream so
that the inference call stays a clean inputs → outputs map.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..framework import LeafSystem, parameters
from ..framework.system_base import UpstreamEvalError
from ..lazy_loader import LazyLoader
from ..logging import logger


ort = LazyLoader("onnxruntime", globals(), "onnxruntime")


__all__ = ["ONNX"]


class ONNX(LeafSystem):
    """ONNX inference block.

    Args:
        file_name: Path to the ``.onnx`` model file.
        num_inputs: Number of input tensors the model expects.
        num_outputs: Number of output tensors the model produces.
        cast_outputs_to_dtype: Optional ``jnp`` dtype name (``"float32"``,
            ``"float64"``, etc.) to cast every output to.  If None, the
            output dtype matches the model's output spec.
        providers: ``onnxruntime`` execution providers; defaults to CPU.
            Pass e.g. ``("CUDAExecutionProvider", "CPUExecutionProvider")``
            on a GPU host.
        name: Optional block name.

    Notes:
        Differentiability is **best-effort** — ``jax.pure_callback`` does
        not define a VJP, so reverse-mode autodiff through the block
        raises.  For end-to-end gradients, look at the T-023a follow-up
        on a JAX-traceable conversion (``onnx2jax`` or similar).

    .. note:: **float32 artifacts under jaxonomy's global x64.**
       ``import jaxonomy`` enables ``jax_enable_x64`` process-wide, so a
       float32 ONNX model receives float64 inputs unless you cast at the
       block boundary — a silent arithmetic change relative to the
       framework the model was exported and validated in.  One-line
       idiom: pass ``cast_outputs_to_dtype="float32"`` and feed the
       block ``x.astype(jnp.float32)`` inputs.
    """

    @parameters(
        static=[
            "file_name",
            "num_inputs",
            "num_outputs",
            "cast_outputs_to_dtype",
            "providers",
        ]
    )
    def __init__(
        self,
        file_name: str,
        num_inputs: int = 1,
        num_outputs: int = 1,
        cast_outputs_to_dtype=None,
        providers=("CPUExecutionProvider",),
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
        providers=("CPUExecutionProvider",),
    ):
        if num_inputs != self._num_inputs:
            raise ValueError(
                f"ONNX: num_inputs cannot be changed after construction "
                f"({self._num_inputs} → {num_inputs})."
            )
        if num_outputs != self._num_outputs:
            raise ValueError(
                f"ONNX: num_outputs cannot be changed after construction "
                f"({self._num_outputs} → {num_outputs})."
            )

        self._dtype_output = (
            getattr(jnp, cast_outputs_to_dtype)
            if cast_outputs_to_dtype is not None
            else None
        )

        self._session = ort.InferenceSession(
            file_name, providers=list(providers),
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        if len(self._input_names) != self._num_inputs:
            raise ValueError(
                f"ONNX: model has {len(self._input_names)} inputs but "
                f"the block declared num_inputs={self._num_inputs}.  "
                f"Model input names: {self._input_names}"
            )
        if len(self._output_names) != self._num_outputs:
            raise ValueError(
                f"ONNX: model has {len(self._output_names)} outputs but "
                f"the block declared num_outputs={self._num_outputs}.  "
                f"Model output names: {self._output_names}"
            )

    # ── shape inference and runtime ───────────────────────────────────────

    def initialize_static_data(self, context):
        try:
            inputs = self.collect_inputs(context)
            outs = self._pure_callback(*inputs)
            self._result_type = [
                jax.ShapeDtypeStruct(o.shape, o.dtype) for o in outs
            ]
        except UpstreamEvalError:
            logger.debug(
                "ONNX.initialize_static_data: UpstreamEvalError, deferring "
                "shape inference until root context is built."
            )
        return super().initialize_static_data(context)

    def _evaluate(self, time, state, *inputs, **params):
        return jax.pure_callback(
            self._pure_callback,
            self._result_type,
            *inputs,
        )

    def _pure_callback(self, *inputs):
        feed = {
            name: np.asarray(value)
            for name, value in zip(self._input_names, inputs)
        }
        outputs_np = self._session.run(self._output_names, feed)
        if self._dtype_output is not None:
            outputs_jax = [jnp.array(o, self._dtype_output) for o in outputs_np]
        else:
            outputs_jax = [jnp.array(o) for o in outputs_np]
        return outputs_jax
