# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple, Optional
import dataclasses

import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import io_callback

from ..results_data import AbstractResultsData

if TYPE_CHECKING:
    from ...framework import SystemCallback, ContextBase
    from ..typing import Array


__all__ = ["JaxResultsData", "PerSignalBuffer", "InterpolantBuffer"]


class InterpolantBuffer(NamedTuple):
    """T-012a-followup: per-step solver-interpolant ring buffer.

    Holds a ``(buffer_length,)`` ring of solver-state samples — one
    record per simulator save point.  When populated, ``query`` can
    locate the bracket containing a query time and evaluate the
    solver's native polynomial interpolant for sub-ULP accuracy
    (as opposed to PCHIP over the recorded ``(time, outputs)``).

    Attributes:
        t_prev: shape ``(buffer_length,)`` — start of each minor-step
            bracket the just-completed solver step covered (i.e.
            ``Dopri5State.t_prev`` at recording time).
        t_step: shape ``(buffer_length,)`` — end of each bracket
            (``Dopri5State.t``).  ``t_eval ∈ [t_prev, t_step]`` selects
            this slot.  Initialized to ``jnp.inf`` so the post-finalize
            ``isfinite`` mask trims unused entries the same way as the
            legacy ``time`` buffer.
        interp_coeff: shape ``(buffer_length, n_coeff, n_y)`` — the
            polynomial coefficients (Dopri5: ``n_coeff=5`` for the
            4th-order quartic).  Evaluated as
            ``unravel(jnp.polyval(coeff, (t-t_prev)/(t_step-t_prev)))``.

    Registered as a JAX pytree via NamedTuple's automatic registration.
    """

    t_prev: "Array"
    t_step: "Array"
    interp_coeff: "Array"


def _raise_buffer_overflow(n_steps, buffer_length):
    # FIXME (WC-291): Should this be a warning, a RuntimeError, or a custom exception?
    # Ideally this would actually just trigger dumping the results to NumPy arrays and
    # clearing the buffers.
    if n_steps > buffer_length:
        raise RuntimeError(
            f"Results buffer overflow: {n_steps} > {buffer_length} steps. "
            "Increase the buffer size to store all results"
        )


def error_buffer_overflow(n_steps, buffer_length):
    """Results-buffer-overflow diagnostic — see T-002b.

    Previously called ``jax.debug.callback(_raise_buffer_overflow, ...)``.
    The callback was an IO effect inside the simulator's ``lax.cond`` and
    broke ``simulate_batch(use_vmap=True)``.  Removed; silent overflow
    instead.  Users whose simulation exceeds the buffer length lose data
    beyond the buffer — increase ``buffer_length`` to avoid this.
    """
    del n_steps, buffer_length  # unused; kept for signature compatibility


class PerSignalBuffer(NamedTuple):
    """T-013a-followup-mode-a-buffers: per-signal `(times, values, count)` ring.

    Holds an independent recording buffer for one signal so unfired
    signals at a given major step do not consume a slot.

    Attributes:
        times: shape ``(buffer_length,)`` — the recorded sample times.
            Initialized to ``jnp.inf`` so unused entries can be trimmed
            by the same ``isfinite`` mask used by the legacy buffer.
        values: shape ``(buffer_length, *signal_shape)`` — the recorded
            sample values.  Unused entries are zero-initialized.
        valid_count: ``jnp.int32`` count of how many slots have been
            populated.  Used as the write index AND as the trim length
            in ``finalize``.

    Registered as a JAX pytree via NamedTuple's automatic registration —
    no manual flatten/unflatten needed.
    """

    times: "Array"
    values: "Array"
    valid_count: "Array"


def _make_empty_solution(
    context: ContextBase,
    recorded_signals: dict[str, SystemCallback],
    buffer_length: int,
    per_signal_classifications: Optional[dict[str, dict]] = None,
    interpolant_template: Optional[tuple[int, int]] = None,
) -> JaxResultsData:
    """Create an empty "buffer" solution data object with the correct shape.
    For each source in "recorded_signals", determine what the signal data type is
    and create an empty vector that can hold enough data to max out the simulation
    buffer.

    When ``per_signal_classifications`` is provided (T-013a-followup-mode-
    a-buffers), each signal also gets its own ``PerSignalBuffer``; the
    legacy shared-time ``time`` vector and ``outputs`` dict are still
    initialized so the default-off path (and any code reading them)
    remains unchanged.  The per-signal buffers are the source of truth
    in finalize when populated.
    """

    def _expand_template(source: SystemCallback):
        # Determine the data type of the signal (shape and dtype)
        x = source.eval(context)
        if jnp.isscalar(x):
            x = jnp.asarray(x)
        # Create a buffer that can hold the maximum number of (major, minor) steps
        return jnp.zeros((buffer_length, *x.shape), dtype=x.dtype)

    signals = {
        key: _expand_template(source) for key, source in recorded_signals.items()
    }
    # The time vector is used to determine the number of steps taken by the ODE solver
    # since diffrax will return inf for unused buffer entries. Then we can use isfinite
    # to trim the unused buffer space.  For this reason, initialize to inf rather
    # than zero.
    times = jnp.full((buffer_length,), jnp.inf)

    per_signal_buffers: Optional[dict[str, PerSignalBuffer]] = None
    if per_signal_classifications is not None:
        per_signal_buffers = {}
        for key, value_buf in signals.items():
            per_signal_buffers[key] = PerSignalBuffer(
                times=jnp.full((buffer_length,), jnp.inf),
                values=value_buf,
                valid_count=jnp.int32(0),
            )

    # T-012a-followup: optional per-step interpolant buffer.  Allocated
    # only when ``interpolant_template`` is supplied (=> the user opted
    # into ``record_solver_states=True``).  ``interpolant_template`` is
    # ``(n_coeff, n_y)`` from the solver's interp_coeff array; default
    # for Dopri5 is ``(5, n_y)``.
    interpolant_buffer: Optional[InterpolantBuffer] = None
    if interpolant_template is not None:
        n_coeff, n_y = interpolant_template
        interpolant_buffer = InterpolantBuffer(
            t_prev=jnp.full((buffer_length,), jnp.inf),
            t_step=jnp.full((buffer_length,), jnp.inf),
            interp_coeff=jnp.zeros((buffer_length, n_coeff, n_y)),
        )

    return JaxResultsData(
        source_dict=recorded_signals,
        outputs=signals,
        time=times,
        buffer_length=buffer_length,
        per_signal_buffers=per_signal_buffers,
        per_signal_classifications=per_signal_classifications,
        interpolant_buffer=interpolant_buffer,
    )


def _trim(solution: JaxResultsData) -> tuple[Array, dict[str, Array]]:
    """Remove unused entries from the buffer and return flattened arrays.

    See `JaxResultsData.finalize` for more details.

    T-017a: switched from JAX boolean indexing (``solution.time[valid_idx]``)
    to a host-side numpy slice.  The JAX path triggered a fresh XLA dispatch
    per call because the result of boolean indexing is a dynamically-shaped
    array; in ``simulate_batch``'s scan-kernel loop this added ~80 ms per
    batch element (~800 ms total at N=10).  Materialising the buffer to
    numpy first and slicing on the host removes that recurring cost — the
    output is identical (numpy arrays, same values).
    """
    # Adaptive ODE solvers should return inf for unused buffer entries.
    # Materialise the time buffer on the host and use a numpy boolean mask;
    # this avoids the per-call JAX dispatch cost of ``jax_array[bool_mask]``.
    np_time = np.asarray(solution.time)
    valid_idx = np.isfinite(np_time)
    time = np_time[valid_idx]

    outputs = {}
    for key, y in solution.outputs.items():
        outputs[key] = np.asarray(y)[valid_idx]

    # If there is stored NumPy data in the solution, add the buffer data to it.
    if solution.np_data.time is not None:
        time = np.append(solution.np_data.time, time, axis=0)
        solution.np_data.time = None
        for key, value in outputs.items():
            outputs[key] = np.append(solution.np_data.outputs[key], value, axis=0)

    return time, outputs


def _trim_per_signal(
    solution: JaxResultsData,
) -> tuple[Array, dict[str, Array], dict[str, Array]]:
    """T-013a-followup-mode-a-buffers: trim per-signal buffers post-finalize.

    Materialises each ``PerSignalBuffer`` on the host and slices it down
    to ``valid_count``.  The shared-time vector is still derived from the
    legacy buffer (it remains the source-of-truth for the global timeline
    and for any signal whose classification doesn't reduce its cadence).

    Returns ``(time, outputs, per_signal_times)`` where:
      - ``time`` is the legacy global time vector (length ``N``).
      - ``outputs`` is keyed by signal name and trimmed to each signal's
        per-signal valid-count (length matches ``per_signal_times[name]``).
      - ``per_signal_times`` is the matching dict of per-signal timestamps.
    """
    # Global time vector still derived from the shared buffer for
    # backwards-compat consumers (e.g. ``SimulationResults.time``).
    np_time = np.asarray(solution.time)
    valid_idx = np.isfinite(np_time)
    global_time = np_time[valid_idx]

    outputs: dict[str, np.ndarray] = {}
    per_signal_times: dict[str, np.ndarray] = {}
    for key, buf in solution.per_signal_buffers.items():
        count = int(np.asarray(buf.valid_count))
        # Defensive: clamp to the buffer length to handle the silent
        # overflow path (T-002b).
        count = max(0, min(count, int(np.asarray(buf.times).shape[0])))
        ts = np.asarray(buf.times)[:count]
        vs = np.asarray(buf.values)[:count]
        per_signal_times[key] = ts
        outputs[key] = vs

    return global_time, outputs, per_signal_times


@dataclasses.dataclass
class _NumpyData:
    """Class to store the solution data in NumPy arrays when the buffer is full.

    This doesn't seem like it should merit its own class, but this seems to be the
    only way to successfully store the data from within the JIT-compiled function.
    """

    time: np.ndarray = None
    outputs: dict[str, np.ndarray] = None

    def dump_buffer(self, buffer_full: bool, time: Array, outputs: dict[str, Array]):
        """If the solution buffer is full, store the results in NumPy arrays."""
        if not buffer_full:
            return

        # Dump the buffer to NumPy arrays
        if self.time is None:
            self.time = np.asarray(time)
            self.outputs = {key: np.asarray(value) for key, value in outputs.items()}

        else:
            self.time = np.append(self.time, np.asarray(time), axis=0)
            for key, value in outputs.items():
                self.outputs[key] = np.append(
                    self.outputs[key], np.asarray(value), axis=0
                )


def _signal_fired_this_step(
    classification: dict, time: "Array", atol_period: float = 1e-9,
) -> "Array":
    """Decide if a signal should be recorded at the current step.

    Returns a JAX-traceable scalar bool.  Three classifications:
      - ``continuous``: always fires (every major step).
      - ``periodic``: fires when ``|time - offset - k*period| <= tol``
        for some integer ``k``, plus always fires at ``time == 0`` so
        the initial sample is captured even when the first tick has
        not yet fired.
      - ``default`` / unknown: always fires (matches the legacy
        per-major-step cadence — Mode B post-finalize dedup will trim
        constants).

    The decision is a pure function of ``classification`` (static aux)
    and ``time`` (traced value); no plumbing through the simulator's
    timed-events collection is required.
    """
    kind = classification.get("kind", "default")
    if kind == "continuous" or kind == "default":
        return jnp.bool_(True)
    if kind == "periodic":
        period = float(classification["period"])
        offset = float(classification.get("offset", 0.0))
        if period <= 0:
            return jnp.bool_(True)
        shifted = time - offset
        k = jnp.round(shifted / period)
        residual = jnp.abs(shifted - k * period)
        tol = max(atol_period, 1e-9 * max(period, 1.0))
        is_tick = residual <= tol
        # Always record at t=0 so the initial sample lands in the buffer.
        is_initial = time == 0.0
        return jnp.logical_or(is_tick, is_initial)
    return jnp.bool_(True)


# Inherits docstring from `AbstractResultsData`
@dataclasses.dataclass
class JaxResultsData(AbstractResultsData):
    n_steps: int = 0  # Number of update() calls (== simulator save points)

    # Index of the next free buffer slot.  With the T-138 decimation
    # strategy this only moves forward within [0, buffer_length]; on
    # overflow the buffer is compacted in place and the index drops back
    # to the surviving-sample count (it no longer wraps to 0).
    buffer_index: int = 0
    buffer_length: int = None  # Maximum number of time stamps to save

    # T-138 — graceful overflow via uniform decimation.  ``record_stride``
    # is the current keep-every-Nth-sample stride (starts at 1 == keep
    # everything); ``step_count`` counts update() calls.  A sample is
    # recorded iff ``step_count % record_stride == 0``.  When the buffer
    # fills, the even-position samples are compacted into the lower half
    # and the stride doubles, so the recorded trajectory always spans
    # [t0, <latest>] at uniform (in step count) reduced resolution
    # instead of silently dropping the head (the pre-T-138 ring-wrap).
    # Both are traced (pytree children) so the scheme is jit/vmap-safe.
    record_stride: int = 1
    step_count: int = 0

    # Data stored in numpy arrays as the buffer fills up
    np_data: _NumpyData = dataclasses.field(default_factory=_NumpyData)

    # T-013a-followup-mode-a-buffers: per-signal `(times, values, count)`
    # buffers.  When ``None`` (default) the legacy single-shared-time
    # path is used and behaviour is byte-equivalent to before.  When
    # populated, ``update`` ALSO writes to per-signal buffers using the
    # cadence classifications (only signals whose classification matches
    # the current time consume a slot in their own buffer).  Finalize
    # uses the per-signal buffers as the source-of-truth for the
    # post-trim ``outputs`` and ``per_signal_times``.
    per_signal_buffers: Optional[dict[str, PerSignalBuffer]] = None

    # Static metadata (NOT traced): cadence classifications keyed by
    # signal name.  Each entry is one of ``{"kind": "continuous"}``,
    # ``{"kind": "periodic", "period": float, "offset": float}``, or
    # ``{"kind": "default"}``.  See ``ResultsRecorder.classify_signal_cadence``.
    per_signal_classifications: Optional[dict[str, dict]] = None

    # T-012a-followup: per-step solver-interpolant buffer.  When non-None,
    # ``update`` accepts an ``ode_solver_state`` and snapshots its
    # ``(t_prev, t, interp_coeff)`` into the ring; finalize emits the
    # trimmed segments alongside ``time`` / ``outputs``.  Default-off
    # path (record_solver_states=False) is byte-equivalent.
    interpolant_buffer: Optional[InterpolantBuffer] = None

    @staticmethod
    def initialize(
        context: ContextBase,
        recorded_signals: dict[str, SystemCallback],
        buffer_length: int,
        per_signal_classifications: Optional[dict[str, dict]] = None,
        interpolant_template: Optional[tuple[int, int]] = None,
    ) -> JaxResultsData:
        return _make_empty_solution(
            context,
            recorded_signals,
            buffer_length,
            per_signal_classifications=per_signal_classifications,
            interpolant_template=interpolant_template,
        )

    def update(
        self,
        context: ContextBase,
        ode_solver_state: Optional[object] = None,
    ) -> JaxResultsData:
        """Update the simulation solution with the results of a simulation step.

        This stores the current state of the system in the buffer arrays corresponding
        to the recorded signals.

        Args:
            context (ContextBase):
                The simulation context at the end of the simulation step.

        Returns:
            JaxResultsData: The updated simulation solution data.
        """

        # Index of the current major step in the solution data buffer.
        index = self.buffer_index

        # initialize the time buffer
        self.time = jnp.where(index == 0, jnp.full_like(self.time, jnp.inf), self.time)

        # T-138 decimation: only every ``record_stride``-th update call
        # consumes a buffer slot.  Unfired calls write to an out-of-bounds
        # index with mode="drop" — an O(1) no-op — so the hot path stays
        # at legacy cost while stride == 1 (the pre-overflow common case).
        fired = (self.step_count % self.record_stride) == 0

        # In this case we only need to get the signal at the current step,
        # since there are no intermediate steps from the ODE solver.
        y = self.eval_sources(context)

        write_idx = jnp.where(fired, index, self.buffer_length)
        outputs = {
            key: self.outputs[key].at[write_idx].set(y[key], mode="drop")
            for key in self.source_dict
        }

        # Set the current entry of the time vector to the current time.
        # Unused entries stay inf, indicating unused buffer slots.
        time = self.time.at[write_idx].set(context.time, mode="drop")

        buffer_index = index + jnp.where(fired, 1, 0)

        # T-138 — graceful overflow.  Historically the buffer wrapped to 0
        # here (after T-002b removed the io_callback dump that broke
        # ``simulate_batch(use_vmap=True)``), silently keeping only the
        # trajectory *tail*.  Now, when the buffer fills, the even-position
        # samples are compacted into the lower half, the write index drops
        # to the surviving count, and the stride doubles — a uniform
        # decimation that always spans [t0, <latest>].  ``results.time[0]``
        # is therefore guaranteed to be the simulation start time
        # regardless of horizon, at the cost of resolution (each overflow
        # halves the sampling density).
        #
        # lax.cond keeps the O(buffer_length) compaction off the hot path
        # under plain jit (one branch executes).  Under vmap the cond
        # lowers to a select that evaluates both branches every step —
        # a bandwidth cost proportional to the buffer size; size
        # ``buffer_length`` to the expected sample count if recording
        # throughput matters in large ensembles.
        buffer_full = buffer_index >= self.buffer_length
        half = (self.buffer_length + 1) // 2  # static: surviving-sample count

        def _compact(operand):
            t, outs = operand
            new_t = jnp.full_like(t, jnp.inf).at[:half].set(t[::2])
            new_outs = {k: v.at[:half].set(v[::2]) for k, v in outs.items()}
            return new_t, new_outs

        time, outputs = lax.cond(
            buffer_full, _compact, lambda operand: operand, (time, outputs)
        )
        buffer_index = jnp.where(buffer_full, half, buffer_index)
        record_stride = jnp.where(
            buffer_full, self.record_stride * 2, self.record_stride
        )

        # T-013a-followup-mode-a-buffers: when per-signal buffers are
        # active, ALSO append to each signal's own ring conditional on
        # its cadence classification.  Continuous and "default"-kind
        # signals append every step (matching the legacy buffer);
        # periodic signals append only on their schedule, so unfired
        # signals do not consume a slot.
        per_signal_buffers = self.per_signal_buffers
        if per_signal_buffers is not None and self.per_signal_classifications is not None:
            new_per_signal: dict[str, PerSignalBuffer] = {}
            t = context.time
            for key, buf in per_signal_buffers.items():
                cls = self.per_signal_classifications.get(
                    key, {"kind": "default"},
                )
                fired = _signal_fired_this_step(cls, t)
                # Where to write: the current valid_count.
                write_idx = buf.valid_count
                new_t = jnp.where(
                    fired,
                    buf.times.at[write_idx].set(t),
                    buf.times,
                )
                new_v = jnp.where(
                    fired,
                    buf.values.at[write_idx].set(y[key]),
                    buf.values,
                )
                new_count = buf.valid_count + jnp.where(
                    fired, jnp.int32(1), jnp.int32(0),
                )
                # Saturate at buffer length to mirror the legacy
                # silent-overflow semantics.
                new_count = jnp.minimum(new_count, jnp.int32(self.buffer_length))
                new_per_signal[key] = PerSignalBuffer(
                    times=new_t, values=new_v, valid_count=new_count,
                )
        else:
            new_per_signal = per_signal_buffers

        # T-012a-followup: snapshot the solver's per-minor-step
        # interpolant data when the buffer was allocated AND the caller
        # passed in the live ode_solver_state.  The first call after
        # init writes a degenerate (t_prev == t) entry — query() skips
        # any segment with zero width.
        new_interp_buf = self.interpolant_buffer
        if new_interp_buf is not None and ode_solver_state is not None:
            ic = getattr(ode_solver_state, "interp_coeff", None)
            tp = getattr(ode_solver_state, "t_prev", None)
            tt = getattr(ode_solver_state, "t", None)
            if ic is not None and tp is not None and tt is not None:
                new_interp_buf = InterpolantBuffer(
                    t_prev=new_interp_buf.t_prev.at[index].set(tp),
                    t_step=new_interp_buf.t_step.at[index].set(tt),
                    interp_coeff=new_interp_buf.interp_coeff.at[index].set(ic),
                )

        return dataclasses.replace(
            self,
            outputs=outputs,
            time=time,
            n_steps=self.n_steps + 1,
            buffer_index=buffer_index,
            record_stride=record_stride,
            step_count=self.step_count + 1,
            per_signal_buffers=new_per_signal,
            interpolant_buffer=new_interp_buf,
        )

    def finalize(self) -> tuple[Array, dict[str, Array]]:
        """Trim unused buffer space from the solution data.

        The raw solution data contains the full 'buffer' of simulation steps. This function
        trims the unused buffer space from the solution data.

        Because this returns variable-length arrays depending on the results of the solver
        calls, it cannot be called from a JAX jit-compiled function.  Instead, call as part
        of a 'postprocessing' step after simulation is complete.  This is done by default
        if the simulation is invoked via the `simulate` function.

        When ``per_signal_buffers`` is populated (T-013a-followup-mode-a-
        buffers), the per-signal arrays are NOT used here — the legacy
        global trim is preserved so existing callers see the same shape.
        Use :meth:`finalize_per_signal` to read the per-signal buffers.
        """
        return _trim(self)

    def finalize_interpolant(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """T-012a-followup: trim and return the per-step interpolant ring.

        Returns ``None`` when ``record_solver_states=False`` (default-
        off; the buffer was never allocated).  Otherwise returns the
        triple ``(t_prev, t_step, interp_coeff)`` as numpy arrays,
        with unused trailing slots dropped.  Slots where ``t_prev``
        equals ``t_step`` (the dummy initial state at simulation start
        before any solver step ran) are also dropped — those represent
        zero-width brackets that cannot host an interpolant query.
        """
        if self.interpolant_buffer is None:
            return None
        t_prev = np.asarray(self.interpolant_buffer.t_prev)
        t_step = np.asarray(self.interpolant_buffer.t_step)
        interp = np.asarray(self.interpolant_buffer.interp_coeff)
        valid = np.isfinite(t_prev) & np.isfinite(t_step) & (t_step > t_prev)
        return t_prev[valid], t_step[valid], interp[valid]

    def finalize_per_signal(
        self,
    ) -> Optional[tuple[Array, dict[str, Array], dict[str, Array]]]:
        """T-013a-followup-mode-a-buffers: trim per-signal buffers.

        Returns ``None`` when per-signal buffers were not allocated
        (default-off and Mode B paths).  Otherwise returns
        ``(global_time, outputs, per_signal_times)`` with each
        ``outputs[name]`` and ``per_signal_times[name]`` trimmed to that
        signal's ``valid_count`` — typically much shorter than the
        global time vector for periodic signals.
        """
        if self.per_signal_buffers is None:
            return None
        return _trim_per_signal(self)

    @classmethod
    def _scan(cls, *args, **kwargs):
        return lax.scan(*args, **kwargs)


#
# Register as custom pytree nodes
#    https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
#
def _solution_flatten(solution: JaxResultsData):
    """Flatten the solution data for tracing."""
    children = (
        solution.time,
        solution.outputs,
        solution.n_steps,
        solution.buffer_index,
        solution.record_stride,
        solution.step_count,
        solution.per_signal_buffers,
        solution.interpolant_buffer,
    )
    aux_data = (
        solution.source_dict,
        solution.buffer_length,
        solution.np_data,
        solution.per_signal_classifications,
    )
    return children, aux_data


def _solution_unflatten(aux_data, children):
    """Unflatten the solution data after tracing."""
    (
        time,
        outputs,
        n_steps,
        buffer_index,
        record_stride,
        step_count,
        per_signal_buffers,
        interpolant_buffer,
    ) = children
    source_dict, buffer_length, np_data, per_signal_classifications = aux_data
    return JaxResultsData(
        source_dict=source_dict,
        time=time,
        outputs=outputs,
        n_steps=n_steps,
        buffer_index=buffer_index,
        record_stride=record_stride,
        step_count=step_count,
        buffer_length=buffer_length,
        np_data=np_data,
        per_signal_buffers=per_signal_buffers,
        per_signal_classifications=per_signal_classifications,
        interpolant_buffer=interpolant_buffer,
    )


jax.tree_util.register_pytree_node(
    JaxResultsData,
    _solution_flatten,
    _solution_unflatten,
)
