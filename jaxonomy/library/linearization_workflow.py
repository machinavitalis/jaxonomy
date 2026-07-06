# SPDX-License-Identifier: MIT

"""Linearization workflow primitives (T-109 phase 1 + followup-fre).

This module ships small, decoupled helpers that sit on top of the existing
:func:`jaxonomy.library.linear_system.linearize` API:

* :func:`frequency_response` — evaluate ``G(jω) = C (jωI − A)⁻¹ B + D`` for a
  :class:`LinearizedSystem` over a vector of angular frequencies.  Returns a
  ``FrequencyResponse`` dataclass with complex response, magnitudes and phases.
* :func:`bode_data` — convenience that returns frequencies (Hz), magnitudes
  in dB and phases in degrees, ready for matplotlib (no plotting dependency
  inside ``jaxonomy``).
* :func:`findop` — a Newton-iteration operating-point solver that finds a
  continuous state ``x*`` such that ``ẋ(x*, u₀) = 0`` for a fixed input ``u₀``
  taken from the supplied ``base_context``.  Built on ``jax.jacrev`` and
  ``npa.linalg.solve`` so the residual remains differentiable through the
  initial guess and through model parameters.
* :func:`estimate_frequency_response` (T-109 followup-fre) — empirical
  transfer-function estimation by simulating the diagram with an excitation
  signal (chirp / PRBS / band-limited noise) on a chosen input port and
  computing ``G(f) = Sxy(f) / Sxx(f)`` from the recorded input/output
  trajectories.  Returns the same :class:`FrequencyResponse` NamedTuple as
  :func:`frequency_response` so downstream code (``bode_data`` etc.) is
  drop-in interchangeable.

All helpers stay strictly inside ``jaxonomy.library`` and do not touch
``simulator.py``, ``lazy_results.py`` or ``library/primitives.py``.
"""

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxonomy.backend import numpy_api as npa
from .linear_system import LinearizedSystem

__all__ = [
    "FrequencyResponse",
    "frequency_response",
    "bode_data",
    "findop",
    "estimate_frequency_response",
    "pole_zero_map",
    "step_response",
    "impulse_response",
    "nyquist_data",
    "discretize",
    "with_observer",
]


@dataclass
class FrequencyResponse:
    """Result of a frequency-response evaluation.

    Attributes:
        omegas: Angular frequency vector ``ω`` in rad/s, shape ``(K,)``.
        response: Complex frequency response ``G(jω)``, shape
            ``(K, n_outputs, n_inputs)``.
        magnitudes: ``|G(jω)|``, shape ``(K, n_outputs, n_inputs)``.
        phases: Phase ``arg G(jω)`` in radians, shape
            ``(K, n_outputs, n_inputs)``.
    """

    omegas: Any  # jax.Array, shape (K,)
    response: Any  # complex jax.Array, shape (K, p, m)
    magnitudes: Any  # real jax.Array, shape (K, p, m)
    phases: Any  # real jax.Array, shape (K, p, m)


def _ensure_2d(M, n_rows, n_cols):
    """Promote a possibly-1d state-space matrix to ``(n_rows, n_cols)``."""
    M = jnp.asarray(M)
    if M.ndim == 0:
        M = M.reshape((1, 1))
    elif M.ndim == 1:
        # Heuristic: row vector if it matches n_cols, else column vector
        if M.size == n_cols and n_rows == 1:
            M = M.reshape((1, n_cols))
        elif M.size == n_rows and n_cols == 1:
            M = M.reshape((n_rows, 1))
        else:
            M = M.reshape((n_rows, n_cols))
    return M


def frequency_response(linsys: LinearizedSystem, omegas) -> FrequencyResponse:
    """Compute the frequency response of a linearized state-space system.

    For a continuous-time LTI ``ẋ = Ax + Bu, y = Cx + Du`` the transfer function
    evaluated at ``s = jω`` is the ``(p, m)`` transfer-function matrix
    ``G(s) = C (sI − A)⁻¹ B + D``.  This helper vectorises that evaluation
    across an ``omegas`` array and naturally handles MIMO systems
    (``m > 1`` inputs and/or ``p > 1`` outputs) — the returned array shape
    is always ``(K, p, m)``.  SISO is the special case ``p = m = 1`` which
    produces shape ``(K, 1, 1)``.

    Args:
        linsys: A :class:`LinearizedSystem` (typically produced by
            :func:`linearize`).  ``A`` is ``(n, n)``, ``B`` is ``(n, m)``,
            ``C`` is ``(p, n)``, ``D`` is ``(p, m)``.
        omegas: 1-D array-like of angular frequencies ``ω`` (rad/s).

    Returns:
        :class:`FrequencyResponse` with ``omegas`` (shape ``(K,)``), complex
        ``response`` (shape ``(K, p, m)``), and corresponding ``magnitudes``
        and ``phases`` (radians).  For MIMO ``response[k, i, j]`` is the
        transfer function from input ``j`` to output ``i`` evaluated at
        ``ω = omegas[k]``.

    Notes:
        The implementation is fully JAX-traceable and differentiable through
        ``A, B, C, D`` and ``omegas`` so it composes with ``jax.grad`` and
        ``jax.vmap``.  ``jnp.linalg.solve`` solves the matrix RHS ``B`` in
        one shot per frequency so the per-omega cost is one LU decomposition
        plus ``m`` triangular back-substitutions — substantially cheaper than
        looping over input channels.
    """
    omegas = jnp.asarray(omegas)
    if omegas.ndim == 0:
        omegas = omegas.reshape((1,))

    A = jnp.asarray(linsys.A)
    if A.ndim == 0:
        A = A.reshape((1, 1))
    elif A.ndim == 1:
        n = A.size
        A = A.reshape((n, n))
    n = A.shape[0]

    B = _ensure_2d(linsys.B, n, max(jnp.asarray(linsys.B).size // n, 1))
    m = B.shape[1]

    C_arr = jnp.asarray(linsys.C)
    if C_arr.ndim <= 1:
        # Single-output row vector or scalar
        C_arr = C_arr.reshape((max(C_arr.size // n, 1), n))
    p = C_arr.shape[0]

    D = _ensure_2d(linsys.D, p, m)

    # Promote to complex once.
    A_c = A.astype(jnp.complex64) if A.dtype == jnp.float32 else A.astype(jnp.complex128)
    B_c = B.astype(A_c.dtype)
    C_c = C_arr.astype(A_c.dtype)
    D_c = D.astype(A_c.dtype)
    eye = jnp.eye(n, dtype=A_c.dtype)

    # Discrete-time systems evaluate the transfer function at z = e^{jωΔt}
    # rather than at s = jω; the matrices (A, B, C, D) are already the
    # discrete realization. ``is_discrete`` is a Python bool, so the branch
    # is resolved statically under jit/vmap.
    is_discrete = linsys.dt is not None
    dt = linsys.dt

    def one(omega):
        if is_discrete:
            s = jnp.exp(1j * omega * dt).astype(A_c.dtype)
        else:
            s = (1j * omega).astype(A_c.dtype)
        # Solve (sI − A) X = B   →   X = (sI − A)⁻¹ B
        # (for discrete systems s is the z-variable e^{jωΔt}).
        X = jnp.linalg.solve(s * eye - A_c, B_c)
        return C_c @ X + D_c

    response = jax.vmap(one)(omegas)  # shape (K, p, m)
    magnitudes = jnp.abs(response)
    phases = jnp.angle(response)
    return FrequencyResponse(
        omegas=omegas,
        response=response,
        magnitudes=magnitudes,
        phases=phases,
    )


def bode_data(linsys: LinearizedSystem, omegas):
    """Return matplotlib-ready Bode arrays for ``linsys``.

    Handles MIMO systems out of the box: when the underlying
    :func:`frequency_response` returns shape ``(K, p, m)`` with ``p > 1``
    or ``m > 1``, the returned ``magnitude_db`` / ``phase_deg`` arrays
    keep the same ``(K, p, m)`` shape — one Bode pair per
    ``(output, input)`` channel pair.  The phase is unwrapped along the
    frequency axis (``axis=0``) independently for each channel, which is
    the standard convention for MIMO Bode plots.

    Args:
        linsys: A :class:`LinearizedSystem` (``p`` outputs, ``m`` inputs).
        omegas: 1-D array-like of angular frequencies ``ω`` (rad/s).

    Returns:
        Dictionary with keys:
            ``"omega"`` — angular frequencies (rad/s), shape ``(K,)``;
            ``"freq_hz"`` — ``ω / (2π)`` for log-Hz plotting, shape ``(K,)``;
            ``"magnitude_db"`` — ``20 log₁₀ |G(jω)|`` (dB); shape ``(K,)``
                for SISO (squeezed for backward-compatibility with phase 1),
                shape ``(K, p, m)`` for MIMO;
            ``"phase_deg"`` — phase in degrees, unwrapped along the
                frequency axis; same shape as ``magnitude_db``.
    """
    fr = frequency_response(linsys, omegas)
    mag = fr.magnitudes
    phase = fr.phases
    # Detect SISO from the underlying (K, p, m) shape; for SISO squeeze
    # to 1-D (K,) for backward-compatibility with the T-109 phase-1 API.
    is_siso = (mag.ndim == 3 and mag.shape[-1] == 1 and mag.shape[-2] == 1)
    if is_siso:
        mag = mag[..., 0, 0]
        phase = phase[..., 0, 0]
        # axis=-1 == axis=0 here; either works on a 1-D array.
        phase_deg = jnp.unwrap(phase) * (180.0 / jnp.pi)
    else:
        # MIMO: unwrap along the frequency axis (axis=0), independently per
        # (output, input) channel pair.  Using axis=-1 (the default) would
        # unwrap along the input axis and produce a meaningless mixture
        # across channels.
        phase_deg = jnp.unwrap(phase, axis=0) * (180.0 / jnp.pi)
    mag_db = 20.0 * jnp.log10(jnp.maximum(mag, 1e-300))
    return {
        "omega": fr.omegas,
        "freq_hz": fr.omegas / (2.0 * jnp.pi),
        "magnitude_db": mag_db,
        "phase_deg": phase_deg,
    }


def _residual_fn(system, base_context, input_port):
    """Build a function ``r(x) = ẋ(x, u₀)`` from the system + frozen ``u₀``."""
    u0 = input_port.eval(base_context)

    def residual(x):
        ctx = base_context.with_continuous_state(x)
        with input_port.fixed(u0):
            xdot = system.eval_time_derivatives(ctx)
        return xdot

    return residual, u0


@dataclass
class OperatingPoint:
    """Result of :func:`findop`.

    Attributes:
        x: Equilibrium continuous state.
        u: Input value held fixed during the search (taken from
            ``base_context`` at call time).
        residual_norm: Final ``‖ẋ(x*, u)‖_∞`` after Newton iterations.
        converged: True if ``residual_norm`` met ``tol`` within
            ``max_iter`` steps.
        iterations: Number of Newton iterations actually executed.
    """

    x: Any
    u: Any
    residual_norm: float
    converged: bool
    iterations: int


def findop(
    system,
    base_context,
    *,
    initial_guess=None,
    input_port=None,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1e-10,
    axis_mask=None,
    residual_fn=None,
    residual_scaling=None,
    scaling_eps: float = 1e-8,
) -> OperatingPoint:
    """Find a continuous-state operating point ``x*`` such that ``ẋ(x*, u₀) ≈ 0``.

    Performs damped Newton iteration on the residual ``r(x) = ẋ(x, u₀)`` where
    ``u₀`` is read from ``base_context`` (and held fixed for the duration of
    the search).  The Jacobian is computed with ``jax.jacrev`` and the linear
    update is solved with ``jnp.linalg.solve`` plus a small Levenberg
    regularisation so singular Jacobians degrade to a least-squares step
    instead of NaN.

    Args:
        system: The system whose equilibrium is sought.
        base_context: A context that supplies the initial state, parameter
            values, and (via ``input_port.eval``) the held-fixed input.
        initial_guess: Optional initial state.  Defaults to
            ``base_context.continuous_state``.
        input_port: Input port to read ``u₀`` from.  Defaults to
            ``system.input_ports[0]`` (errors if ``system`` has multiple
            inputs and none is specified).
        tol: Stop when ``max(|scaled residual|) < tol`` over the solved-for
            components (equals ``max(|ẋ|)`` when ``residual_scaling`` is off).
        max_iter: Hard cap on Newton iterations.
        damping: Tikhonov damping added to ``JᵀJ`` for ill-conditioned solves.
        axis_mask: Optional selector for which state components the Newton
            iteration drives to zero.  Either a boolean array (length =
            number of flat state components, ``True`` = solve this component)
            or a sequence of integer indices.  Components *not* selected are
            held at ``initial_guess`` and excluded from both the residual and
            the unknowns — use this to trim systems with *passive* states whose
            equilibrium derivative is intrinsically nonzero (a cornering
            vehicle's heading ``ψ̇ = r ≠ 0``, a free integrator), which would
            otherwise dominate the full-state Newton step and prevent
            convergence.  ``None`` (default) solves the full state.
        residual_fn: Optional ``(x) -> residual_vector`` overriding the default
            ``ẋ(x, u₀)`` — e.g. to add a custom equilibrium condition or drop
            terms.  Receives the full state ``x``; its output is masked /
            scaled like the default residual.
        residual_scaling: Optional per-component residual weighting to put
            disparate units on a common footing (cf. MATLAB ``findop``'s
            ``XScaling`` / ``YScaling``) — e.g. a chassis residual in ``m/s²``
            ~10 alongside a wheel residual in ``rad/s²`` ~100.  One of:
            ``None`` (no scaling), ``"auto"`` (component ``i`` scaled by
            ``1/max(|rᵢ(x₀)|, scaling_eps)`` so each starts at order 1), or an
            explicit array (full-state length, or solved-subset length when
            ``axis_mask`` is given).  Applied to the Newton step *and* the
            convergence test; ``residual_norm`` is then reported in scaled
            units.
        scaling_eps: Floor for the ``"auto"`` scaling denominator.

    Returns:
        :class:`OperatingPoint` carrying the equilibrium state and convergence
        metadata.  ``x`` always has the shape of ``initial_guess`` (held
        components carry their initial values when ``axis_mask`` is used).

    Notes:
        The returned ``x`` is a JAX array, so the residual function used here
        is differentiable: ``jax.grad(lambda x0: jnp.sum(residual(x0)**2))``
        works.  Composing :func:`findop` itself under ``jax.grad`` requires an
        implicit-differentiation wrapper which is deferred to a follow-up.

        **Robust fallback.**  Newton operating-point search can stall on stiff,
        strongly-coupled, or badly-scaled systems even with ``axis_mask`` and
        ``residual_scaling``.  The most robust equilibrium finder is simply to
        *integrate to steady state*: ``simulate`` the system from a reasonable
        initial condition over a horizon long relative to its slowest mode and
        take the final state (optionally asserting ``max(|ẋ|)`` is small
        there).  Use that when ``findop`` reports ``converged=False``.
    """
    if input_port is None:
        if len(system.input_ports) != 1:
            raise ValueError(
                "findop(): system has multiple input ports — pass "
                "input_port=... explicitly."
            )
        input_port = system.input_ports[0]

    default_residual, u0 = _residual_fn(system, base_context, input_port)
    residual = residual_fn if residual_fn is not None else default_residual

    if initial_guess is None:
        initial_guess = base_context.continuous_state
    x = jnp.asarray(initial_guess)
    x_shape = x.shape
    x0_flat = x.ravel()
    n_total = x0_flat.shape[0]

    # Resolve axis_mask -> integer indices of the components the Newton
    # iteration solves for.  ``None`` -> all components (legacy full solve).
    if axis_mask is None:
        free_idx = np.arange(n_total)
    else:
        m = np.asarray(axis_mask)
        if m.dtype == bool:
            if m.ravel().shape[0] != n_total:
                raise ValueError(
                    f"findop(): boolean axis_mask has {m.size} entries but the "
                    f"state has {n_total} components."
                )
            free_idx = np.where(m.ravel())[0]
        else:
            free_idx = m.ravel().astype(int)
        if free_idx.size == 0:
            raise ValueError("findop(): axis_mask selects no state components.")

    free_idx_j = jnp.asarray(free_idx)

    def _full_from_z(z):
        return x0_flat.at[free_idx_j].set(z).reshape(x_shape)

    def _masked_residual(z):
        r = jnp.atleast_1d(jnp.ravel(residual(_full_from_z(z))))
        return r[free_idx_j]

    z = x0_flat[free_idx_j]

    # Per-component residual scaling (applied to the solved-for subset).
    r0 = _masked_residual(z)
    if residual_scaling is None:
        scale = jnp.ones_like(r0)
    elif isinstance(residual_scaling, str) and residual_scaling == "auto":
        scale = 1.0 / jnp.maximum(jnp.abs(r0), scaling_eps)
    else:
        scale = jnp.asarray(residual_scaling).ravel()
        if scale.shape[0] == n_total and free_idx.size != n_total:
            scale = scale[free_idx_j]
        if scale.shape[0] != r0.shape[0]:
            raise ValueError(
                f"findop(): residual_scaling has {scale.shape[0]} entries but "
                f"the solved residual has {r0.shape[0]} components."
            )

    def _scaled_residual(z):
        return _masked_residual(z) * scale

    # Restore the previous fixed value (if any) at the end so we don't
    # accidentally mutate caller state.
    restore_fixed_val = bool(getattr(input_port, "is_fixed", False))

    jac_fn = jax.jit(jax.jacrev(_scaled_residual))
    res_fn = jax.jit(_scaled_residual)

    converged = False
    iterations = 0
    res_val = res_fn(z)
    res_norm = float(jnp.max(jnp.abs(res_val)))

    for k in range(max_iter):
        iterations = k
        if res_norm < tol:
            converged = True
            break
        J = jnp.atleast_2d(jac_fn(z))
        r = jnp.atleast_1d(res_val)
        # Damped normal-equation step: (JᵀJ + λI) Δ = Jᵀ r
        n = J.shape[1]
        JtJ = J.T @ J + damping * jnp.eye(n, dtype=J.dtype)
        rhs = J.T @ r
        delta = npa.linalg.solve(JtJ, rhs)
        z = z - delta.reshape(z.shape)
        res_val = res_fn(z)
        res_norm = float(jnp.max(jnp.abs(res_val)))
    else:
        # Loop exhausted without break: record the post-loop iteration count.
        iterations = max_iter
        if res_norm < tol:
            converged = True

    x = _full_from_z(z)

    if restore_fixed_val:
        input_port.fix_value(u0)

    return OperatingPoint(
        x=x,
        u=u0,
        residual_norm=res_norm,
        converged=converged,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# T-109 followup-fre — Empirical Frequency Response Estimation
# ---------------------------------------------------------------------------


def _resample_to_uniform(t, x, n_samples=None):
    """Resample a possibly non-uniformly-sampled signal onto a uniform grid.

    Args:
        t: 1-D array of (possibly non-uniform) sample times.
        x: 1-D array of samples co-indexed with ``t``.
        n_samples: Number of points on the uniform grid.  Defaults to
            ``len(t)``.

    Returns:
        Tuple ``(t_uniform, x_uniform, dt)``.
    """
    t = np.asarray(t).ravel()
    x = np.asarray(x).ravel()
    if n_samples is None:
        n_samples = len(t)
    t_uniform = np.linspace(t[0], t[-1], n_samples)
    x_uniform = np.interp(t_uniform, t, x)
    dt = (t[-1] - t[0]) / (n_samples - 1)
    return t_uniform, x_uniform, dt


def _hann_window(n):
    """Hann window (numpy implementation, no scipy dependency)."""
    if n <= 1:
        return np.ones(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def estimate_frequency_response(
    diagram,
    ctx,
    t_span,
    input_port,
    output_port,
    freq_grid,
    *,
    options=None,
    recorded_signals_extra=None,
    window: bool = True,
    coherence_floor: float = 1e-12,
    n_segments: int = 8,
    segment_overlap: float = 0.5,
) -> FrequencyResponse:
    """Empirically estimate the SISO transfer function of ``diagram``.

    Drives ``diagram`` with whatever signal is already wired to ``input_port``
    (typically a :class:`jaxonomy.library.Chirp`, :class:`PRBS`, or
    :class:`BandLimitedNoise` source connected upstream of ``input_port``)
    and records the input/output trajectories.  The empirical transfer
    function is computed as the cross-spectral ratio
    ``G(f) = Sxy(f) / Sxx(f)`` where ``Sxx`` and ``Sxy`` are the (Hann-
    windowed) auto- and cross-spectral densities of input/output.  Results
    are interpolated onto the user-supplied ``freq_grid`` (Hz).

    This is the practical alternative to analytic :func:`linearize` /
    :func:`frequency_response` when:

    * the system contains hard nonlinearities (lookup tables, saturation,
      contact dynamics) that make symbolic linearization fragile, or
    * you want an empirical sanity-check against the linearized model.

    Args:
        diagram: A built diagram (typically with a chirp or PRBS source
            wired to the block-under-test's input port).
        ctx: Initial simulation context.
        t_span: ``(t0, tf)`` simulation horizon.  Make this comfortably
            longer than the slowest period of interest in ``freq_grid``.
        input_port: ``OutputPort`` whose recorded trajectory provides the
            excitation samples ``u(t)``.  This is typically the upstream
            source's output port (the same signal that drives the
            block-under-test).
        output_port: ``OutputPort`` whose recorded trajectory provides the
            measured response ``y(t)``.
        freq_grid: 1-D array of frequencies (Hz) at which the empirical
            response should be evaluated.  Frequencies outside the
            simulation's resolved band ``[1/T, fs/2]`` are clamped — the
            caller should keep ``freq_grid`` inside that band.
        options: Optional :class:`SimulatorOptions`.  ``recorded_signals``
            is overridden internally; everything else (rtol, atol, solver,
            etc.) is honoured.
        recorded_signals_extra: Optional ``dict[str, OutputPort]`` of
            additional signals to record alongside the input/output
            (useful for debugging / plotting).  Not used by the estimator
            itself.
        window: If True (default) apply a Hann window before the FFT to
            suppress spectral leakage.  If False (rectangular) the
            transfer-function ratio is more sensitive to leakage but
            faithful to the raw FFT.
        coherence_floor: Minimum ``|U(f)|²`` in the auto-spectrum below
            which the ratio is set to zero — guards against division by
            zero at frequencies the excitation never visited.
        n_segments: Number of overlapping segments to average (Welch's
            method).  More segments → less variance, lower frequency
            resolution.  ``n_segments=1`` falls back to a single-window
            FFT estimate.  Default 8 is a reasonable trade-off for most
            chirp/PRBS excitations.
        segment_overlap: Fractional overlap between consecutive segments
            (Welch's method), in ``[0, 1)``.  Default 0.5 (50%).

    Returns:
        :class:`FrequencyResponse` with ``omegas = 2π·freq_grid``, complex
        ``response`` of shape ``(K, 1, 1)``, and corresponding
        ``magnitudes`` and ``phases``.  Drop-in compatible with
        :func:`bode_data`.

    Notes:
        * The implementation is intentionally pure-NumPy on the
          post-simulation arrays; it does not need to be JAX-traceable
          (callers can JIT downstream code that consumes the returned
          ``response`` array).
        * For best results pick an excitation that covers the band of
          interest densely: a linear :class:`Chirp` from ``f0 ≪ freq_min``
          to ``f1 ≳ freq_max`` over a horizon of several seconds, or a
          :class:`PRBS` with sample time ``≪ 1/(2·freq_max)``.
        * The returned ``response`` is a NumPy complex array (consumers
          calling :func:`bode_data` will see ``jnp.asarray`` promotion);
          this is fine because :class:`FrequencyResponse` fields are typed
          ``Any``.
    """
    # Lazy import to avoid a top-level circular dependency between
    # ``jaxonomy.library`` and ``jaxonomy.simulation``.
    from jaxonomy.simulation import simulate
    from jaxonomy.simulation.types import SimulatorOptions

    recorded = {"_fre_in": input_port, "_fre_out": output_port}
    if recorded_signals_extra:
        # Rename collisions are caller-error; just merge.
        recorded.update(recorded_signals_extra)

    # Auto-bump ``buffer_length`` when the user requests a fine
    # ``max_major_step_length`` but supplies a fixed buffer too small
    # for the recording — otherwise the recorded time series gets
    # truncated to the buffer tail and the FFT sees almost no data.
    # ``buffer_length=None`` (post-T-002b auto-size default) is honoured
    # by ``_check_options`` downstream, so no bump is needed in that case.
    if options is not None and options.max_major_step_length is not None:
        t0, tf = float(t_span[0]), float(t_span[1])
        needed = int(np.ceil((tf - t0) / float(options.max_major_step_length))) + 8
        current = options.buffer_length
        if current is not None and current < needed:
            # Replace the dataclass with an updated copy so we don't
            # mutate the caller's options object.
            import dataclasses as _dc
            options = _dc.replace(options, buffer_length=needed)

    results = simulate(
        diagram,
        ctx,
        t_span=t_span,
        options=options,
        recorded_signals=recorded,
    )

    t = np.asarray(results.time)
    u_raw = np.asarray(results.outputs["_fre_in"])
    y_raw = np.asarray(results.outputs["_fre_out"])

    # Coerce vector-valued single-channel outputs to scalar-per-sample.
    if u_raw.ndim > 1:
        u_raw = u_raw.reshape(u_raw.shape[0], -1)
        if u_raw.shape[1] != 1:
            raise ValueError(
                "estimate_frequency_response(): input_port must be SISO "
                f"(got width {u_raw.shape[1]}).  Use Demux to select a "
                "single channel."
            )
        u_raw = u_raw[:, 0]
    if y_raw.ndim > 1:
        y_raw = y_raw.reshape(y_raw.shape[0], -1)
        if y_raw.shape[1] != 1:
            raise ValueError(
                "estimate_frequency_response(): output_port must be SISO "
                f"(got width {y_raw.shape[1]}).  Use Demux to select a "
                "single channel."
            )
        y_raw = y_raw[:, 0]

    if t.size < 4:
        raise ValueError(
            "estimate_frequency_response(): need at least 4 samples; got "
            f"{t.size}.  Lengthen ``t_span`` or relax ``max_major_step_length``."
        )

    # Resample to a uniform grid so np.fft.rfft is correct.
    n = int(t.size)
    t_u, u, dt = _resample_to_uniform(t, u_raw, n_samples=n)
    _, y, _ = _resample_to_uniform(t, y_raw, n_samples=n)

    # Linear-detrend (a DC offset or a slow drift — e.g. from an integrator
    # reacting to the small DC component of a chirp — would dominate the
    # low-frequency bins and bleed into higher bins via spectral leakage).
    def _linear_detrend(arr):
        idx = np.arange(arr.size, dtype=np.float64)
        # Least-squares fit y = a*idx + b in closed form.
        n_arr = arr.size
        sum_x = idx.sum()
        sum_y = arr.sum()
        sum_xx = (idx * idx).sum()
        sum_xy = (idx * arr).sum()
        denom = n_arr * sum_xx - sum_x * sum_x
        if denom == 0.0:
            return arr - arr.mean()
        slope = (n_arr * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n_arr
        return arr - (slope * idx + intercept)

    u = _linear_detrend(u)
    y = _linear_detrend(y)

    # ---- Welch-style segmented average of cross-/auto-spectra. ----
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if not (0.0 <= segment_overlap < 1.0):
        raise ValueError("segment_overlap must be in [0, 1)")

    if n_segments == 1:
        seg_len = n
        starts = [0]
    else:
        # Choose a segment length so ``n_segments`` overlapping windows
        # fit inside ``n``.  ``hop = seg_len * (1 - overlap)`` →
        # ``n >= seg_len + (n_segments - 1) * hop``  →
        # ``seg_len <= n / (1 + (n_segments - 1)*(1 - overlap))``.
        denom = 1.0 + (n_segments - 1) * (1.0 - segment_overlap)
        seg_len = max(int(n / denom), 8)
        if seg_len >= n:
            seg_len = n
            starts = [0]
        else:
            hop = max(int(seg_len * (1.0 - segment_overlap)), 1)
            starts = [k * hop for k in range(n_segments) if k * hop + seg_len <= n]
            if not starts:
                starts = [0]
                seg_len = n

    if window:
        w = _hann_window(seg_len)
    else:
        w = np.ones(seg_len)

    n_freqs = seg_len // 2 + 1
    Sxx_acc = np.zeros(n_freqs, dtype=np.float64)
    Sxy_acc = np.zeros(n_freqs, dtype=np.complex128)

    for s in starts:
        u_seg = u[s : s + seg_len] * w
        y_seg = y[s : s + seg_len] * w
        U = np.fft.rfft(u_seg)
        Y = np.fft.rfft(y_seg)
        Sxx_acc += (U.conj() * U).real
        Sxy_acc += U.conj() * Y

    Sxx_acc /= len(starts)
    Sxy_acc /= len(starts)
    freqs = np.fft.rfftfreq(seg_len, d=dt)

    # ---- Smoothed transfer-function estimate. ----
    # The bin-by-bin ratio ``Sxy/Sxx`` is high-variance for excitations
    # like a chirp (which dwell only briefly at any instantaneous freq).
    # The standard fix is to compute the ratio of *smoothed* spectra:
    # smooth Sxx and Sxy with a small box (or Hann) kernel along
    # frequency, then take the ratio.  This is the H1 estimator from
    # the system-ID literature.
    freq_grid = np.asarray(freq_grid, dtype=np.float64).ravel()

    # Choose smoothing kernel half-width.  We want the smoothing window
    # to be much narrower than the spacing between user-requested target
    # frequencies (so we don't mush them together).  A safe heuristic:
    # 1/4 of the smallest target-frequency gap, or 3 bins minimum.
    df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    if freq_grid.size >= 2:
        min_gap = float(np.min(np.diff(np.sort(freq_grid))))
        kernel_half = max(3, int(min_gap / (4.0 * df)))
    else:
        kernel_half = 3
    # Cap at 1% of the spectrum to avoid degenerate cases.
    kernel_half = min(kernel_half, max(3, Sxx_acc.size // 100))
    kernel_len = 2 * kernel_half + 1
    kernel = np.ones(kernel_len) / kernel_len  # box smoother

    # Same convolution length via 'same' mode.
    Sxx_smooth = np.convolve(Sxx_acc, kernel, mode="same")
    # Smooth real and imaginary of Sxy independently to keep it complex.
    Sxy_smooth = (
        np.convolve(Sxy_acc.real, kernel, mode="same")
        + 1j * np.convolve(Sxy_acc.imag, kernel, mode="same")
    )

    safe = Sxx_smooth > coherence_floor
    G_emp = np.zeros_like(Sxy_smooth, dtype=np.complex128)
    G_emp[safe] = Sxy_smooth[safe] / Sxx_smooth[safe]

    if not np.any(safe):
        G_grid = np.zeros(freq_grid.size, dtype=np.complex128)
    else:
        # Linear interp of magnitude and unwrapped phase — std Bode plot.
        mag = np.abs(G_emp)
        phase = np.unwrap(np.angle(G_emp))
        mag_grid = np.interp(freq_grid, freqs, mag)
        phase_grid = np.interp(freq_grid, freqs, phase)
        G_grid = mag_grid * np.exp(1j * phase_grid)

    # Reshape to ``(K, 1, 1)`` to match :func:`frequency_response`.
    K = freq_grid.size
    response = G_grid.reshape((K, 1, 1))
    magnitudes = np.abs(response)
    phases = np.angle(response)
    omegas = 2.0 * np.pi * freq_grid

    # Promote via npa so consumers that JIT downstream see jax arrays.
    return FrequencyResponse(
        omegas=npa.asarray(omegas),
        response=npa.asarray(response),
        magnitudes=npa.asarray(magnitudes),
        phases=npa.asarray(phases),
    )


# ---------------------------------------------------------------------------
# T-109 followup-pzmap-step-impulse — pole-zero map + step/impulse responses
# ---------------------------------------------------------------------------


def _coerce_state_space(linsys):
    """Promote ``(A, B, C, D)`` to 2-D arrays and return ``(A, B, C, D, n, m, p)``.

    Shared helper for :func:`pole_zero_map`, :func:`step_response`, and
    :func:`impulse_response`.  Accepts the scalar/1-D shorthand that
    :class:`LinearizedSystem` users sometimes hand-construct (e.g.
    ``A=-1.0, B=1.0, C=1.0, D=0.0``).
    """
    A = jnp.asarray(linsys.A)
    if A.ndim == 0:
        A = A.reshape((1, 1))
    elif A.ndim == 1:
        n = A.size
        A = A.reshape((n, n))
    n = A.shape[0]

    B_raw = jnp.asarray(linsys.B)
    if B_raw.ndim == 0:
        B = B_raw.reshape((1, 1))
    elif B_raw.ndim == 1:
        # Column vector by default (single input).
        B = B_raw.reshape((n, 1)) if B_raw.size == n else B_raw.reshape((-1, 1))
    else:
        B = B_raw
    m = B.shape[1]

    C_raw = jnp.asarray(linsys.C)
    if C_raw.ndim == 0:
        C = C_raw.reshape((1, 1))
    elif C_raw.ndim == 1:
        # Row vector by default (single output).
        C = C_raw.reshape((1, n)) if C_raw.size == n else C_raw.reshape((1, -1))
    else:
        C = C_raw
    p = C.shape[0]

    D = _ensure_2d(linsys.D, p, m)

    return A, B, C, D, n, m, p


def pole_zero_map(linsys: LinearizedSystem) -> dict:
    """Compute poles and zeros of a :class:`LinearizedSystem`.

    Poles are the eigenvalues of ``A``.  Zeros are the (transmission)
    zeros of the SISO transfer function ``G(s) = C (sI − A)⁻¹ B + D``,
    computed as the finite generalised eigenvalues of the Rosenbrock
    system pencil

    .. code-block:: text

        P(s) = [[ sI − A, −B ],
                [   C   ,  D ]]

    by solving the generalised eigenproblem ``λ E v = M v`` with

    .. code-block:: text

        E = [[ I, 0 ],   M = [[ A, B ],
             [ 0, 0 ]]        [ C, D ]]

    Finite eigenvalues (those with non-zero ``E``-weight) of this pencil
    are the invariant zeros of the system; for SISO they coincide with
    the numerator roots of the transfer function.  The high-frequency
    gain is reported as ``D[0, 0]`` (the asymptotic value of
    ``G(s) → D`` for ``|s| → ∞``); for a strictly-proper system
    (``D = 0``) the leading-coefficient gain is harder to define
    unambiguously without polynomial fitting and is left to a deeper
    follow-up.

    Args:
        linsys: A :class:`LinearizedSystem`.  Phase 1 ships SISO support
            only — for MIMO systems the first input/output channel is
            used.

    Returns:
        Dictionary with keys:
            ``"poles"`` — complex 1-D array of eigenvalues of ``A``,
            ``"zeros"`` — complex 1-D array of (finite) invariant zeros,
            ``"gain"`` — feedthrough scalar ``D[0, 0]``.

    Notes:
        Pole computation is differentiable through ``A`` (via
        ``jnp.linalg.eigvals``).  Zero computation uses
        :func:`scipy.linalg.eig` on the generalised problem and is
        therefore not currently traceable by JAX — call it eagerly,
        outside ``jit``.  A differentiable variant is a deeper follow-up.
    """
    A, B, C, D, n, m, p = _coerce_state_space(linsys)

    # Poles: eigenvalues of A.
    poles = jnp.linalg.eigvals(A)

    # Zeros: finite generalised eigenvalues of the Rosenbrock pencil.
    # SISO assumption — restrict to first input/output for phase 1.
    B0 = B[:, :1]  # (n, 1)
    C0 = C[:1, :]  # (1, n)
    D0 = D[:1, :1]  # (1, 1)

    # Block-assemble M and E (size n+1 x n+1).
    M_top = jnp.concatenate([A, B0], axis=1)            # (n, n+1)
    M_bot = jnp.concatenate([C0, D0], axis=1)           # (1, n+1)
    M = jnp.concatenate([M_top, M_bot], axis=0)         # (n+1, n+1)

    E_top = jnp.concatenate(
        [jnp.eye(n, dtype=A.dtype), jnp.zeros((n, 1), dtype=A.dtype)],
        axis=1,
    )
    E_bot = jnp.zeros((1, n + 1), dtype=A.dtype)
    E = jnp.concatenate([E_top, E_bot], axis=0)

    # Generalised eigenproblem — defer to scipy.linalg.eig because
    # jax.numpy does not yet expose a generalised eigensolver.  Pure
    # NumPy fallback would also work; scipy gives us a stable QZ.
    try:
        from scipy.linalg import eig as _scipy_eig
        eigvals, _ = _scipy_eig(np.asarray(M), np.asarray(E))
        eigvals = np.asarray(eigvals)
        # Filter infinities — those correspond to dynamic modes the
        # pencil decoupled into the singular part of E.
        finite = np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)
        zeros = jnp.asarray(eigvals[finite])
    except Exception:
        # If scipy unavailable, fall back to the strictly-proper
        # invertible-A formula: zeros are eigenvalues of (A − B*C/D) when
        # D ≠ 0.  Otherwise return an empty zeros vector.
        d_scalar = float(np.asarray(D0).reshape(-1)[0])
        if abs(d_scalar) > 1e-300:
            zeros = jnp.linalg.eigvals(A - (B0 @ C0) / d_scalar)
        else:
            zeros = jnp.zeros((0,), dtype=jnp.complex128)

    gain = jnp.asarray(D0).reshape(())

    return {
        "poles": poles,
        "zeros": zeros,
        "gain": float(np.asarray(gain)),
    }


def _augmented_step_block(A, B, t):
    """Return ``∫₀ᵗ expm(A·s) ds · B`` via the augmented-matrix trick.

    For an LTI system the step response involves the matrix integral
    ``∫₀ᵗ expm(A·s) ds`` which equals ``A⁻¹(expm(A·t) − I)`` when ``A``
    is invertible.  We avoid the inverse (which fails for integrators
    ``A = 0``) by computing ``expm`` of the block matrix

    .. code-block:: text

        M = [[ A·t, B·t ],
             [  0 ,  0  ]]

    and reading the top-right ``(n, m)`` block of ``expm(M)``; this
    block equals exactly ``∫₀ᵗ expm(A·s) ds · B``.  See Van Loan (1978)
    or Moler-Van Loan (2003), "Nineteen Dubious Ways..." for the proof.
    """
    from jax.scipy.linalg import expm

    n = A.shape[0]
    m = B.shape[1]
    # Top row: [A·t, B·t]; bottom row: zeros of shape (m, n+m).
    top = jnp.concatenate([A * t, B * t], axis=1)              # (n, n+m)
    bottom = jnp.zeros((m, n + m), dtype=A.dtype)              # (m, n+m)
    M = jnp.concatenate([top, bottom], axis=0)                 # (n+m, n+m)
    EM = expm(M)
    return EM[:n, n:]  # (n, m) block = ∫₀ᵗ expm(A·s) ds · B


def step_response(linsys: LinearizedSystem, t_grid):
    """Closed-form step response of a continuous-time LTI system.

    For zero initial state and unit step input ``u(t) = 1`` (for ``t ≥ 0``)
    the response is

    .. code-block:: text

        y(t) = C · ∫₀ᵗ expm(A·s) ds · B  +  D·1

    The matrix integral is computed via the augmented-matrix expm trick
    (see :func:`_augmented_step_block`) so the routine works correctly
    for non-invertible ``A`` (e.g. integrators).  When ``A`` is
    invertible the same value equals ``C·A⁻¹·(expm(A·t) − I)·B + D``.

    Args:
        linsys: A :class:`LinearizedSystem`.
        t_grid: Scalar or 1-D array of evaluation times.  Negative times
            are evaluated formally (the closed-form result is still well
            defined; physically the step starts at ``t = 0``).

    Returns:
        Array of shape ``(K, p, m)`` — step response at each ``t`` for
        each ``(output, input)`` pair, where ``K = len(t_grid)``,
        ``p = n_outputs``, ``m = n_inputs``.  If ``t_grid`` is a scalar
        the returned shape is ``(p, m)``.  For SISO systems with
        scalar ``t_grid`` the result squeezes naturally to a scalar.

    Notes:
        Fully differentiable through ``A, B, C, D`` via
        :func:`jax.scipy.linalg.expm`.  For very large state dimensions
        (``n > 50``) the augmented expm may be slow — the honest fall-
        back is to simulate the diagram with a :class:`Step` source
        (deferred to a deeper follow-up).
    """
    if linsys.is_discrete():
        raise ValueError(
            "step_response is defined for continuous-time LinearizedSystems "
            f"only, but got a discrete-time system (dt={linsys.dt}). The "
            "closed-form continuous matrix-exponential formula does not "
            "apply to a discrete recurrence x[k+1] = A x[k] + B u[k]. "
            "Simulate the discrete diagram directly to obtain its step "
            "response (see KNOWN_GAPS.md)."
        )

    A, B, C, D, n, m, p = _coerce_state_space(linsys)

    t_arr = jnp.asarray(t_grid)
    scalar_input = (t_arr.ndim == 0)
    if scalar_input:
        t_arr = t_arr.reshape((1,))

    def one(t):
        block = _augmented_step_block(A, B, t)  # (n, m)
        return C @ block + D                    # (p, m)

    out = jax.vmap(one)(t_arr)  # (K, p, m)
    if scalar_input:
        return out[0]
    return out


def impulse_response(linsys: LinearizedSystem, t_grid):
    """Closed-form impulse response of a continuous-time LTI system.

    For zero initial state the (finite part of the) impulse response is

    .. code-block:: text

        y(t) = C · expm(A·t) · B            for t > 0

    The Dirac component ``D · δ(t)`` is omitted from the returned
    samples since it is not representable on a numeric grid; consumers
    that need it can add ``D`` to the ``t = 0`` sample explicitly.

    Args:
        linsys: A :class:`LinearizedSystem`.
        t_grid: Scalar or 1-D array of evaluation times.

    Returns:
        Array of shape ``(K, p, m)`` for vector ``t_grid``, or ``(p, m)``
        for scalar ``t_grid``.  ``K = len(t_grid)``, ``p = n_outputs``,
        ``m = n_inputs``.

    Notes:
        Fully differentiable through ``A, B, C, D``.
    """
    from jax.scipy.linalg import expm

    if linsys.is_discrete():
        raise ValueError(
            "impulse_response is defined for continuous-time "
            "LinearizedSystems only, but got a discrete-time system "
            f"(dt={linsys.dt}). The closed-form continuous matrix-"
            "exponential formula does not apply to a discrete recurrence "
            "x[k+1] = A x[k] + B u[k]. Simulate the discrete diagram "
            "directly to obtain its impulse response (see KNOWN_GAPS.md)."
        )

    A, B, C, D, n, m, p = _coerce_state_space(linsys)

    t_arr = jnp.asarray(t_grid)
    scalar_input = (t_arr.ndim == 0)
    if scalar_input:
        t_arr = t_arr.reshape((1,))

    def one(t):
        return C @ expm(A * t) @ B           # (p, m)

    out = jax.vmap(one)(t_arr)               # (K, p, m)
    if scalar_input:
        return out[0]
    return out


# ---------------------------------------------------------------------------
# T-109 followup-nyquist — Nyquist contour data
# ---------------------------------------------------------------------------


def nyquist_data(linsys: LinearizedSystem, omegas) -> dict:
    """Return Nyquist-contour arrays for ``linsys``.

    The Nyquist plot traces ``G(jω)`` through the complex plane.  This
    helper returns the real and imaginary parts of ``G(jω)`` over the
    supplied positive angular frequencies and additionally the reflected
    negative-frequency arrays (since ``G(-jω) = conj(G(jω))`` for a
    real-coefficient LTI, the reflection is given exactly by
    ``(Re, -Im)``).  Consumers can concatenate the negative and positive
    arrays to obtain the full closed contour used for encirclement
    counting; for stability margins computed only from the positive
    sweep, ``real`` and ``imag`` are sufficient.

    For MIMO systems the returned ``real`` / ``imag`` arrays preserve
    the ``(K, p, m)`` channel structure from :func:`frequency_response`;
    for SISO they are squeezed to ``(K,)``, matching the convention of
    :func:`bode_data`.

    Args:
        linsys: A :class:`LinearizedSystem`.
        omegas: 1-D array-like of *positive* angular frequencies
            ``ω`` (rad/s).  Negative or zero entries are not rejected —
            ``G(0)`` is the DC gain and is returned unchanged, but
            duplicating with the mirror is not meaningful at ω = 0.

    Returns:
        Dictionary with keys:
            ``"omega"`` — positive angular frequencies (rad/s), shape
                ``(K,)``;
            ``"real"`` — ``Re G(jω)``; shape ``(K,)`` for SISO,
                ``(K, p, m)`` for MIMO;
            ``"imag"`` — ``Im G(jω)``; same shape as ``real``;
            ``"real_neg"`` — ``Re G(-jω) = Re G(jω)``; same shape as
                ``real`` (mirror; provided for convenience when plotting
                the full closed contour);
            ``"imag_neg"`` — ``Im G(-jω) = -Im G(jω)``; same shape as
                ``imag``.

    Notes:
        Fully differentiable through ``A, B, C, D`` and ``omegas`` via
        the underlying :func:`frequency_response`.
    """
    fr = frequency_response(linsys, omegas)
    response = fr.response
    # SISO squeeze, matching bode_data convention.
    is_siso = (
        response.ndim == 3 and response.shape[-1] == 1 and response.shape[-2] == 1
    )
    if is_siso:
        response = response[..., 0, 0]
    real = jnp.real(response)
    imag = jnp.imag(response)
    return {
        "omega": fr.omegas,
        "real": real,
        "imag": imag,
        "real_neg": real,
        "imag_neg": -imag,
    }


# ---------------------------------------------------------------------------
# T-109 phase 4 (LTI sub-piece): discretize a LinearizedSystem
# ---------------------------------------------------------------------------
#
# Phase 4 of T-109 calls for diagram-level higher-order operators —
# ``jaxonomy.discretize(diagram, dt, method)`` and
# ``jaxonomy.with_observer(diagram, gain)`` — promoted from the
# matrix-level helpers in ``jaxonomy.library.state_estimators.utils``.
# The full diagram-level lift (walking a Diagram tree and converting
# every continuous block to its discrete equivalent) is a large
# refactor; this ships the *LTI-level* sub-piece, which is what every
# downstream controller-design call actually consumes once linearize
# has run.
#
# Future work: a diagram-level ``discretize`` that walks the tree
# (filed as T-109-followup-diagram-level-discretize when the demand
# materialises).


def discretize(
    linsys,
    dt: float,
    *,
    method: str = "zoh",
    base_context=None,
    input_port=None,
    output_port=None,
) -> LinearizedSystem:
    """Discretize a continuous-time linear system (T-109 phase 4).

    Two call patterns, dispatched on the type of ``linsys``:

    1. ``discretize(linsys: LinearizedSystem, dt, *, method)``  — the
       LTI-level path (shipped first as the T-109 phase-4 sub-piece).
       Wraps the matrix-level helpers in
       :mod:`jaxonomy.library.state_estimators.utils`.
    2. ``discretize(system: SystemBase, dt, *, method, base_context, input_port, output_port)``
       — the **diagram-level lift** (T-109 phase 4 completion).
       Linearizes ``system`` about ``base_context`` (via :func:`linearize`)
       then routes the result through path 1. Equivalent to
       ``discretize(linearize(system, base_context, ...), dt, method=method)``;
       provided so controller-design workflows can write
       ``ddiagram = jaxonomy.discretize(diagram, dt, base_context=ctx)``
       in one call.

    Converts ``dx/dt = Ax + Bu`` into ``x[k+1] = A_d x[k] + B_d u[k]``
    while keeping ``C``, ``D``, and the operating point untouched (the
    output map is unaffected by discretization). The returned
    :class:`LinearizedSystem` carries ``dt`` so downstream consumers
    (e.g. :meth:`LinearizedSystem.is_stable`) interpret it as
    discrete-time.

    Args:
        linsys: Either a continuous-time :class:`LinearizedSystem`
            (``dt`` must be ``None``) or a :class:`SystemBase` /
            :class:`Diagram`. In the diagram case, ``base_context`` is
            required.
        dt: Sampling period in seconds. Must be positive.
        method: Discretization rule.
            ``"zoh"`` (default) — exact zero-order-hold:
            ``A_d = expm(A·dt)``,
            ``B_d = A⁻¹ (A_d − I) B`` (with a first-order Taylor
            fallback when ``A`` is near-singular, so integrator dynamics
            ``A = 0`` work cleanly).
            ``"euler"`` — first-order forward-Euler:
            ``A_d = I + A·dt``, ``B_d = B·dt``. Cheap and JAX-clean but
            biased; use ``"zoh"`` unless you specifically need the
            Euler shape for hardware-in-the-loop parity.
        base_context: Required when ``linsys`` is a SystemBase /
            Diagram; ignored when it's already a LinearizedSystem.
            The operating point about which to linearize.
        input_port: Optional input port for :func:`linearize` (diagram
            path only). Defaults to the diagram's single input.
        output_port: Optional output port for :func:`linearize`
            (diagram path only). Defaults to the diagram's single
            output.

    Returns:
        A new :class:`LinearizedSystem` with discrete matrices and
        ``dt`` set. The output map (``C``, ``D``) and
        ``operating_point`` are forwarded unchanged.

    Raises:
        ValueError: If ``dt <= 0``, ``method`` is not ``"zoh"`` or
            ``"euler"``, ``linsys`` is already discrete, or the
            diagram path is invoked without ``base_context``.

    Notes:
        Differentiable through ``A``, ``B``, ``C``, ``D``, and ``dt``
        via the JAX-traceable matrix exponential and linear solve.
        The diagram path is differentiable through whatever
        :func:`linearize` is itself differentiable through.

    See also:
        :func:`linearize` — the continuous-time linearization step.
        :func:`jaxonomy.library.state_estimators.utils.discretize_forward_zoh`
            and :func:`discretize_forward_euler` — the matrix-level
            primitives the LTI path wraps.
    """
    # Diagram-level dispatch: linearize first, then route through the
    # LTI path. Detect by the absence of LinearizedSystem-y attributes
    # rather than isinstance so we accept any structurally-equivalent
    # wrapper.
    if not isinstance(linsys, LinearizedSystem):
        if base_context is None:
            raise ValueError(
                "discretize: when the first argument is a SystemBase / "
                "Diagram, base_context= is required (it's the operating "
                "point to linearize about)."
            )
        # Lazy import to avoid the framework→library cycle at module load.
        from .linear_system import linearize

        kwargs = {}
        if input_port is not None:
            kwargs["input_port"] = input_port
        if output_port is not None:
            kwargs["output_port"] = output_port
        linsys = linearize(linsys, base_context, **kwargs)

    from .state_estimators.utils import (
        discretize_forward_euler,
        discretize_forward_zoh,
    )

    if linsys.dt is not None:
        raise ValueError(
            f"discretize: linsys already carries dt={linsys.dt!r}; "
            f"discretizing an already-discrete LinearizedSystem is not "
            f"well-defined without re-continuization first."
        )
    if method not in ("zoh", "euler"):
        raise ValueError(
            f"discretize: unknown method {method!r}; expected "
            f"'zoh' or 'euler'."
        )
    # Concrete-only positivity check — under jax.grad / jit ``dt`` is a
    # traced array and we cannot coerce it to a Python float. Skip the
    # eager validation in that case; XLA will surface any value-domain
    # issues at runtime via the underlying linear-algebra ops.
    try:
        dt_concrete = float(dt)
    except (TypeError, jax.errors.ConcretizationTypeError):
        dt_concrete = None
    if dt_concrete is not None and not (dt_concrete > 0):
        raise ValueError(f"discretize: dt must be positive; got {dt!r}.")

    A = jnp.asarray(linsys.A)
    if A.ndim == 0:
        A = A.reshape((1, 1))
    elif A.ndim == 1:
        n = A.size
        A = A.reshape((n, n))
    n = A.shape[0]

    B = _ensure_2d(linsys.B, n, max(jnp.asarray(linsys.B).size // n, 1))

    if method == "zoh":
        Ad, Bd = discretize_forward_zoh(A, B, dt)
    else:  # "euler"
        Ad, Bd = discretize_forward_euler(A, B, dt)

    # Stamp dt as a Python float when concrete (so the dataclass repr is
    # readable and is_discrete() is cheap); pass through the traced array
    # unchanged when called under jit / grad.
    dt_field = dt_concrete if dt_concrete is not None else dt
    return LinearizedSystem(
        A=Ad,
        B=Bd,
        C=linsys.C,
        D=linsys.D,
        operating_point=linsys.operating_point,
        dt=dt_field,
    )


# ---------------------------------------------------------------------------
# T-109-followup-with-observer: attach a Luenberger observer to a plant
# ---------------------------------------------------------------------------


def with_observer(
    plant,
    observer,
    *,
    plant_u_port: int = 0,
    plant_y_port: int = 0,
    name: str = "plant_with_observer",
):
    """Build a new diagram with ``observer`` attached to ``plant``.

    Wires the plant's control input through to both the plant and the
    observer; wires the plant's measurement output through to the
    observer; exports the observer's ``x_hat`` estimate as a top-level
    output of the augmented diagram. The plant's other output ports
    are not re-exported automatically — call sites that need them can
    wire them up in a parent diagram.

    Args:
        plant: A built :class:`Diagram` (or :class:`SystemBase`)
            representing the open-loop plant. Must expose at least one
            input port (for control ``u``) and one output port (for
            measurement ``y``).
        observer: A built observer block (typically
            :class:`jaxonomy.library.Luenberger` or any block that
            accepts ``(u, y)`` inputs and produces ``x_hat`` as its
            first output port).
        plant_u_port: Index of the plant input port carrying the
            control signal ``u``. Defaults to 0.
        plant_y_port: Index of the plant output port carrying the
            measurement ``y``. Defaults to 0.
        name: Name for the resulting wrapper diagram.

    Returns:
        A new :class:`Diagram` containing both ``plant`` and
        ``observer`` wired as described, with:

        * a single exported input port ``u`` (the control signal,
          fed to both the plant and the observer's ``u`` input),
        * a single exported output port ``x_hat`` (the observer's
          state estimate).

    Notes:
        The result is a "passive" instrumentation pattern — the
        observer reads ``(u, y)`` and emits an estimate, but the
        estimate is not fed back into the plant. To close a loop
        around the estimate (state-feedback control with observed
        state), build a controller subdiagram and wire its output
        back to ``plant.u`` in a parent diagram.
    """
    # Lazy import to avoid the framework→library cycle.
    from ..framework import DiagramBuilder
    from .routing import IOPort

    builder = DiagramBuilder()
    p = builder.add(plant)
    o = builder.add(observer)

    # The control signal needs to fan out to both the plant's u-input
    # and the observer's u-input. ``DiagramBuilder.export_input`` only
    # maps an exported port to one destination, so we use an IOPort
    # passthrough as the fan-out node: external "u" → IOPort → both.
    u_router = builder.add(IOPort(name="u_router"))
    builder.connect(u_router.output_ports[0], p.input_ports[plant_u_port])
    builder.connect(u_router.output_ports[0], o.input_ports[0])
    builder.connect(p.output_ports[plant_y_port], o.input_ports[1])

    builder.export_input(u_router.input_ports[0], name="u")
    builder.export_output(o.output_ports[0], name="x_hat")
    return builder.build(name=name)
