# SPDX-License-Identifier: MIT

"""Parameter distributions for UQ workflows.

Each distribution exposes three methods used by :mod:`jaxonomy.uq`:

* ``sample(key, shape) -> jnp.ndarray`` — IID samples in the original parameter
  space.
* ``ppf(u) -> jnp.ndarray`` — inverse-CDF (a.k.a. percent-point function),
  mapping uniform samples on ``[0, 1]`` to the parameter space.  Used by
  Saltelli (Sobol) and Morris sampling, which generate quasi-random ``u``
  values in the unit cube and map them through each parameter's PPF.
* ``log_pdf(x) -> jnp.ndarray`` — natural-log probability density at ``x``.
  Returns ``-inf`` outside the support. Consumed by importance sampling
  (T-126 followup) which needs ``log p_target / p_proposal`` weights.
* ``cdf(x) -> jnp.ndarray`` — forward cumulative distribution function
  ``P(X <= x)``.  Differentiable through ``x`` for continuous
  distributions (T-122-followup-distributions-cdf).
  ``MultivariateNormal`` and ``CorrelatedMarginals`` raise
  :class:`NotImplementedError` — joint multivariate CDFs need Genz's
  quasi-MC machinery.

Currently shipped:

* :class:`Uniform` — ``low + (high - low) * u``
* :class:`Normal` — ``loc + scale * Phi^-1(u)``
* :class:`LogNormal` — ``exp(loc + scale * Phi^-1(u))``
* :class:`Triangular` — closed-form inverse CDF over ``(low, mode, high)``
* :class:`Exponential` — ``-log(1 - u) / rate`` (T-122-followup-poisson)
* :class:`Poisson` — discrete counts via ``jax.random.poisson`` with
  ``log_pmf`` (no ``ppf`` — see class docstring).
* :class:`Categorical` — discrete choice over an explicit ``values`` list
  with matching ``probs`` (T-122-followup-categorical).  ``log_pmf`` for
  log-probability lookups and ``differentiable_sample`` (Gumbel-softmax)
  for continuous-relaxation gradient flow through ``probs``.
* :class:`Bernoulli` — binary-outcome convenience wrapper over
  ``Categorical([0, 1], [1 - p, p])`` (T-122-followup-bernoulli).  Adds
  the standard ``p``-only constructor and a native ``log_pmf``.

Each distribution carries an optional ``kind`` tag (``"aleatoric"`` —
irreducible randomness from inherent variability, the default; or
``"epistemic"`` — reducible uncertainty from limited knowledge).
:mod:`jaxonomy.uq.aleatoric_epistemic` consumes this tag to decompose
output variance into per-source contributions (T-126 phase 1).

Example:

    >>> import jax
    >>> from jaxonomy.uq import Uniform
    >>> Uniform(0.0, 1.0).sample(jax.random.PRNGKey(0), (5,))  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol, Sequence

import jax
import jax.numpy as jnp

__all__ = [
    "Distribution",
    "DistributionKind",
    "Uniform",
    "Normal",
    "LogNormal",
    "Triangular",
    "Exponential",
    "Poisson",
    "Categorical",
    "Bernoulli",
    "Beta",
    "Gamma",
    "Weibull",
    "Pareto",
    "MultivariateNormal",
    "CorrelatedMarginals",
]


# Valid values for the ``kind`` attribute on every distribution. Kept as a
# module-level Literal alias so downstream code (and the
# aleatoric/epistemic decomposition helper) can reuse the type annotation.
DistributionKind = Literal["aleatoric", "epistemic"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Distribution(Protocol):
    """Minimum surface every distribution must expose."""

    kind: DistributionKind

    def sample(self, key, shape) -> jnp.ndarray: ...
    def ppf(self, u) -> jnp.ndarray: ...
    def log_pdf(self, x) -> jnp.ndarray: ...


def _validate_kind(kind: str) -> None:
    if kind not in ("aleatoric", "epistemic"):
        raise ValueError(
            f"Distribution.kind must be 'aleatoric' or 'epistemic'; got {kind!r}."
        )


# ---------------------------------------------------------------------------
# Concrete distributions
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Uniform:
    """Uniform on ``[low, high]``."""

    low: float
    high: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError(
                f"Uniform: high ({self.high}) must exceed low ({self.low})."
            )
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
        return self.ppf(u)

    def ppf(self, u) -> jnp.ndarray:
        u = jnp.asarray(u)
        return self.low + (self.high - self.low) * u

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        # Density 1/(high-low) inside the support, 0 outside (-inf in log-space).
        in_support = (x >= self.low) & (x <= self.high)
        log_density = -jnp.log(self.high - self.low)
        return jnp.where(in_support, log_density, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: ``P(X <= x) = (x - low) / (high - low)`` clipped to ``[0, 1]``.

        Differentiable through ``x`` inside the support (slope
        ``1 / (high - low)``); flat (zero gradient) outside.
        """
        x = jnp.asarray(x)
        u = (x - self.low) / (self.high - self.low)
        return jnp.clip(u, 0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Normal:
    """Normal with mean ``loc`` and standard deviation ``scale``."""

    loc: float
    scale: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"Normal: scale ({self.scale}) must be positive.")
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        z = jax.random.normal(key, shape=shape)
        return self.loc + self.scale * z

    def ppf(self, u) -> jnp.ndarray:
        u = jnp.asarray(u)
        # Clamp so the inverse-CDF stays finite at the open boundaries.
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        return self.loc + self.scale * jax.scipy.stats.norm.ppf(u)

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        z = (x - self.loc) / self.scale
        return -0.5 * z * z - jnp.log(self.scale) - 0.5 * jnp.log(2.0 * jnp.pi)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: ``Phi((x - loc) / scale)`` via ``erf``.

        ``0.5 * (1 + erf((x - loc) / (scale * sqrt(2))))``.  Differentiable
        through ``x`` (smooth Gaussian kernel).
        """
        x = jnp.asarray(x)
        z = (x - self.loc) / (self.scale * jnp.sqrt(2.0))
        return 0.5 * (1.0 + jax.scipy.special.erf(z))


@dataclasses.dataclass(frozen=True)
class LogNormal:
    """Log-normal: ``X = exp(Normal(mu, sigma))``."""

    mu: float
    sigma: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"LogNormal: sigma ({self.sigma}) must be positive.")
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        return jnp.exp(Normal(self.mu, self.sigma).sample(key, shape))

    def ppf(self, u) -> jnp.ndarray:
        return jnp.exp(Normal(self.mu, self.sigma).ppf(u))

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        # log p(x) = log Normal_pdf(log x; mu, sigma) - log x
        in_support = x > 0.0
        # Avoid log(0) inside the where; the safe value is unused on the
        # masked branch but the trace must stay finite.
        x_safe = jnp.where(in_support, x, 1.0)
        log_x = jnp.log(x_safe)
        z = (log_x - self.mu) / self.sigma
        log_normal = -0.5 * z * z - jnp.log(self.sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
        result = log_normal - log_x
        return jnp.where(in_support, result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: Normal CDF on ``log(x)``.

        ``Phi((log x - mu) / sigma)`` for ``x > 0``; 0 for ``x <= 0``.
        Differentiable through ``x`` inside the support.
        """
        x = jnp.asarray(x)
        in_support = x > 0.0
        # Guard log(0) so the trace stays finite on the masked branch; the
        # in_support mask drives the final 0 outcome.
        x_safe = jnp.where(in_support, x, 1.0)
        z = (jnp.log(x_safe) - self.mu) / (self.sigma * jnp.sqrt(2.0))
        result = 0.5 * (1.0 + jax.scipy.special.erf(z))
        return jnp.where(in_support, result, 0.0)


@dataclasses.dataclass(frozen=True)
class Triangular:
    """Triangular on ``[low, high]`` peaking at ``mode``."""

    low: float
    mode: float
    high: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high) or self.high <= self.low:
            raise ValueError(
                "Triangular: require low <= mode <= high and high > low; "
                f"got low={self.low}, mode={self.mode}, high={self.high}."
            )
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
        return self.ppf(u)

    def ppf(self, u) -> jnp.ndarray:
        u = jnp.asarray(u)
        c = (self.mode - self.low) / (self.high - self.low)
        # Inverse-CDF for Triangular: piecewise sqrt of the CDF.
        left = self.low + jnp.sqrt(jnp.clip(u * (self.high - self.low) * (self.mode - self.low), 0.0))
        right = self.high - jnp.sqrt(
            jnp.clip((1.0 - u) * (self.high - self.low) * (self.high - self.mode), 0.0)
        )
        return jnp.where(u < c, left, right)

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        # Triangular density:
        #   p(x) = 2 (x - low) / ((high - low)(mode - low))   for low <= x <= mode
        #   p(x) = 2 (high - x) / ((high - low)(high - mode)) for mode <= x <= high
        in_support = (x >= self.low) & (x <= self.high)
        # Guard the two branches against degenerate (mode == low or mode == high)
        # Triangulars: those degenerate cases collapse one branch and the
        # in-support mask handles the singular density there.
        width = self.high - self.low
        left_denom = width * (self.mode - self.low)
        right_denom = width * (self.high - self.mode)
        left_density = jnp.where(
            self.mode > self.low, 2.0 * (x - self.low) / jnp.where(left_denom > 0, left_denom, 1.0), 0.0
        )
        right_density = jnp.where(
            self.high > self.mode, 2.0 * (self.high - x) / jnp.where(right_denom > 0, right_denom, 1.0), 0.0
        )
        density = jnp.where(x < self.mode, left_density, right_density)
        # Clip to a positive value before log to avoid -inf inside the
        # supported branch; the in_support mask still drives the final -inf.
        density_safe = jnp.where(density > 0, density, 1.0)
        result = jnp.log(density_safe)
        return jnp.where(in_support & (density > 0), result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Piecewise-quadratic forward CDF.

        For ``low <= x <= mode``:
            ``F(x) = (x - low)**2 / ((high - low) * (mode - low))``
        For ``mode <= x <= high``:
            ``F(x) = 1 - (high - x)**2 / ((high - low) * (high - mode))``
        Returns 0 below ``low`` and 1 above ``high``.  Differentiable
        through ``x`` inside the support (slope matches the triangular
        density); flat outside.
        """
        x = jnp.asarray(x)
        width = self.high - self.low
        # Clamp x to the support so the quadratic branches stay in their
        # valid domain (we restore 0/1 outside via the final mask).
        x_clamped = jnp.clip(x, self.low, self.high)
        # Guard against degenerate (mode == low or mode == high) Triangulars
        # in the same way as ``log_pdf``: substitute a 1.0 denominator on
        # the collapsed branch and let the piecewise selection drive the
        # final value.
        left_denom = width * (self.mode - self.low)
        right_denom = width * (self.high - self.mode)
        left_cdf = jnp.where(
            self.mode > self.low,
            (x_clamped - self.low) ** 2
            / jnp.where(left_denom > 0, left_denom, 1.0),
            0.0,
        )
        right_cdf = jnp.where(
            self.high > self.mode,
            1.0 - (self.high - x_clamped) ** 2
            / jnp.where(right_denom > 0, right_denom, 1.0),
            1.0,
        )
        # The piecewise branch boundary is at ``x == mode``; both formulas
        # agree there, so a strict ``<`` works cleanly.
        result = jnp.where(x_clamped < self.mode, left_cdf, right_cdf)
        # Restore 0 below low and 1 above high (clamping makes both
        # branches produce boundary values, but make the contract explicit).
        result = jnp.where(x <= self.low, 0.0, result)
        result = jnp.where(x >= self.high, 1.0, result)
        return result


# ---------------------------------------------------------------------------
# T-122-followup-poisson — Exponential and Poisson distributions.
#
# Exponential is a continuous, strictly-positive, single-rate distribution
# useful for event inter-arrival times.  The inverse-CDF
# ``F^{-1}(u) = -log(1 - u) / rate`` is smooth in ``rate``, so the
# standard reparameterisation (``x = -log(1-u)/rate`` with ``u`` drawn
# under stop_gradient) flows gradients cleanly through ``rate``.
#
# Poisson is the discrete count distribution; samples are non-negative
# integers, so it is *not* differentiable through ``rate`` w.r.t. its
# samples.  We ship ``sample`` (wrapped in stop_gradient) and ``log_pmf``
# but deliberately omit ``ppf``: the inverse-CDF of a discrete
# distribution is a step function, not a smooth quantile transform,
# and would break the Saltelli/Morris pipelines that expect a smooth
# mapping from the unit cube to parameter space.  Importance-sampling
# / aleatoric-epistemic decomposition still works because
# ``log_pmf`` is what those routines consume.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Exponential:
    """Exponential with rate ``rate`` (``f(x) = rate * exp(-rate * x)``).

    Mean ``1 / rate``; variance ``1 / rate**2``.  Useful for modelling
    event inter-arrival times when arrivals are memoryless.

    The inverse-CDF reparameterisation ``x = -log(1 - u) / rate`` is
    smooth in ``rate``, so :func:`jax.grad` flows cleanly through the
    rate parameter when ``u`` is drawn under ``stop_gradient``.
    """

    rate: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(
                f"Exponential: rate ({self.rate}) must be positive."
            )
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        # ``jax.random.exponential`` draws a rate=1 sample; scale by
        # ``1/rate`` to apply the desired rate.  Equivalent to the
        # inverse-CDF reparameterisation but uses JAX's tuned sampler.
        z = jax.random.exponential(key, shape=shape)
        return z / self.rate

    def ppf(self, u) -> jnp.ndarray:
        u = jnp.asarray(u)
        # Clamp to keep ``log(1 - u)`` finite at the open right boundary.
        u = jnp.clip(u, 0.0, 1.0 - 1e-12)
        return -jnp.log1p(-u) / self.rate

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        in_support = x >= 0.0
        # log p(x) = log(rate) - rate * x  for x >= 0.
        result = jnp.log(self.rate) - self.rate * x
        return jnp.where(in_support, result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: ``1 - exp(-rate * x)`` for ``x >= 0``; 0 for ``x < 0``.

        Differentiable through ``x`` (smooth exponential).  Numerically
        stable near ``x -> 0`` via the standard ``-expm1`` form.
        """
        x = jnp.asarray(x)
        in_support = x >= 0.0
        # ``-expm1(-rate * x) = 1 - exp(-rate * x)`` is numerically stable
        # near ``x -> 0`` (where the naive subtraction loses precision).
        result = -jnp.expm1(-self.rate * x)
        return jnp.where(in_support, result, 0.0)


@dataclasses.dataclass(frozen=True)
class Poisson:
    """Poisson with rate ``rate`` (``P(k) = rate**k * exp(-rate) / k!``).

    Mean ``rate``; variance ``rate``.  Samples are non-negative integers,
    so this distribution is *not* differentiable through ``rate`` w.r.t.
    its samples — :func:`sample` is wrapped in ``stop_gradient`` and the
    inverse-CDF ``ppf`` is deliberately omitted (step-function inverse;
    unsuitable for the smooth quantile-transform pipelines in Sobol /
    Morris).  ``log_pmf`` is provided for importance-sampling /
    likelihood workflows.
    """

    rate: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(
                f"Poisson: rate ({self.rate}) must be positive."
            )
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        # ``jax.random.poisson`` returns int32/int64 (dtype-policy-aware);
        # wrap in ``stop_gradient`` to make the non-differentiability
        # explicit at the boundary, mirroring the T-122 stochastic-source
        # contract.
        return jax.lax.stop_gradient(
            jax.random.poisson(key, self.rate, shape=shape)
        )

    def log_pmf(self, k) -> jnp.ndarray:
        """Log probability mass at non-negative integer ``k``.

        Returns ``-inf`` for negative ``k``; non-integer ``k`` is
        evaluated against the analytic continuation
        ``k log(rate) - rate - gammaln(k + 1)`` (matches scipy's
        ``poisson.logpmf`` semantics).
        """
        k = jnp.asarray(k)
        in_support = k >= 0
        # log P(k) = k*log(rate) - rate - log(k!) = k*log(rate) - rate - gammaln(k+1)
        result = k * jnp.log(self.rate) - self.rate - jax.scipy.special.gammaln(k + 1.0)
        return jnp.where(in_support, result, -jnp.inf)

    # Alias so callers that uniformly use ``log_pdf`` (the
    # importance-sampling / decomposition code paths) still work; the
    # method-name *log_pmf* is preserved as the canonical entry point.
    def log_pdf(self, x) -> jnp.ndarray:
        return self.log_pmf(x)

    def cdf(self, k) -> jnp.ndarray:
        """Forward CDF: ``P(K <= k) = gammaincc(floor(k) + 1, rate)``.

        This is the standard closed form for the Poisson CDF in terms of
        the upper regularised incomplete gamma function.  Discrete /
        step-shaped in ``k`` (so the gradient w.r.t. ``k`` is zero almost
        everywhere — :func:`jax.grad` returns 0 except at integer steps).
        Returns 0 for ``k < 0``.
        """
        k = jnp.asarray(k)
        in_support = k >= 0
        # ``floor(k) + 1`` is the standard discrete-CDF formula.  Cast
        # through float to keep gammaincc's signature happy.
        floor_k = jnp.floor(k).astype(jnp.float64)
        result = jax.scipy.special.gammaincc(floor_k + 1.0, self.rate)
        return jnp.where(in_support, result, 0.0)


# ---------------------------------------------------------------------------
# T-122-followup-categorical — Categorical / discrete-choice distribution.
#
# Many UQ workflows need to model uncertain *discrete* selections rather than
# continuous variation: a load demand that is 50%, 75%, or 100% with
# probabilities (0.3, 0.5, 0.2); a regime selector that flips between
# "nominal" / "degraded" / "failed"; a Bernoulli outcome (special case
# ``values=[0, 1]``).  ``Categorical(values, probs)`` exposes:
#
#   * ``sample(key, shape)``  — hard categorical draw via
#     ``jax.random.choice``; output dtype follows ``values`` dtype.
#     Non-differentiable in the sample path (indexing through a discrete
#     index is a step function), so ``stop_gradient`` is applied to the
#     selected indices.
#   * ``log_pmf(value)``  — log probability of an observed ``value``
#     (matches the value against the ``values`` table; returns ``-inf``
#     if no match).  Differentiable through ``probs``.
#   * ``differentiable_sample(key, shape, temperature=1.0)`` — Gumbel-
#     softmax (Jang et al. 2017) continuous relaxation: ``softmax(
#     (log probs + Gumbel(0,1)) / temperature)`` produces a soft simplex
#     sample, and we return ``sum(soft @ values)`` so the output is a
#     continuous-valued convex combination of ``values``.  Gradients
#     flow through ``probs`` (and through ``values`` for continuous
#     ``values``).  Note this is a *relaxation*: it is not equal to the
#     hard categorical sample in distribution; lower ``temperature`` ->
#     closer to hard but higher gradient variance.
#
# ``log_pdf`` aliases ``log_pmf`` so callers that uniformly use
# ``log_pdf`` (importance-sampling / decomposition) keep working.
# ``ppf`` is deliberately omitted — same rationale as ``Poisson``: a
# discrete inverse-CDF is a step function and breaks the smooth
# quantile-transform contract of Saltelli/Morris pipelines.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Categorical:
    """Discrete-choice distribution over an explicit ``values`` table.

    ``values`` is a sequence (or array) of possible outputs (any dtype —
    int, float, or vector — the only requirement is that ``jnp.asarray``
    can stack the entries along a leading axis).  ``probs`` is a
    matching-length sequence of non-negative probabilities; if it does
    not sum to exactly 1, it is normalised at construction (with a
    tolerance check on the input — :func:`__post_init__` rejects all-
    zero or negative inputs).

    Use cases:

    * Discrete uncertainty: load demand is 50%, 75%, or 100% with probs
      ``(0.3, 0.5, 0.2)`` -> ``Categorical([0.5, 0.75, 1.0], [0.3, 0.5, 0.2])``.
    * Bernoulli special case: ``Categorical([0, 1], [1 - p, p])``.
    * Regime selection: ``Categorical(["nominal", "degraded", "failed"],
      [0.95, 0.04, 0.01])`` (string ``values`` work for ``log_pmf`` /
      ``sample`` lookups; ``differentiable_sample`` requires numeric
      ``values`` because it returns a weighted sum).

    Differentiability:

    * ``sample`` returns the hard categorical draw and is *not*
      differentiable through ``probs`` (the gather through the selected
      index is a step function).  The selected indices are wrapped in
      ``stop_gradient`` to make this explicit.
    * ``log_pmf`` is differentiable through ``probs``.
    * ``differentiable_sample`` is the Gumbel-softmax relaxation; it
      *is* differentiable through ``probs`` but is a continuous
      approximation, not the hard sample.
    """

    values: Sequence
    probs: Sequence
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        values_arr = jnp.asarray(self.values)
        probs_arr = jnp.asarray(self.probs, dtype=jnp.float64)
        if probs_arr.ndim != 1:
            raise ValueError(
                f"Categorical: probs must be 1D; got shape {probs_arr.shape}."
            )
        if values_arr.shape[0] != probs_arr.shape[0]:
            raise ValueError(
                f"Categorical: values length {values_arr.shape[0]} must "
                f"match probs length {probs_arr.shape[0]}."
            )
        if probs_arr.shape[0] == 0:
            raise ValueError("Categorical: probs must be non-empty.")
        # Concrete-trace validation: reject negative probs and all-zero
        # input.  We use ``bool(...)`` so the check fires at construction
        # under concrete values; if these arrive as tracers (e.g. the
        # user constructs a Categorical inside jit), JAX raises a
        # ConcretisationError, which is the right failure mode.
        if bool(jnp.any(probs_arr < 0.0)):
            raise ValueError(
                f"Categorical: probs must be non-negative; got {probs_arr}."
            )
        total = float(jnp.sum(probs_arr))
        if total <= 0.0:
            raise ValueError(
                f"Categorical: probs must have positive sum; got {probs_arr}."
            )
        _validate_kind(self.kind)
        # Normalise probs to sum exactly to 1 (within float roundoff).
        # We do *not* re-normalise on every sample — store the normalised
        # form so ``log_pmf`` and ``sample`` agree.
        object.__setattr__(self, "values", values_arr)
        object.__setattr__(self, "probs", probs_arr / total)

    @property
    def n_categories(self) -> int:
        return int(self.probs.shape[0])

    def sample(self, key, shape) -> jnp.ndarray:
        """Hard categorical draw.  Returns elements of ``values`` per ``probs``.

        Output shape is ``shape`` (broadcast over the trailing per-value
        shape if ``values`` are vector-typed) — for scalar ``values``,
        the output shape exactly equals ``shape``.

        The selected indices are wrapped in ``stop_gradient`` to make
        the non-differentiability of the hard sample explicit.
        """
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        # ``jax.random.choice`` with ``p=probs`` is the canonical
        # categorical sampler.  We index into ``values`` by the returned
        # integer indices so per-category outputs (vectors, etc.) round-
        # trip correctly.
        idx = jax.random.choice(
            key, self.n_categories, shape=shape, p=self.probs
        )
        idx = jax.lax.stop_gradient(idx)
        return self.values[idx]

    def log_pmf(self, value) -> jnp.ndarray:
        """Log probability of observing ``value``.

        Matches ``value`` against the ``values`` table (with broadcasting
        if ``value`` has the leading shape of a batch).  Returns ``-inf``
        if ``value`` does not match any entry in ``values``.
        Differentiable through ``probs``.
        """
        value = jnp.asarray(value)
        # Build a match mask of shape (n_categories, *value.shape):
        # element-wise equality with the ``values`` table broadcast
        # across the value tensor.  For vector-typed ``values``, we
        # compare on the trailing axis with ``jnp.all`` along the value-
        # element axis.
        if self.values.ndim == 1:
            # Scalar-valued categorical.  match[i, ...] = (values[i] == value)
            match = self.values.reshape((self.n_categories,) + (1,) * value.ndim) == value
        else:
            # Vector-valued categorical: compare the trailing dim.
            value_shape = value.shape
            if value.ndim == 0 or value.shape[-1] != self.values.shape[-1]:
                # No structural match -> all -inf.
                return jnp.full(value_shape[:-1] if value_shape else (), -jnp.inf)
            # match[i, ...] = all(values[i] == value[..., :])
            reshaped = self.values.reshape(
                (self.n_categories,) + (1,) * (value.ndim - 1) + self.values.shape[1:]
            )
            match = jnp.all(reshaped == value, axis=-1)
        # logp[...] = log(sum_i probs[i] * match[i, ...]).  If no
        # category matches, the sum is zero and log -> -inf, which is
        # the documented behaviour.
        probs_reshaped = self.probs.reshape((self.n_categories,) + (1,) * (match.ndim - 1))
        weighted = jnp.where(match, probs_reshaped, 0.0)
        total = jnp.sum(weighted, axis=0)
        # ``jnp.log(0) = -inf`` — that is exactly what we want for
        # unmatched values.  Suppress the runtime warning by routing
        # through a safe value.
        safe_total = jnp.where(total > 0.0, total, 1.0)
        result = jnp.log(safe_total)
        return jnp.where(total > 0.0, result, -jnp.inf)

    # Alias so callers that uniformly use ``log_pdf`` keep working.
    def log_pdf(self, x) -> jnp.ndarray:
        return self.log_pmf(x)

    def differentiable_sample(
        self, key, shape, temperature: float = 1.0
    ) -> jnp.ndarray:
        """Gumbel-softmax continuous-relaxation sample.

        Returns a continuous-valued convex combination of ``values``:

            soft = softmax((log probs + Gumbel(0,1)) / temperature)
            out  = sum_i soft[i] * values[i]

        Gradients flow through ``probs`` (and through ``values`` for
        continuous ``values``).  This is *not* equal to the hard sample
        in distribution — it is a relaxation that converges to the hard
        sample as ``temperature -> 0`` (with diverging gradient
        variance) and approaches a uniform mix as ``temperature -> inf``.

        Requires numeric ``values`` (the weighted sum is meaningless for
        string / object dtypes).
        """
        if temperature <= 0:
            raise ValueError(
                f"differentiable_sample: temperature ({temperature}) must be positive."
            )
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        if not jnp.issubdtype(self.values.dtype, jnp.number):
            raise TypeError(
                "differentiable_sample requires numeric ``values``; "
                f"got dtype {self.values.dtype}."
            )
        # Draw Gumbel(0,1) noise as -log(-log(u)) for u ~ U(0,1).
        # Clamp ``u`` away from the open boundaries so the double-log
        # stays finite.
        u = jax.random.uniform(
            key, shape=shape + (self.n_categories,), minval=1e-12, maxval=1.0 - 1e-12
        )
        gumbel = -jnp.log(-jnp.log(u))
        log_probs = jnp.log(self.probs)
        # log_probs broadcasts across ``shape``.
        logits = (log_probs + gumbel) / temperature
        soft = jax.nn.softmax(logits, axis=-1)
        # Convex combination of ``values`` weighted by ``soft``.
        # For scalar values: soft @ values -> shape == ``shape``.
        # For vector values: soft @ values -> shape == ``shape + values.shape[1:]``.
        if self.values.ndim == 1:
            return jnp.tensordot(soft, self.values, axes=([-1], [0]))
        else:
            return jnp.tensordot(soft, self.values, axes=([-1], [0]))

    # T-122-followup-categorical: ``ppf`` is deliberately omitted (step-
    # function inverse-CDF; same rationale as ``Poisson``).

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: ``sum(probs[i] for i where values[i] <= x)``.

        Requires sortable scalar ``values`` (the comparison ``values[i] <= x``
        must be well defined elementwise).  For non-numeric or vector-typed
        ``values``, raises :class:`TypeError` — there is no natural ordering
        on those, and no scalar CDF.

        Step-shaped in ``x``: the gradient w.r.t. ``x`` is zero almost
        everywhere (jumps at each value in the table).  Returns 0 below
        the smallest value, 1 at-or-above the largest.
        """
        if self.values.ndim != 1 or not jnp.issubdtype(self.values.dtype, jnp.number):
            raise TypeError(
                "Categorical.cdf requires sortable scalar numeric values; "
                f"got dtype {self.values.dtype} ndim {self.values.ndim}."
            )
        x = jnp.asarray(x)
        # Build a mask of shape (n_categories, *x.shape): True where the
        # category value is <= x.  Sum the corresponding probs along the
        # category axis.
        values_b = self.values.reshape((self.n_categories,) + (1,) * x.ndim)
        probs_b = self.probs.reshape((self.n_categories,) + (1,) * x.ndim)
        mask = values_b <= x
        return jnp.sum(jnp.where(mask, probs_b, 0.0), axis=0)


# ---------------------------------------------------------------------------
# T-122-followup-bernoulli — Bernoulli convenience distribution.
#
# Bernoulli is the two-outcome special case of Categorical with values
# ``[0, 1]`` and ``probs=[1-p, p]``.  Binary events (coin flips,
# masking, failure indicators) are common enough in simulation work
# that a dedicated ``Bernoulli(p)`` constructor is far more readable
# than the equivalent ``Categorical([0, 1], [1 - p, p])`` spelling.
#
# Honest-fallback note (per the task spec): we ship Bernoulli as a thin
# subclass-by-composition of ``Categorical``: ``sample`` and
# ``differentiable_sample`` delegate to the underlying Categorical so
# the math is shared.  ``log_pmf`` has a native two-branch
# implementation because the direct ``log(p) if k==1 else log(1-p)``
# expression is cheaper and more numerically robust than the generic
# table lookup, and because gradients through ``p`` in the native form
# avoid the comparison-mask plumbing.  The redundancy is intentional:
# Bernoulli is a convenience API, not a new mathematical primitive.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Bernoulli:
    """Bernoulli distribution: ``P(1) = p``, ``P(0) = 1 - p``.

    Convenience wrapper over :class:`Categorical` for the very common
    binary-outcome special case.  Use cases:

    * Coin flips / Bernoulli trials: ``Bernoulli(p=0.5).sample(...)``.
    * Binary masking: outputs in ``{0, 1}`` to gate other signals.
    * Failure indicators: ``Bernoulli(p=failure_rate)``.

    The redundancy with ``Categorical([0, 1], [1-p, p])`` is intentional
    — Bernoulli's single-scalar ``p`` API is significantly more readable
    than the two-list form, and ``log_pmf`` admits a cheaper native
    expression.  See the module-level docstring for the broader rationale.

    Differentiability:

    * ``sample`` is the hard binary draw and is *not* differentiable
      through ``p`` (the discrete sample is wrapped in ``stop_gradient``
      via the underlying Categorical sampler).
    * ``log_pmf`` is differentiable through ``p`` (smooth ``log(p)`` /
      ``log(1-p)`` branches).
    * ``differentiable_sample`` is the Gumbel-softmax relaxation
      delegated to the underlying Categorical; *is* differentiable
      through ``p`` but is a continuous approximation, not the hard
      sample.
    """

    p: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        # ``p`` must be a probability — reject anything outside [0, 1].
        # Use ``float(...)`` to fail fast on tracers (mirrors the
        # Categorical / Normal / Uniform construction-time validation).
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"Bernoulli: p ({self.p}) must be in [0, 1]."
            )
        _validate_kind(self.kind)
        # Build the underlying Categorical exactly once; reuse for all
        # sample / differentiable_sample calls.  Storing it on the
        # frozen dataclass requires ``object.__setattr__``.
        cat = Categorical(values=[0, 1], probs=[1.0 - p, p], kind=self.kind)
        object.__setattr__(self, "_categorical", cat)

    def sample(self, key, shape) -> jnp.ndarray:
        """Hard binary draw.  Returns 0/1 samples per ``p``.

        Delegates to the underlying Categorical's ``sample`` so the
        PRNG-stream / shape / stop_gradient semantics match exactly.
        """
        return self._categorical.sample(key, shape)

    def log_pmf(self, k) -> jnp.ndarray:
        """Log probability mass at ``k`` (0 or 1).

        Returns ``log(p)`` for ``k == 1``, ``log(1 - p)`` for ``k == 0``,
        and ``-inf`` otherwise.  Differentiable through ``p`` (smooth
        ``log`` branches; the where-mask is constant in ``p`` so
        gradient flow is clean).
        """
        k = jnp.asarray(k)
        p = jnp.asarray(self.p)
        # Two-branch native form — cheaper than Categorical's general
        # table lookup, and the gradient through ``p`` flows via the
        # straight ``jnp.log`` calls (no comparison-mask gymnastics).
        # Guard log(0) at the boundary p in {0, 1}: ``jnp.log`` already
        # returns ``-inf`` cleanly, which is exactly what we want.
        is_one = k == 1
        is_zero = k == 0
        in_support = is_one | is_zero
        # Use ``log1p(-p)`` for log(1-p): numerically stable near p -> 1.
        log_p = jnp.log(p)
        log_1mp = jnp.log1p(-p)
        result = jnp.where(is_one, log_p, log_1mp)
        return jnp.where(in_support, result, -jnp.inf)

    # Alias so callers that uniformly use ``log_pdf`` (importance-
    # sampling / decomposition) keep working.
    def log_pdf(self, x) -> jnp.ndarray:
        return self.log_pmf(x)

    def differentiable_sample(
        self, key, shape, temperature: float = 1.0
    ) -> jnp.ndarray:
        """Gumbel-softmax continuous-relaxation sample.

        Delegates to the underlying ``Categorical([0, 1], [1-p, p])``'s
        ``differentiable_sample``.  Returns a continuous-valued
        approximation in ``[0, 1]`` whose gradient flows through ``p``
        via the Gumbel-softmax reparameterisation.  See
        :meth:`Categorical.differentiable_sample` for the precise
        formula and gradient-variance / temperature trade-off.
        """
        return self._categorical.differentiable_sample(
            key, shape, temperature=temperature
        )

    # T-122-followup-bernoulli: ``ppf`` is deliberately omitted (same
    # rationale as ``Categorical`` and ``Poisson`` — step-function
    # inverse CDF; breaks Saltelli/Morris smooth-quantile pipelines).

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: step function ``0`` for ``x < 0``, ``1 - p`` for
        ``0 <= x < 1``, ``1`` for ``x >= 1``.

        Step-shaped in ``x`` so the gradient w.r.t. ``x`` is zero
        almost everywhere; smooth in ``p`` (linear in ``1 - p``).
        """
        x = jnp.asarray(x)
        p = jnp.asarray(self.p)
        # Three regions: below 0 -> 0; in [0, 1) -> 1-p; >= 1 -> 1.
        below = x < 0.0
        above = x >= 1.0
        middle_value = 1.0 - p
        # Default to middle, then override the two edge regions.
        result = jnp.where(below, jnp.zeros_like(x + p), middle_value + 0.0 * x)
        result = jnp.where(above, jnp.ones_like(result), result)
        return result


# ---------------------------------------------------------------------------
# T-122-followup-beta-gamma — Beta and Gamma distributions.
#
# Two continuous distributions that round out the parameter-uncertainty
# toolkit:
#
#   * ``Beta(alpha, beta)`` — supported on ``[0, 1]``.  Canonical model
#     for bounded fractions (utilisation, mixture weights), probabilities
#     of probabilities (Bayesian priors on a Bernoulli ``p``), and
#     beta-binomial workflows.  Mean = ``alpha / (alpha + beta)``;
#     variance = ``alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))``.
#   * ``Gamma(shape, scale)`` — supported on ``[0, inf)``.  Canonical
#     model for wait times (sum of ``shape`` exponentials), positive-
#     valued physical parameters (resistance, viscosity), and prior
#     distributions over rate parameters.  Mean = ``shape * scale``;
#     variance = ``shape * scale**2``.
#
# Both expose ``sample(key, shape)``, ``log_pdf(x)`` (analytic via
# ``gammaln``), and ``ppf(u)``.
#
# Differentiability:
#   * ``Gamma`` sampling routes through ``jax.random.gamma`` (which
#     under the hood uses the Marsaglia–Tsang reparameterisation for
#     shape >= 1 and a boost trick for shape < 1) and then multiplies
#     by ``scale``.  Gradients flow cleanly through ``scale`` via the
#     scale-by-multiplication path; gradients through ``shape`` flow
#     via JAX's implicit-reparam machinery (``jax.random.gamma`` is
#     reparameterised end-to-end).
#   * ``Beta`` sampling routes through ``jax.random.beta``.  Beta
#     reparameterisation is implicit and not a straight inverse-CDF;
#     JAX's underlying sampler does support gradient flow through
#     ``alpha`` and ``beta`` via implicit reparam, but the gradient
#     variance can be high — callers needing tight gradient estimates
#     should consider score-function / RSVGD style estimators.  We
#     document this caveat rather than gating sampling.
#
# ``ppf`` uses ``jax.pure_callback`` into ``scipy.stats`` because the
# inverse-regularised-incomplete-gamma / -beta functions are not yet
# exposed in ``jax.scipy.special``.  This is host-side and breaks
# ``jit``-compiled gradient flow through ``u``; the ``ppf`` is only
# used by the Saltelli / Morris pipelines, which evaluate eagerly
# anyway.  Numerically matches scipy bit-for-bit (it *is* scipy).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Beta:
    """Beta distribution on ``[0, 1]`` with shape parameters ``alpha``, ``beta``.

    PDF::

        p(x) = Gamma(alpha + beta) / (Gamma(alpha) Gamma(beta))
               * x**(alpha - 1) * (1 - x)**(beta - 1)

    Mean ``alpha / (alpha + beta)``; mode (for ``alpha, beta > 1``)
    ``(alpha - 1) / (alpha + beta - 2)``.

    Use cases: bounded fractions, mixture weights, conjugate prior on a
    Bernoulli/Binomial ``p`` (Bayesian beta-binomial workflow), models
    of capacity utilisation or duty-cycle uncertainty.

    Differentiability: ``sample`` routes through ``jax.random.beta``
    which supports implicit-reparam gradients through ``alpha`` and
    ``beta`` (though with higher gradient variance than the
    location-scale reparam used by ``Normal`` / ``LogNormal``).
    ``log_pdf`` is differentiable through ``alpha`` / ``beta``
    analytically via ``gammaln``.  ``ppf`` uses a SciPy
    ``pure_callback`` (host-side, eager — see module-level note).
    """

    alpha: float
    beta: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(
                f"Beta: alpha ({self.alpha}) must be positive."
            )
        if self.beta <= 0:
            raise ValueError(
                f"Beta: beta ({self.beta}) must be positive."
            )
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        # ``jax.random.beta`` takes ``a, b`` shape parameters and an
        # output ``shape``.  The result is on the open ``(0, 1)``
        # interval.  Cast to float64 to preserve the T-005 dtype policy.
        return jax.random.beta(key, self.alpha, self.beta, shape=shape)

    def ppf(self, u) -> jnp.ndarray:
        """Inverse-CDF via ``scipy.stats.beta.ppf`` (host callback).

        ``jax.scipy.special`` does not (yet) ship ``betaincinv``, so we
        route through SciPy via ``jax.pure_callback``.  This is eager-
        only — fine for the Saltelli / Morris quasi-random pipelines
        that consume ``ppf`` outside ``jit`` — and matches SciPy bit-
        for-bit numerically.
        """
        import scipy.stats as _sps

        u = jnp.asarray(u)
        u_clamped = jnp.clip(u, 1e-12, 1.0 - 1e-12)

        def _host(u_arr):
            import numpy as _np

            return _np.asarray(
                _sps.beta.ppf(u_arr, self.alpha, self.beta), dtype=_np.float64
            )

        result_shape = jax.ShapeDtypeStruct(u_clamped.shape, jnp.float64)
        return jax.pure_callback(_host, result_shape, u_clamped)

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        # log p(x) = (alpha-1) log x + (beta-1) log(1-x)
        #            + gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta).
        in_support = (x > 0.0) & (x < 1.0)
        # Guard the log calls against 0 / 1 boundary so the trace stays
        # finite; the in_support mask drives the final -inf.
        x_safe = jnp.where(in_support, x, 0.5)
        log_x = jnp.log(x_safe)
        log_1mx = jnp.log1p(-x_safe)
        gln = jax.scipy.special.gammaln
        log_norm = gln(self.alpha + self.beta) - gln(self.alpha) - gln(self.beta)
        result = (
            (self.alpha - 1.0) * log_x
            + (self.beta - 1.0) * log_1mx
            + log_norm
        )
        # Boundary: pdf is finite at 0/1 only for alpha == 1 or beta == 1,
        # but the standard convention (matching scipy) is that the open-
        # interval support is what we report; samples at exactly 0 or 1
        # are zero-measure events.
        return jnp.where(in_support, result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: regularised incomplete beta ``I_x(alpha, beta)``.

        Uses :func:`jax.scipy.special.betainc`, which is differentiable
        through ``x``.  Returns 0 for ``x <= 0`` and 1 for ``x >= 1``.
        """
        x = jnp.asarray(x)
        # Clamp inside [0, 1] so betainc stays in domain; the where-mask
        # below restores the 0/1 boundary semantics.
        x_clamped = jnp.clip(x, 0.0, 1.0)
        result = jax.scipy.special.betainc(self.alpha, self.beta, x_clamped)
        result = jnp.where(x <= 0.0, 0.0, result)
        result = jnp.where(x >= 1.0, 1.0, result)
        return result


@dataclasses.dataclass(frozen=True)
class Gamma:
    """Gamma distribution on ``[0, inf)`` with shape ``k`` and scale ``theta``.

    PDF::

        p(x) = x**(k - 1) * exp(-x / theta) / (theta**k * Gamma(k))

    Mean ``shape * scale``; variance ``shape * scale**2``.

    Use cases: wait times (sum of ``shape`` exponentials with rate
    ``1/scale``), positive-valued physical parameters (resistance,
    viscosity, time constants), conjugate prior over a rate / precision
    parameter in Bayesian workflows.

    Differentiability: ``sample`` routes through ``jax.random.gamma``
    (which implements the reparameterised Marsaglia–Tsang sampler for
    ``shape >= 1`` and the boosted-shape trick for ``shape < 1``) and
    then multiplies by ``scale``.  Gradients flow cleanly through both
    parameters: the ``scale`` multiplication is a trivial location-
    scale reparam, and the ``shape`` gradient is supported by JAX's
    implicit-reparam machinery inside ``jax.random.gamma``.
    ``log_pdf`` is differentiable analytically through both parameters
    via ``gammaln``.  ``ppf`` uses a SciPy ``pure_callback`` (host-
    side, eager — see module-level note).
    """

    shape_param: float
    scale: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.shape_param <= 0:
            raise ValueError(
                f"Gamma: shape ({self.shape_param}) must be positive."
            )
        if self.scale <= 0:
            raise ValueError(
                f"Gamma: scale ({self.scale}) must be positive."
            )
        _validate_kind(self.kind)

    @property
    def shape(self) -> float:
        """Alias for ``shape_param`` (matches scipy ``a`` parameter naming).

        The dataclass field is named ``shape_param`` to avoid colliding
        with the conventional ``shape`` argument to ``sample(key, shape)``
        (which is the *output array shape*, not the distribution's shape
        parameter).  Read access via ``.shape`` is provided as a
        convenience.
        """
        return self.shape_param

    def sample(self, key, shape) -> jnp.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        # ``jax.random.gamma`` draws a ``shape=k``, ``scale=1`` sample;
        # multiply by ``scale`` to apply the requested scale.  This is
        # the standard scale-reparameterisation: dGamma(k, theta)/dtheta
        # is well-defined via the multiplicative form.
        z = jax.random.gamma(key, self.shape_param, shape=shape)
        return z * self.scale

    def ppf(self, u) -> jnp.ndarray:
        """Inverse-CDF via ``scipy.stats.gamma.ppf`` (host callback).

        ``jax.scipy.special`` does not (yet) ship ``gammaincinv``, so
        we route through SciPy via ``jax.pure_callback``.  Equivalently
        ``scale * gammaincinv(shape, u)``.  Host-side / eager-only,
        matching SciPy bit-for-bit.
        """
        import scipy.stats as _sps

        u = jnp.asarray(u)
        u_clamped = jnp.clip(u, 1e-12, 1.0 - 1e-12)

        def _host(u_arr):
            import numpy as _np

            return _np.asarray(
                _sps.gamma.ppf(u_arr, self.shape_param, scale=self.scale),
                dtype=_np.float64,
            )

        result_shape = jax.ShapeDtypeStruct(u_clamped.shape, jnp.float64)
        return jax.pure_callback(_host, result_shape, u_clamped)

    def log_pdf(self, x) -> jnp.ndarray:
        x = jnp.asarray(x)
        # log p(x) = (k - 1) log x - x/theta - k log theta - gammaln(k).
        in_support = x > 0.0
        x_safe = jnp.where(in_support, x, 1.0)
        log_x = jnp.log(x_safe)
        k = self.shape_param
        theta = self.scale
        gln = jax.scipy.special.gammaln
        result = (
            (k - 1.0) * log_x
            - x_safe / theta
            - k * jnp.log(theta)
            - gln(k)
        )
        return jnp.where(in_support, result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: regularised lower-incomplete gamma ``P(k, x/scale)``.

        Uses :func:`jax.scipy.special.gammainc`, which is differentiable
        through ``x``.  Returns 0 for ``x <= 0``.
        """
        x = jnp.asarray(x)
        in_support = x > 0.0
        # Guard the division so the trace stays finite on the masked branch.
        x_safe = jnp.where(in_support, x, 0.0)
        result = jax.scipy.special.gammainc(self.shape_param, x_safe / self.scale)
        return jnp.where(in_support, result, 0.0)


# ---------------------------------------------------------------------------
# T-122-followup-weibull — Weibull distribution for reliability / failure
# modeling.
#
# ``Weibull(shape, scale)`` with PDF
#
#     f(x) = (k/lambda) * (x/lambda)**(k-1) * exp(-(x/lambda)**k)   x >= 0
#
# where ``k = shape`` and ``lambda = scale``.  Canonical model for:
#   * Component lifetimes / time-to-failure (reliability engineering).
#   * Wind-speed distributions (k ~ 2 is the Rayleigh special case).
#   * Particle-size / wear-out / fatigue distributions.
#
# Inverse-CDF reparameterisation is closed-form:
#
#     ppf(u) = scale * (-log(1 - u))**(1 / shape)
#
# Smooth in *both* ``shape`` and ``scale``, so :func:`jax.grad` flows
# cleanly through both parameters via the standard reparameterisation
# trick (``u ~ U[0,1)`` drawn under ``stop_gradient``; the gradient
# enters analytically through the closed-form quantile transform).
# This is the same pattern as ``Exponential`` (Weibull's shape=1 special
# case) and gives Weibull a strictly better differentiability profile
# than the implicit-reparam ``Gamma`` sampler.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Weibull:
    """Weibull distribution on ``[0, inf)`` with shape ``k`` and scale ``lambda``.

    PDF::

        p(x) = (k / scale) * (x / scale)**(k - 1) * exp(-(x / scale)**k)

    Mean ``scale * Gamma(1 + 1/shape)``; variance
    ``scale**2 * (Gamma(1 + 2/shape) - Gamma(1 + 1/shape)**2)``.

    Use cases: component lifetimes (reliability engineering's bread-and-
    butter — ``shape < 1`` models infant-mortality, ``shape == 1`` is
    Exponential / memoryless, ``shape > 1`` models wear-out), wind-speed
    distributions, fatigue / wear-out modelling, particle-size
    distributions.

    Differentiability: ``sample`` and ``ppf`` use the closed-form
    inverse-CDF reparameterisation ``x = scale * (-log(1-u))**(1/shape)``
    which is smooth in *both* ``shape`` and ``scale``.  :func:`jax.grad`
    flows cleanly through both parameters with low gradient variance
    (no implicit-reparam machinery needed).  ``log_pdf`` is also
    analytically differentiable through both.
    """

    shape_param: float
    scale: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.shape_param <= 0:
            raise ValueError(
                f"Weibull: shape ({self.shape_param}) must be positive."
            )
        if self.scale <= 0:
            raise ValueError(
                f"Weibull: scale ({self.scale}) must be positive."
            )
        _validate_kind(self.kind)

    @property
    def shape(self) -> float:
        """Alias for ``shape_param`` (matches scipy ``c`` parameter naming).

        The dataclass field is named ``shape_param`` to avoid colliding
        with the conventional ``shape`` argument to ``sample(key, shape)``
        (which is the *output array shape*, not the distribution's shape
        parameter).  Read access via ``.shape`` is provided as a
        convenience.
        """
        return self.shape_param

    def sample(self, key, shape) -> jnp.ndarray:
        """Draw Weibull samples via the closed-form inverse CDF."""
        u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
        return self.ppf(u)

    def ppf(self, u) -> jnp.ndarray:
        """Closed-form inverse CDF: ``scale * (-log(1 - u))**(1/shape)``.

        Smooth in both ``shape`` and ``scale``; gradients flow analytically.
        """
        u = jnp.asarray(u)
        # Clamp so ``log(1 - u)`` stays finite at the open right boundary.
        u_clamped = jnp.clip(u, 0.0, 1.0 - 1e-12)
        # ``log1p(-u)`` is numerically stable near ``u -> 0``.
        return self.scale * jnp.power(-jnp.log1p(-u_clamped), 1.0 / self.shape_param)

    def log_pdf(self, x) -> jnp.ndarray:
        """Analytic log-pdf.

        ``log f(x) = log(k) - log(scale) + (k-1) * log(x/scale) - (x/scale)**k``
        for ``x > 0``; ``-inf`` otherwise.  Differentiable through both
        ``shape`` and ``scale``.
        """
        x = jnp.asarray(x)
        in_support = x > 0.0
        # Guard ``log(0)`` so the trace stays finite on the masked branch.
        x_safe = jnp.where(in_support, x, 1.0)
        k = self.shape_param
        lam = self.scale
        z = x_safe / lam
        # log f = log(k/lam) + (k-1)*log(z) - z**k
        result = (
            jnp.log(k)
            - jnp.log(lam)
            + (k - 1.0) * jnp.log(z)
            - jnp.power(z, k)
        )
        return jnp.where(in_support, result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: ``1 - exp(-(x / scale)**shape)`` for ``x >= 0``.

        Closed-form, smooth in ``x``, ``shape``, and ``scale``.
        Numerically stable via ``-expm1`` near the origin.
        """
        x = jnp.asarray(x)
        in_support = x >= 0.0
        # Guard against negative x in the power: clamp to 0.0 on the masked
        # branch so the trace stays finite (the where below restores 0).
        x_safe = jnp.where(in_support, x, 0.0)
        z = x_safe / self.scale
        result = -jnp.expm1(-jnp.power(z, self.shape_param))
        return jnp.where(in_support, result, 0.0)


# ---------------------------------------------------------------------------
# T-122-followup-pareto — Pareto (heavy-tail / power-law) distribution.
#
# ``Pareto(scale, alpha)`` with PDF
#
#     f(x) = alpha * scale**alpha / x**(alpha + 1)   for x >= scale
#
# Canonical heavy-tail / power-law distribution.  Use cases:
#   * Wealth / income distributions ("Pareto principle", 80/20 rule).
#   * File-size / packet-size distributions in network traffic.
#   * Component-failure tail behaviour where Weibull underestimates the
#     extreme upper tail.
#   * Earthquake magnitudes, city populations, word frequencies (Zipf
#     is the discrete cousin).
#
# The mean ``alpha * scale / (alpha - 1)`` is finite only for
# ``alpha > 1``; the variance only for ``alpha > 2``.  This is the
# defining feature of a heavy tail: low-``alpha`` Paretos have
# undefined moments.
#
# Inverse-CDF reparameterisation is closed-form:
#
#     ppf(u) = scale * (1 - u)**(-1/alpha)
#
# Smooth in *both* ``scale`` and ``alpha``, so :func:`jax.grad` flows
# cleanly through both parameters via the standard reparameterisation
# trick (``u ~ U[0, 1)`` drawn under ``stop_gradient``; the gradient
# enters analytically through the closed-form quantile transform).
# Same differentiability story as ``Exponential`` and ``Weibull``.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pareto:
    """Pareto distribution on ``[scale, inf)`` with shape ``alpha``.

    PDF::

        p(x) = alpha * scale**alpha / x**(alpha + 1)   for x >= scale

    Mean ``alpha * scale / (alpha - 1)`` (finite only for ``alpha > 1``);
    variance ``alpha * scale**2 / ((alpha - 1)**2 * (alpha - 2))``
    (finite only for ``alpha > 2``).  These divergent moments are the
    defining heavy-tail property — a low-``alpha`` Pareto has all its
    mass in the upper tail, and most extreme-value workflows lean on
    this rather than the comparatively light Gaussian / Exponential
    tails.

    Use cases: wealth / income (Pareto principle), file-size / packet-
    size distributions in network traffic, extreme-value modelling
    where Weibull underestimates the upper tail, earthquake
    magnitudes, city populations.

    Differentiability: ``sample`` and ``ppf`` use the closed-form
    inverse-CDF reparameterisation ``x = scale * (1 - u)**(-1 / alpha)``
    which is smooth in *both* ``scale`` and ``alpha``.  :func:`jax.grad`
    flows cleanly through both parameters analytically (no implicit-
    reparam machinery needed).  ``log_pdf`` is also analytically
    differentiable through both.
    """

    scale: float
    alpha: float
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(
                f"Pareto: scale ({self.scale}) must be positive."
            )
        if self.alpha <= 0:
            raise ValueError(
                f"Pareto: alpha ({self.alpha}) must be positive."
            )
        _validate_kind(self.kind)

    def sample(self, key, shape) -> jnp.ndarray:
        """Draw Pareto samples via the closed-form inverse CDF."""
        u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
        return self.ppf(u)

    def ppf(self, u) -> jnp.ndarray:
        """Closed-form inverse CDF: ``scale * (1 - u)**(-1 / alpha)``.

        Smooth in both ``scale`` and ``alpha``; gradients flow analytically.
        """
        u = jnp.asarray(u)
        # Clamp so ``(1 - u)`` stays strictly positive at the open right
        # boundary — the ``-1/alpha`` exponent diverges at ``u -> 1``.
        u_clamped = jnp.clip(u, 0.0, 1.0 - 1e-12)
        # Compute via ``exp(-log1p(-u) / alpha)`` for numerical stability
        # near ``u -> 0`` (where ``1 - u`` is close to 1 and the naive
        # ``(1 - u)**(-1/alpha)`` form loses precision in the log).
        return self.scale * jnp.exp(-jnp.log1p(-u_clamped) / self.alpha)

    def log_pdf(self, x) -> jnp.ndarray:
        """Analytic log-pdf.

        ``log f(x) = log(alpha) + alpha * log(scale) - (alpha + 1) * log(x)``
        for ``x >= scale``; ``-inf`` otherwise.  Differentiable through
        both ``scale`` and ``alpha``.
        """
        x = jnp.asarray(x)
        in_support = x >= self.scale
        # Guard ``log(x)`` so the trace stays finite on the masked branch
        # (``x`` could be 0 or negative outside the support).
        x_safe = jnp.where(in_support, x, self.scale)
        result = (
            jnp.log(self.alpha)
            + self.alpha * jnp.log(self.scale)
            - (self.alpha + 1.0) * jnp.log(x_safe)
        )
        return jnp.where(in_support, result, -jnp.inf)

    def cdf(self, x) -> jnp.ndarray:
        """Forward CDF: ``1 - (scale / x)**alpha`` for ``x >= scale``; 0 below.

        Closed-form, smooth in ``x``, ``scale``, and ``alpha``.
        """
        x = jnp.asarray(x)
        in_support = x >= self.scale
        # Guard against x below ``scale`` (and especially x <= 0) in the
        # power: substitute ``scale`` on the masked branch so the trace
        # stays finite; the where below restores 0.
        x_safe = jnp.where(in_support, x, self.scale)
        result = 1.0 - jnp.power(self.scale / x_safe, self.alpha)
        return jnp.where(in_support, result, 0.0)


# ---------------------------------------------------------------------------
# T-126-followup-correlated-multivariate — correlated multivariate sampling.
#
# Real-world parameters often covary (battery R0 vs capacity; engine mass vs
# stiffness).  Univariate marginal sampling misses this correlation and can
# produce non-physical combinations.
#
# Two new distributions:
#   * ``MultivariateNormal(means, cov)`` — sample N(mu, Sigma) via Cholesky.
#     Differentiable through both ``means`` and ``cov`` (Cholesky is smooth
#     on the cone of SPD matrices).
#   * ``CorrelatedMarginals(marginals, corr_matrix)`` — Gaussian copula.
#     Sample a unit-variance multivariate Normal with the prescribed
#     correlation, push each component through the standard-normal CDF to
#     get uniforms, then through each marginal's ``ppf``.  Rank correlation
#     (Spearman) is preserved exactly; Pearson correlation is approximated
#     (the copula transform is rank-preserving but not linear).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MultivariateNormal:
    """Multivariate Normal ``N(means, cov)``.

    ``means`` is a length-``n`` vector; ``cov`` is an ``n x n`` symmetric
    positive-definite covariance matrix.  Sampling uses a Cholesky factor
    of ``cov``, which is smooth on the SPD cone, so :func:`jax.grad`
    flows through both ``means`` and ``cov`` (or, more naturally, through
    the Cholesky factor if you parameterise that directly).

    The ``log_pdf`` matches :func:`scipy.stats.multivariate_normal.logpdf`
    to roundoff under the T-005 float64 policy.
    """

    means: jnp.ndarray
    cov: jnp.ndarray
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        means = jnp.asarray(self.means)
        cov = jnp.asarray(self.cov)
        if means.ndim != 1:
            raise ValueError(
                f"MultivariateNormal: means must be 1D; got shape {means.shape}."
            )
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f"MultivariateNormal: cov must be square 2D; got shape {cov.shape}."
            )
        if cov.shape[0] != means.shape[0]:
            raise ValueError(
                f"MultivariateNormal: cov shape {cov.shape} must match means "
                f"length {means.shape[0]}."
            )
        _validate_kind(self.kind)
        # Stash the canonicalised arrays so ``sample`` / ``log_pdf`` see
        # jnp arrays even when the user passed lists / numpy arrays.
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "cov", cov)

    @property
    def n_dim(self) -> int:
        return int(self.means.shape[0])

    def sample(self, key, shape) -> jnp.ndarray:
        """Draw samples of shape ``(*shape, n_dim)`` from ``N(means, cov)``.

        Implemented as ``means + L @ z`` where ``L`` is the lower-Cholesky
        factor of ``cov`` and ``z`` is iid standard normal — this is the
        reparameterisation that keeps gradients through ``cov`` finite.
        """
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        n = self.n_dim
        L = jnp.linalg.cholesky(self.cov)
        z = jax.random.normal(key, shape=shape + (n,))
        # x = means + z @ L.T  (so that Cov[x] = L L.T = cov).
        return self.means + z @ L.T

    def ppf(self, u) -> jnp.ndarray:  # pragma: no cover
        """Not implemented for multivariate distributions.

        A multivariate inverse-CDF requires a choice of ordering and is not
        what the Sobol/Morris pipelines consume.  Raise to flag misuse.
        """
        raise NotImplementedError(
            "MultivariateNormal has no scalar inverse-CDF; use "
            "CorrelatedMarginals for copula-style transforms."
        )

    def log_pdf(self, x) -> jnp.ndarray:
        """Log multivariate-normal density at ``x`` (shape ``(..., n_dim)``)."""
        x = jnp.asarray(x)
        n = self.n_dim
        diff = x - self.means
        L = jnp.linalg.cholesky(self.cov)
        # Solve L y = diff; quadform = ||y||^2 = diff^T cov^{-1} diff.
        # jax.scipy.linalg.solve_triangular supports batched RHS along axis -1.
        # We solve along the last axis: treat diff as ``(..., n)`` -> reshape
        # to 2D for the triangular solve.
        flat = diff.reshape((-1, n))
        y = jax.scipy.linalg.solve_triangular(L, flat.T, lower=True).T
        quad = jnp.sum(y * y, axis=-1).reshape(diff.shape[:-1])
        # log|cov| = 2 * sum(log(diag(L))).
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * (quad + log_det + n * jnp.log(2.0 * jnp.pi))

    def cdf(self, x, n_samples: int = 4096) -> jnp.ndarray:
        """Multivariate-normal CDF ``P(X_1 <= x_1, ..., X_n <= x_n)``.

        There is no closed form for the joint multivariate-normal CDF.
        This implements the **Genz (1992)** separation-of-variables
        transform followed by a quasi-Monte-Carlo quadrature:

        1. Cholesky-factor ``cov = L L^T`` (``L`` lower-triangular).
        2. Shift the integration region by ``b = x - means``.
        3. Genz's recursive variable elimination turns the
           ``n``-dimensional orthant integral into an iterated integral
           over the unit cube ``(0, 1)^{n-1}``: dimension ``i``'s bound
           is a function of the already-sampled coordinates ``y_1..y_{i-1}``,
           evaluated through the standard-normal CDF ``Phi`` and its
           inverse ``Phi^{-1}``.
        4. Average the integrand ``f_n`` over ``n_samples`` Sobol QMC
           points (``sobol_sequence`` from :mod:`jaxonomy.uq.quasi_mc`).

        The estimate is a smooth function of the sampled coordinates, so
        :func:`jax.grad` flows cleanly through both ``x`` and ``means``
        (the QMC points themselves are constants w.r.t. the gradient).

        Args:
            x: Upper integration bound, shape ``(n_dim,)``.
            n_samples: Number of QMC points (power of 2 preferred for
                Sobol uniformity).  More points -> tighter estimate;
                QMC error falls roughly as ``O(1/n_samples)``.

        Returns:
            Scalar ``jnp.ndarray`` — the estimated joint CDF in ``[0, 1]``.

        Reference:
            Genz, A. (1992). "Numerical computation of multivariate
            normal probabilities." J. Comput. Graph. Stat. 1, 141-149.
        """
        # Local import to avoid a circular import at module load time
        # (quasi_mc imports Distribution from this module).
        from .quasi_mc import sobol_sequence

        x = jnp.asarray(x)
        if x.shape != (self.n_dim,):
            raise ValueError(
                f"MultivariateNormal.cdf: x must have shape ({self.n_dim},); "
                f"got {x.shape}."
            )
        n = self.n_dim
        L = jnp.linalg.cholesky(self.cov)
        b = x - self.means

        # Standard-normal CDF / inverse-CDF helpers.
        _SQRT2 = jnp.sqrt(jnp.asarray(2.0))

        def _phi(z):  # Phi(z)
            return 0.5 * (1.0 + jax.scipy.special.erf(z / _SQRT2))

        def _phi_inv(p):  # Phi^{-1}(p)
            # Clamp away from the open boundaries so erfinv stays finite.
            p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
            return _SQRT2 * jax.scipy.special.erfinv(2.0 * p - 1.0)

        # n == 1 degenerates to the univariate normal CDF — no QMC needed.
        if n == 1:
            return _phi(b[0] / L[0, 0])

        # QMC points fill the (n-1) "free" integration dimensions; the
        # first dimension of the Genz recursion is deterministic.
        w = sobol_sequence(n_samples, n - 1, seed=0, scramble=True)

        def _genz_one(w_row):
            """Genz integrand f_n for a single QMC point w_row in (0,1)^{n-1}."""
            e0 = _phi(b[0] / L[0, 0])
            f = e0
            # y holds the transformed coordinates y_1..y_{i-1}; pre-fill
            # with zeros so the unfilled tail contributes nothing to the
            # running dot product L[i, :i] @ y[:i].
            y = jnp.zeros(n)
            e_prev = e0
            for i in range(1, n):
                y = y.at[i - 1].set(_phi_inv(w_row[i - 1] * e_prev))
                # sum_{j<i} L[i, j] * y[j]
                s = jnp.dot(L[i, :i], y[:i])
                e_i = _phi((b[i] - s) / L[i, i])
                f = f * e_i
                e_prev = e_i
            return f

        f_vals = jax.vmap(_genz_one)(w)
        return jnp.mean(f_vals)


@dataclasses.dataclass(frozen=True)
class CorrelatedMarginals:
    """Gaussian-copula joint distribution with arbitrary marginals.

    Given a list of univariate ``marginals`` (each exposing ``ppf``) and
    an ``n x n`` correlation matrix ``corr_matrix`` (diagonal 1, off-diagonal
    in ``[-1, 1]``, symmetric positive-definite), :meth:`sample` draws

        1. ``z ~ N(0, corr_matrix)`` via Cholesky reparameterisation,
        2. ``u = Phi(z)`` (componentwise standard-normal CDF),
        3. ``x_i = marginals[i].ppf(u_i)``.

    This preserves Spearman (rank) correlation exactly and approximates
    Pearson correlation when the marginals are close to Gaussian.  For
    arbitrary marginals, the induced Pearson correlation is biased by the
    nonlinear inverse-CDF transform; that bias is the cost of working
    with non-Gaussian copulas in closed form.

    ``log_pdf`` is non-trivial under arbitrary inverse-CDF transforms
    (requires the copula density and per-marginal density Jacobians).
    We expose it but raise :class:`NotImplementedError` — a numerical
    Gaussian-copula density is left as a deeper followup.

    Differentiability: gradients flow through the Cholesky factor of
    ``corr_matrix`` and through each marginal's ``ppf`` (continuous
    distributions only — discrete marginals such as :class:`Poisson`
    have no ``ppf`` and are rejected at construction).
    """

    marginals: Sequence
    corr_matrix: jnp.ndarray
    # Keyword-only: a positional third/fourth argument silently landing in
    # `kind` is an easy mistake in dict-comprehension construction of mixed
    # aleatoric/epistemic distribution sets (T-130).
    kind: DistributionKind = dataclasses.field(default="aleatoric", kw_only=True)

    def __post_init__(self) -> None:
        corr = jnp.asarray(self.corr_matrix)
        n = len(self.marginals)
        if corr.ndim != 2 or corr.shape != (n, n):
            raise ValueError(
                f"CorrelatedMarginals: corr_matrix must be {n}x{n}; "
                f"got shape {corr.shape}."
            )
        # Reject marginals without a smooth inverse-CDF (e.g. Poisson).
        for i, m in enumerate(self.marginals):
            if not hasattr(m, "ppf"):
                raise ValueError(
                    f"CorrelatedMarginals: marginal {i} ({type(m).__name__}) "
                    "lacks a 'ppf' method; copula transform requires it."
                )
        _validate_kind(self.kind)
        object.__setattr__(self, "corr_matrix", corr)
        # Materialise the tuple form to keep the dataclass immutable.
        object.__setattr__(self, "marginals", tuple(self.marginals))

    @property
    def n_dim(self) -> int:
        return len(self.marginals)

    def sample(self, key, shape) -> jnp.ndarray:
        """Draw samples of shape ``(*shape, n_dim)`` via the Gaussian copula."""
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        n = self.n_dim
        L = jnp.linalg.cholesky(self.corr_matrix)
        z_std = jax.random.normal(key, shape=shape + (n,))
        z = z_std @ L.T  # z ~ N(0, corr_matrix)
        # Componentwise standard-normal CDF -> uniforms in (0, 1).
        u = jax.scipy.stats.norm.cdf(z)
        # Clamp to keep heavy-tailed ppfs finite at the open boundaries.
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        # Apply each marginal's ppf along the last axis.  We stack the
        # per-component transforms — the marginals list is a Python
        # sequence so we cannot vmap over it directly.
        components = [self.marginals[i].ppf(u[..., i]) for i in range(n)]
        return jnp.stack(components, axis=-1)

    def log_pdf(self, x) -> jnp.ndarray:  # pragma: no cover
        """Joint log-density under the Gaussian copula transform.

        Left as a deeper followup: requires the copula density
        ``c(u) = phi_R(Phi^{-1}(u)) / prod phi(Phi^{-1}(u_i))`` plus
        each marginal's ``log_pdf``.  Numerically delicate at the
        boundaries; ship after the copula machinery is wired into the
        importance-sampling pipeline.
        """
        raise NotImplementedError(
            "CorrelatedMarginals.log_pdf is a deeper followup "
            "(Gaussian copula density + per-marginal Jacobians)."
        )

    def cdf(self, x) -> jnp.ndarray:  # pragma: no cover
        """Joint CDF under the Gaussian copula — deferred to a deeper followup.

        The joint CDF
        ``F(x_1, ..., x_n) = Phi_R(Phi^{-1}(F_1(x_1)), ..., Phi^{-1}(F_n(x_n)))``
        requires the multivariate-normal CDF ``Phi_R`` under the copula
        correlation matrix, which (per :meth:`MultivariateNormal.cdf`) has
        no closed form and needs Genz's quasi-MC algorithm.  Track
        separately as T-122-followup-mvn-cdf.
        """
        raise NotImplementedError(
            "CorrelatedMarginals.cdf is a deeper followup (depends on the "
            "multivariate-normal CDF; see MultivariateNormal.cdf)."
        )
