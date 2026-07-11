# SPDX-License-Identifier: MIT

"""Aleatoric vs epistemic variance decomposition (T-126 phase 1 + followup).

Two flavours of uncertainty matter for a UQ workflow:

* **Aleatoric** — irreducible randomness inherent to the system: measurement
  noise, manufacturing variation, environmental disturbances. Pulling more
  data does not shrink it.
* **Epistemic** — reducible uncertainty from limited knowledge: an unknown
  but fixed parameter that more measurements would pin down.

Distributions in :mod:`jaxonomy.uq.distributions` carry a ``kind`` attribute
(``"aleatoric"`` or ``"epistemic"``) so the user can declare each parameter's
nature. The helpers in this module consume that tag.

Public surface:

* :func:`split_distributions_by_kind` — partition a ``{name: Distribution}``
  mapping into the aleatoric and epistemic subsets.
* :func:`monte_carlo_with_kinds` — IID-sample every distribution and return
  both the per-parameter sample dict and a sibling ``kind_labels`` mapping.
* :func:`decompose_variance` — *first-order / Taylor* decomposition (T-126
  phase 1). Cheap, exact for linear QoIs, biased on strongly nonlinear ones.
* :func:`decompose_variance_sobol` — *formal* Sobol-style ANOVA grouped
  decomposition (T-126 followup). Uses the Saltelli/Jansen pair-difference
  estimators applied at the *group* level (aleatoric vs epistemic) to
  recover ``V[Y] = V_aleatoric + V_epistemic + interaction`` without the
  small-uncertainty approximation.
* :func:`mean_and_variance_by_kind` — honest-fallback reporter that does not
  attempt a formal decomposition; just reports per-partition stats.
* :func:`vmap_qoi` — wrap a single-sample ``qoi_fn(params: dict) -> scalar``
  so it accepts a dict of batched arrays and returns ``(N,)``. Lets UQ
  callers write a natural per-sample callable and pass it directly to
  :func:`decompose_variance_sobol`, :func:`sobol_indices`, and
  :func:`morris_screening` without manually broadcasting (T-126 followup).
* :func:`conditional_monte_carlo` — IID-sample, evaluate the QoI, and filter
  by a user-supplied predicate (T-126 followup). The filtered samples can be
  fed straight back into :func:`decompose_variance` for a conditional-on-event
  variance decomposition.
* :func:`importance_sample` — draw from a *proposal* distribution and reweight
  to a *target* distribution via Radon-Nikodym log-density ratios. Standard
  variance-reduction trick when the tail of interest is far from the bulk
  (T-126 followup).
* :func:`quantile_summary` — common quantile-based summary stats
  (q05/q50/q95 + mean + std) for Monte Carlo output (T-126 followup).
* :func:`value_at_risk` / :func:`conditional_value_at_risk` — risk-summary
  one-sided alpha-quantile (VaR) and tail-expectation (CVaR / Expected
  Shortfall) helpers for Monte Carlo output (T-126 followup).
"""

from __future__ import annotations

from typing import Callable, Mapping

import jax
import jax.numpy as jnp

from jaxonomy.backend import numpy_api as npa

from .distributions import Distribution, DistributionKind
from .sampling import sample_parameters
from .sobol import saltelli_sample

__all__ = [
    "split_distributions_by_kind",
    "monte_carlo_with_kinds",
    "decompose_variance",
    "decompose_variance_sobol",
    "mean_and_variance_by_kind",
    "vmap_qoi",
    "conditional_monte_carlo",
    "importance_sample",
    "quantile_summary",
    "value_at_risk",
    "conditional_value_at_risk",
]


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

def split_distributions_by_kind(
    distributions: Mapping[str, Distribution],
) -> dict[str, dict[str, Distribution]]:
    """Partition ``distributions`` by their ``kind`` attribute.

    Args:
        distributions: ``{param_name: Distribution}`` mapping.

    Returns:
        ``{"aleatoric": {...}, "epistemic": {...}}``. Either inner dict may be
        empty (e.g. if every parameter is aleatoric, the epistemic dict is
        ``{}``).
    """
    out: dict[str, dict[str, Distribution]] = {"aleatoric": {}, "epistemic": {}}
    for name, dist in distributions.items():
        kind = getattr(dist, "kind", "aleatoric")
        if kind not in out:
            raise ValueError(
                f"split_distributions_by_kind: distribution {name!r} has "
                f"unrecognised kind={kind!r}; expected 'aleatoric' or "
                "'epistemic'."
            )
        out[kind][name] = dist
    return out


def monte_carlo_with_kinds(
    distributions: Mapping[str, Distribution],
    n_samples: int,
    key,
) -> tuple[dict[str, jnp.ndarray], dict[str, DistributionKind]]:
    """IID-sample each distribution and return samples plus per-name labels.

    A thin convenience wrapper over :func:`jaxonomy.uq.sample_parameters`
    that also surfaces a ``{name: kind}`` mapping so downstream variance
    decomposition can split parameters without re-querying each
    distribution.

    Args:
        distributions: ``{param_name: Distribution}`` mapping. Each
            distribution must expose a ``kind`` attribute (default
            ``"aleatoric"`` for the shipped types).
        n_samples: Number of IID samples per parameter.
        key: ``jax.random.PRNGKey``. A fresh subkey is split per parameter
            so the marginals are independent.

    Returns:
        ``(samples, kind_labels)`` where ``samples[name]`` is an ``(n_samples,)``
        array and ``kind_labels[name]`` is the parameter's kind string.
    """
    if n_samples <= 0:
        raise ValueError(
            f"monte_carlo_with_kinds: n_samples must be > 0, got {n_samples}"
        )
    if not distributions:
        raise ValueError(
            "monte_carlo_with_kinds: distributions dict must be non-empty."
        )

    keys = jax.random.split(key, len(distributions))
    samples: dict[str, jnp.ndarray] = {}
    kind_labels: dict[str, DistributionKind] = {}
    for k, (name, dist) in zip(keys, distributions.items()):
        samples[name] = dist.sample(k, (n_samples,))
        kind_labels[name] = getattr(dist, "kind", "aleatoric")
    return samples, kind_labels


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def decompose_variance(
    qoi_samples: jnp.ndarray,
    parameter_samples: Mapping[str, jnp.ndarray],
    kind_labels: Mapping[str, DistributionKind],
) -> dict[str, float]:
    """First-order aleatoric/epistemic variance decomposition.

    Implements the small-uncertainty (linearised / Taylor) decomposition:

    .. math::
        \\mathrm{Var}(y) \\approx \\sum_i (\\partial_i y)^2 \\, \\mathrm{Var}(\\theta_i)

    where the partial derivatives are estimated as the sample-covariance
    sensitivity ``Cov(theta_i, y) / Var(theta_i)``. The aleatoric component
    is the sum of first-order contributions over parameters tagged
    ``"aleatoric"``; the epistemic component is the same sum over parameters
    tagged ``"epistemic"``. Higher-order interactions (the gap between the
    linearised sum and the empirical total) land in a ``"residual"`` field
    so users can see when the linear approximation is breaking down.

    Args:
        qoi_samples: ``(n_samples,)`` array of the scalar quantity-of-interest
            evaluated at each parameter draw.
        parameter_samples: ``{name: (n_samples,) array}`` of the parameter
            draws used to evaluate ``qoi_samples``. Names must match
            ``kind_labels``.
        kind_labels: ``{name: "aleatoric" | "epistemic"}`` mapping. Names
            without an entry default to ``"aleatoric"``.

    Returns:
        Dict with float fields ``var_total``, ``var_aleatoric``,
        ``var_epistemic``, and ``residual`` (= ``var_total -
        var_aleatoric - var_epistemic``).
    """
    qoi = npa.asarray(qoi_samples)
    if qoi.ndim != 1:
        raise ValueError(
            "decompose_variance: qoi_samples must be 1-D (n_samples,); "
            f"got shape {qoi.shape}."
        )
    if not parameter_samples:
        raise ValueError(
            "decompose_variance: parameter_samples must be non-empty."
        )

    n = int(qoi.shape[0])
    var_total = float(npa.var(qoi))

    var_aleatoric = 0.0
    var_epistemic = 0.0
    qoi_mean = float(npa.mean(qoi))
    for name, theta in parameter_samples.items():
        theta = npa.asarray(theta)
        if theta.shape != (n,):
            raise ValueError(
                f"decompose_variance: parameter {name!r} has shape "
                f"{theta.shape}; expected ({n},) to match qoi_samples."
            )
        var_theta = float(npa.var(theta))
        if var_theta <= 0.0:
            # Degenerate (constant) parameter contributes nothing.
            continue
        theta_mean = float(npa.mean(theta))
        cov = float(npa.mean((theta - theta_mean) * (qoi - qoi_mean)))
        # First-order sensitivity (regression slope) and its first-order
        # variance contribution. Equivalent to the Sobol first-order index
        # in the linear / small-uncertainty limit.
        slope = cov / var_theta
        contribution = (slope ** 2) * var_theta
        kind = kind_labels.get(name, "aleatoric")
        if kind == "aleatoric":
            var_aleatoric += contribution
        elif kind == "epistemic":
            var_epistemic += contribution
        else:
            raise ValueError(
                f"decompose_variance: parameter {name!r} has unrecognised "
                f"kind={kind!r}; expected 'aleatoric' or 'epistemic'."
            )

    residual = var_total - var_aleatoric - var_epistemic
    return {
        "var_total": var_total,
        "var_aleatoric": var_aleatoric,
        "var_epistemic": var_epistemic,
        "residual": residual,
    }


def decompose_variance_sobol(
    qoi_fn: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray],
    aleatoric_dists: Mapping[str, Distribution],
    epistemic_dists: Mapping[str, Distribution],
    n_samples: int = 1024,
    key=None,
) -> dict[str, float]:
    """Formal Sobol-style ANOVA decomposition for two parameter groups.

    Computes the *grouped* first-order Sobol indices for the aleatoric and
    epistemic parameter sets without the linearised / small-uncertainty
    approximation that :func:`decompose_variance` uses. The decomposition is
    the standard ANOVA split:

    .. math::
        \\mathrm{Var}(Y) = V_{\\text{aleatoric}} + V_{\\text{epistemic}}
                          + I_{\\text{aleatoric}\\times\\text{epistemic}}

    where

    * ``V_aleatoric = Var_a[E_e[Y | X_a]]`` — the variance explained by the
      aleatoric group alone (i.e. the first-order Sobol index for the
      aleatoric group, multiplied by ``Var(Y)``).
    * ``V_epistemic = Var_e[E_a[Y | X_e]]`` — symmetric definition for the
      epistemic group.
    * ``interaction = Var(Y) - V_aleatoric - V_epistemic`` — the residual
      explained jointly by aleatoric*epistemic interactions (zero for
      additively separable QoIs).

    Estimator
    ---------
    Reuses :func:`jaxonomy.uq.sobol.saltelli_sample` to draw two independent
    parameter matrices ``A`` and ``B``. The *grouped* Saltelli design then
    builds two extra matrices

    * ``AB_a``: take ``A`` and replace every aleatoric column with the
      corresponding column of ``B``;
    * ``AB_e``: take ``A`` and replace every epistemic column.

    The Jansen pair-difference estimators give

    .. math::
        V_{\\text{aleatoric}} \\approx V[Y] - \\tfrac{1}{2}
        \\mathrm{mean}\\bigl((y_B - y_{AB_a})^2\\bigr),

        V_{\\text{epistemic}} \\approx V[Y] - \\tfrac{1}{2}
        \\mathrm{mean}\\bigl((y_B - y_{AB_e})^2\\bigr).

    Compute cost
    ------------
    Total ``qoi_fn`` evaluations: ``4 * n_samples`` (matrices ``A``, ``B``,
    ``AB_a``, ``AB_e``). At the default ``n_samples=1024`` that is 4096
    evaluations regardless of the number of parameters in each group —
    exactly the same cost as a 2-parameter Sobol run, because the indices
    are *grouped*, not per-parameter. The four matrices are stacked into a
    single ``qoi_fn`` call so the JIT-compile cost is paid once.

    A pedagogical "double-loop" reference implementation
    (sample epistemic, then for each draw sample aleatoric and compute the
    inner conditional variance) is *not* shipped — the Saltelli/Jansen path
    is fast enough that no fallback is needed. A future follow-up could add
    higher-order grouped indices (three-way splits, screening interactions
    inside each group) but the two-group ANOVA covers the aleatoric vs
    epistemic separation called out in the T-126 task statement.

    Args:
        qoi_fn: Callable taking a dict ``{name: (N,) array}`` and returning a
            scalar QoI of shape ``(N,)``. **Contract:** the dict holds the
            *union* of aleatoric and epistemic parameter names — every key,
            even when one group has a single parameter — and ``qoi_fn`` must
            index it by name (``params["k"]``), not by position or by
            assuming only one group's keys are present. A ``qoi_fn`` written
            for one group only typically fails with a ``KeyError`` or a
            broadcasting error; the wrapper re-raises those with the key list
            attached so the mismatch is attributable at this boundary.
        aleatoric_dists: Mapping ``name -> Distribution`` for the aleatoric
            group. May be empty (then ``var_aleatoric`` is reported as
            ``0.0`` and the decomposition collapses to ``var_epistemic ==
            var_total``).
        epistemic_dists: Mapping ``name -> Distribution`` for the epistemic
            group. May be empty (symmetric).
        n_samples: Base sample count ``N``; total evaluations ``4 * N``.
            Default 1024 (matches ``sobol_indices``).
        key: ``jax.random.PRNGKey``. Defaults to ``PRNGKey(0)`` for
            reproducibility.

    Returns:
        Dict with float fields

        * ``var_total`` — empirical ``Var(Y)`` over the pooled ``A``/``B``
          samples.
        * ``var_aleatoric`` — first-order grouped variance of the aleatoric
          set (clipped to ``[0, var_total]`` to absorb MC negativity at low
          ``N`` for nearly-flat directions).
        * ``var_epistemic`` — symmetric for the epistemic set.
        * ``interaction`` — ``var_total - var_aleatoric - var_epistemic``.

    Raises:
        ValueError: If both groups are empty, if any distribution name is
            shared across the two groups, or if ``n_samples <= 0``.
    """
    if n_samples <= 0:
        raise ValueError(
            f"decompose_variance_sobol: n_samples must be > 0, got {n_samples}"
        )
    if not aleatoric_dists and not epistemic_dists:
        raise ValueError(
            "decompose_variance_sobol: at least one of aleatoric_dists / "
            "epistemic_dists must be non-empty."
        )
    overlap = set(aleatoric_dists).intersection(epistemic_dists)
    if overlap:
        raise ValueError(
            f"decompose_variance_sobol: parameter name(s) {sorted(overlap)} "
            "appear in both aleatoric_dists and epistemic_dists; each "
            "parameter must belong to exactly one group."
        )
    if key is None:
        key = jax.random.PRNGKey(0)

    aleatoric_names = list(aleatoric_dists.keys())
    epistemic_names = list(epistemic_dists.keys())
    all_dists: dict[str, Distribution] = {}
    all_dists.update(aleatoric_dists)
    all_dists.update(epistemic_dists)

    # We only need A and B (Saltelli's per-parameter AB_list is overkill for
    # a 2-group split). Reuse `saltelli_sample` for the A/B draw and then
    # build the two grouped AB matrices ourselves.
    A, B, _ab_per_param = saltelli_sample(all_dists, n_samples, key)

    # AB_a: A with aleatoric columns swapped to B.
    AB_a = dict(A)
    for name in aleatoric_names:
        AB_a[name] = B[name]
    # AB_e: A with epistemic columns swapped to B.
    AB_e = dict(A)
    for name in epistemic_names:
        AB_e[name] = B[name]

    # Stack the four matrices and call qoi_fn once so the kernel-path JIT
    # compile is amortised across all 4*N evaluations.
    N = n_samples
    stacked = {
        name: jnp.concatenate([A[name], B[name], AB_a[name], AB_e[name]])
        for name in all_dists
    }
    try:
        y = jnp.asarray(qoi_fn(stacked))
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            "decompose_variance_sobol: qoi_fn raised while evaluating the "
            f"stacked sample dict (keys {sorted(stacked)}, each of shape "
            f"({4 * N},)). qoi_fn must consume the union of aleatoric and "
            "epistemic parameter names by key — a qoi_fn written for only "
            "one group's keys (or assuming positional order) is the usual "
            "cause."
        ) from err
    if y.shape != (4 * N,):
        raise ValueError(
            "decompose_variance_sobol: qoi_fn returned shape "
            f"{y.shape}; expected ({4 * N},). qoi_fn must reduce each "
            "sample row to one scalar while consuming every parameter in "
            f"{sorted(stacked)}."
        )
    yA = y[:N]
    yB = y[N:2 * N]
    yAB_a = y[2 * N:3 * N]
    yAB_e = y[3 * N:4 * N]

    var_total = float(jnp.var(jnp.concatenate([yA, yB])))

    if aleatoric_names:
        var_aleatoric = var_total - 0.5 * float(jnp.mean((yB - yAB_a) ** 2))
    else:
        var_aleatoric = 0.0
    if epistemic_names:
        var_epistemic = var_total - 0.5 * float(jnp.mean((yB - yAB_e) ** 2))
    else:
        var_epistemic = 0.0

    # Clip MC negativity (small-N noise on near-zero indices). Cap each
    # contribution at var_total so the interaction term, computed by
    # subtraction, stays bounded.
    var_aleatoric = max(0.0, min(var_aleatoric, var_total))
    var_epistemic = max(0.0, min(var_epistemic, var_total))

    interaction = var_total - var_aleatoric - var_epistemic
    return {
        "var_total": var_total,
        "var_aleatoric": var_aleatoric,
        "var_epistemic": var_epistemic,
        "interaction": interaction,
    }


def mean_and_variance_by_kind(
    parameter_samples: Mapping[str, jnp.ndarray],
    kind_labels: Mapping[str, DistributionKind],
) -> dict[str, dict[str, float]]:
    """Honest-fallback reporter: per-kind mean and variance of *parameters*.

    Splits parameters by ``kind`` and reports the mean and variance of each
    partition's joint sample tensor (stacked per-parameter). Useful when the
    formal QoI-variance decomposition (see :func:`decompose_variance`) is
    overkill and the user just wants to see how big each uncertainty source
    is in parameter space.

    Args:
        parameter_samples: ``{name: (n_samples,) array}`` mapping.
        kind_labels: ``{name: "aleatoric" | "epistemic"}`` mapping.

    Returns:
        ``{"aleatoric": {"mean": ..., "var": ...}, "epistemic": {...}}``.
        A partition with no parameters reports ``{"mean": 0.0, "var": 0.0}``.
    """
    out: dict[str, dict[str, float]] = {
        "aleatoric": {"mean": 0.0, "var": 0.0},
        "epistemic": {"mean": 0.0, "var": 0.0},
    }
    for kind in ("aleatoric", "epistemic"):
        names = [n for n, k in kind_labels.items() if k == kind]
        if not names:
            continue
        stacked = npa.concatenate([npa.asarray(parameter_samples[n]) for n in names])
        out[kind]["mean"] = float(npa.mean(stacked))
        out[kind]["var"] = float(npa.var(stacked))
    return out


# ---------------------------------------------------------------------------
# vmap wrapper helper (T-126 followup)
# ---------------------------------------------------------------------------

def vmap_qoi(
    qoi_fn: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray],
    in_axes: int = 0,
) -> Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]:
    """Wrap a per-sample ``qoi_fn`` so it accepts a dict of batched arrays.

    The UQ entry points :func:`decompose_variance_sobol`,
    :func:`jaxonomy.uq.sobol_indices`, and
    :func:`jaxonomy.uq.morris_screening` all take a callable that consumes a
    ``{name: (N,) array}`` dict and returns a ``(N,)`` array — i.e. the user
    is expected to write the per-sample math in already-batched form. That
    is fine for trivially elementwise QoIs but cumbersome when the per-sample
    computation involves indexing, conditionals, or calls to library blocks
    that only accept scalar inputs.

    ``vmap_qoi`` removes the friction. Pass a natural per-sample callable

    .. code-block:: python

        def qoi(params: dict[str, float]) -> float:
            return params["a"] * params["x"] + params["b"]

        batched = vmap_qoi(qoi)
        decompose_variance_sobol(batched, aleatoric_dists=..., epistemic_dists=...)

    and ``batched`` will accept ``{"a": (N,), "x": (N,), "b": (N,)}`` and
    return ``(N,)`` via :func:`jax.vmap`.

    The wrapped callable also validates the per-sample QoI's output once on
    a small probe so the failure mode (wrong shape) surfaces with a clear
    error early instead of as an opaque shape mismatch deep inside a
    Saltelli matrix swap.

    Args:
        qoi_fn: Per-sample callable taking a ``{name: scalar (or shape ())}``
            dict and returning a scalar (shape ``()``). Anything that
            :func:`jax.vmap` can map over works — including pytrees of
            scalars on the input, but the output must reduce to a scalar.
        in_axes: Forwarded to :func:`jax.vmap`. Default ``0`` (vmap over
            the leading axis of every dict value). Pass other ints or a
            pytree of axis specs for advanced layouts.

    Returns:
        Batched callable suitable for the UQ entry points: takes a
        ``{name: (N,) array}`` dict, returns a ``(N,)`` array.

    Raises:
        ValueError: When the user-supplied ``qoi_fn`` returns a non-scalar
            output on the probe call (typed as a dict of zero scalars).

    Notes:
        Bit-equivalent to writing ``jax.vmap(qoi_fn, in_axes=({name: 0,
        ...},))`` by hand. The wrapper exists so users do not have to
        construct the per-name in_axes pytree themselves and so the
        shape-mismatch error message points at *their* QoI rather than at
        the Saltelli plumbing.
    """
    if not callable(qoi_fn):
        raise TypeError(
            f"vmap_qoi: qoi_fn must be callable; got {type(qoi_fn).__name__}."
        )

    batched = jax.vmap(qoi_fn, in_axes=(in_axes,))

    def wrapped(params: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        # Convert input dict values to arrays once (so users may pass plain
        # python lists / numpy arrays for ad-hoc calls). Preserve dict type
        # for downstream consumers that key off it.
        as_arrays = {name: jnp.asarray(v) for name, v in params.items()}
        if not as_arrays:
            raise ValueError("vmap_qoi: params dict must be non-empty.")
        # Sanity-check: every leaf must share the leading sample axis.
        sample_lens = {name: arr.shape[0] if arr.ndim > 0 else None
                       for name, arr in as_arrays.items()}
        first_n = next(iter(sample_lens.values()))
        if first_n is None:
            raise ValueError(
                "vmap_qoi: all params must have at least one axis (the "
                "sample axis); got a 0-D scalar."
            )
        for name, n in sample_lens.items():
            if n != first_n:
                raise ValueError(
                    "vmap_qoi: parameter arrays must share the leading "
                    f"sample axis; got {name!r} with shape "
                    f"{as_arrays[name].shape}, expected leading dim "
                    f"{first_n}."
                )
        out = batched(as_arrays)
        out = jnp.asarray(out)
        if out.shape != (first_n,):
            raise ValueError(
                "vmap_qoi: wrapped qoi_fn returned shape "
                f"{out.shape}; expected ({first_n},). Make sure the "
                "per-sample qoi_fn returns a scalar (shape ())."
            )
        return out

    return wrapped


# ---------------------------------------------------------------------------
# Conditional Monte Carlo / importance sampling (T-126 followup)
# ---------------------------------------------------------------------------

def conditional_monte_carlo(
    qoi_fn: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray],
    dists: Mapping[str, Distribution],
    condition_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_samples: int,
    key=None,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Filter Monte Carlo samples by a user-supplied predicate on the QoI.

    Standard "conditional Monte Carlo" rare-event analysis: draw ``n_samples``
    IID parameter vectors from ``dists``, evaluate ``qoi_fn`` once over the
    batch, then keep only those samples whose QoI satisfies ``condition_fn``.
    Useful for tail-risk analysis (e.g. "show me every parameter draw whose
    output exceeded the 99th percentile threshold") without spending
    importance-sampling effort on the non-tail bulk.

    The conditional probability ``P(condition)`` is the ratio of returned
    samples to ``n_samples`` (caller can compute it from the lengths). For
    rare events at fixed ``n_samples`` this estimator has high relative
    variance — use :func:`importance_sample` for ``P(condition) << 1``.

    Args:
        qoi_fn: Batched callable taking ``{name: (N,) array}`` and returning a
            ``(N,)`` array. Wrap a per-sample callable with :func:`vmap_qoi`
            if you only have a scalar form.
        dists: ``{name: Distribution}`` mapping over which to sample.
        condition_fn: Callable taking the QoI ``(N,)`` array and returning a
            boolean ``(N,)`` mask. Samples where the mask is True are kept.
        n_samples: Number of base IID draws (before filtering).
        key: ``jax.random.PRNGKey``. Defaults to ``PRNGKey(0)``.

    Returns:
        ``(filtered_samples, filtered_qoi)`` where ``filtered_samples[name]``
        is a ``(K,)`` array (``K = mask.sum()``, may be 0) and
        ``filtered_qoi`` is the matching ``(K,)`` QoI evaluation. The user
        can compute ``P(condition) = K / n_samples`` and apply
        :func:`decompose_variance` on the filtered tensors directly for a
        conditional-on-event variance decomposition.

    Raises:
        ValueError: If ``n_samples <= 0``, if ``dists`` is empty, or if
            ``condition_fn`` does not return a boolean array shaped ``(N,)``.
    """
    if n_samples <= 0:
        raise ValueError(
            f"conditional_monte_carlo: n_samples must be > 0, got {n_samples}"
        )
    if not dists:
        raise ValueError("conditional_monte_carlo: dists must be non-empty.")
    if key is None:
        key = jax.random.PRNGKey(0)

    samples = sample_parameters(dists, n_samples, key)
    qoi = jnp.asarray(qoi_fn(samples))
    if qoi.shape != (n_samples,):
        raise ValueError(
            "conditional_monte_carlo: qoi_fn returned shape "
            f"{qoi.shape}; expected ({n_samples},)."
        )
    mask = jnp.asarray(condition_fn(qoi))
    if mask.shape != (n_samples,):
        raise ValueError(
            "conditional_monte_carlo: condition_fn returned shape "
            f"{mask.shape}; expected ({n_samples},)."
        )
    if mask.dtype != jnp.bool_:
        # Accept anything that JAX treats as truthy but warn via cast — the
        # contract is "boolean predicate", an integer mask is a footgun.
        mask = mask.astype(jnp.bool_)

    # Boolean indexing materialises on host; the filtered tensors are no
    # longer fixed-shape so we cannot keep the JAX trace anyway.
    mask_np = npa.asarray(mask)
    filtered = {name: npa.asarray(arr)[mask_np] for name, arr in samples.items()}
    filtered_qoi = npa.asarray(qoi)[mask_np]
    return filtered, filtered_qoi


def importance_sample(
    qoi_fn: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray],
    target_dists: Mapping[str, Distribution],
    proposal_dists: Mapping[str, Distribution],
    n_samples: int,
    key=None,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Importance sampling: draw from ``proposal_dists``, reweight to ``target_dists``.

    Standard variance-reduction trick when the tail of interest under
    ``target_dists`` is far from its bulk. Sample from ``proposal_dists``
    (chosen so the tail is well covered) and apply Radon-Nikodym weights
    ``w_i = p_target(x_i) / p_proposal(x_i)`` so that ``E_target[f(X)] ≈
    sum(w_i f(x_i)) / n_samples`` and equivalently ``≈ sum(w_i f(x_i)) /
    sum(w_i)`` for the self-normalised estimator (preferred when only a
    proportional weight is reliable).

    The two distributions must share the same parameter names so weights
    factorise as ``prod_n p_target_n(x_n) / p_proposal_n(x_n)``. Computed in
    log-space and exponentiated at the end to avoid under/overflow for
    high-dimensional parameter spaces.

    Args:
        qoi_fn: Batched callable taking ``{name: (N,) array}`` and returning a
            ``(N,)`` array.
        target_dists: ``{name: Distribution}`` for the *target* measure (the
            distribution under which expectations are wanted).
        proposal_dists: ``{name: Distribution}`` for the *proposal* measure
            (from which we actually draw samples). Must have identical key
            set to ``target_dists``.
        n_samples: Number of IID proposal draws.
        key: ``jax.random.PRNGKey``. Defaults to ``PRNGKey(0)``.

    Returns:
        ``(samples, weights, qoi)`` where ``samples[name]`` is a ``(N,)``
        array drawn from ``proposal_dists``, ``weights`` is the ``(N,)``
        importance weight ``p_target / p_proposal``, and ``qoi`` is the
        ``(N,)`` QoI evaluation. Two consistent estimators of
        ``E_target[f(X)]`` are then ``mean(w * qoi)`` (unbiased) and
        ``sum(w * qoi) / sum(w)`` (self-normalised, lower variance for
        skewed weights). Tail-event probabilities use the indicator
        ``E_target[1{f(X) > t}] ≈ mean(w * (qoi > t))``.

    Raises:
        ValueError: If ``n_samples <= 0``, if either dist dict is empty, or
            if the two dicts disagree on parameter names.
    """
    if n_samples <= 0:
        raise ValueError(
            f"importance_sample: n_samples must be > 0, got {n_samples}"
        )
    if not target_dists or not proposal_dists:
        raise ValueError(
            "importance_sample: target_dists and proposal_dists must be non-empty."
        )
    if set(target_dists.keys()) != set(proposal_dists.keys()):
        missing_t = set(proposal_dists) - set(target_dists)
        missing_p = set(target_dists) - set(proposal_dists)
        raise ValueError(
            "importance_sample: target_dists and proposal_dists must share "
            f"the same parameter names; missing in target: {sorted(missing_t)}, "
            f"missing in proposal: {sorted(missing_p)}."
        )
    for name in target_dists:
        for label, dist in (("target", target_dists[name]),
                            ("proposal", proposal_dists[name])):
            if not hasattr(dist, "log_pdf"):
                raise ValueError(
                    f"importance_sample: {label} distribution {name!r} of "
                    f"type {type(dist).__name__} does not implement log_pdf; "
                    "add a log_pdf method or use conditional_monte_carlo."
                )
    if key is None:
        key = jax.random.PRNGKey(0)

    samples = sample_parameters(proposal_dists, n_samples, key)
    qoi = jnp.asarray(qoi_fn(samples))
    if qoi.shape != (n_samples,):
        raise ValueError(
            "importance_sample: qoi_fn returned shape "
            f"{qoi.shape}; expected ({n_samples},)."
        )

    # Sum log-densities across parameters (independence assumption matches
    # how `sample_parameters` draws each marginal independently).
    log_target = jnp.zeros((n_samples,))
    log_proposal = jnp.zeros((n_samples,))
    for name, x in samples.items():
        log_target = log_target + target_dists[name].log_pdf(x)
        log_proposal = log_proposal + proposal_dists[name].log_pdf(x)

    # Subtract the maximum for numerical stability before exponentiating;
    # the constant cancels in the self-normalised estimator and is harmless
    # for the unbiased estimator (we restore it explicitly so callers who
    # divide by `n_samples` get the right scale).
    log_w = log_target - log_proposal
    log_w_max = jnp.nanmax(log_w)
    # Guard the (degenerate) all-(-inf) case so we do not propagate NaN.
    log_w_max = jnp.where(jnp.isfinite(log_w_max), log_w_max, 0.0)
    weights = jnp.exp(log_w - log_w_max) * jnp.exp(log_w_max)

    return samples, weights, qoi


# ---------------------------------------------------------------------------
# Quantile-based summary statistics (T-126 followup)
# ---------------------------------------------------------------------------

def _quantile_label(q: float) -> str:
    """Format a quantile as the ``"qNN"`` key used by :func:`quantile_summary`.

    Rounds to two decimal digits (e.g. ``0.05 -> "q05"``, ``0.5 -> "q50"``,
    ``0.975 -> "q98"``). Two digits is enough for the standard risk-reporting
    levels (5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%) and keeps keys short.
    """
    pct = int(round(q * 100))
    return f"q{pct:02d}"


def quantile_summary(
    samples,
    quantiles=(0.05, 0.5, 0.95),
) -> dict[str, float | jnp.ndarray]:
    """Quantile-based summary statistics for a Monte Carlo output tensor.

    Standard risk-reporting summary for the output of an MC sweep: the
    requested quantiles plus the empirical mean and standard deviation.
    Returned as a flat dict so callers can plug results into structured
    logging, dataframes, or YAML without unpacking nested objects.

    Args:
        samples: ``(N,)`` 1-D array (or ``(N, ...)`` multi-output) of MC
            samples. Quantiles and statistics are computed along axis 0, so a
            ``(N, d)`` input returns ``(d,)`` arrays per field. Pass a 1-D
            array for scalar QoIs and floats are returned.
        quantiles: Iterable of probability levels in ``(0, 1)``. Defaults to
            ``(0.05, 0.5, 0.95)``. The returned dict keys are formatted as
            ``"qNN"`` (``q05``, ``q50``, ``q95``).

    Returns:
        Dict with one entry per requested quantile (key ``"qNN"``) plus
        ``"mean"`` and ``"std"``. Values are floats when ``samples`` is 1-D
        and ndarrays of shape ``samples.shape[1:]`` otherwise.

    Raises:
        ValueError: If ``samples`` is empty, any quantile is outside
            ``(0, 1)``, or duplicate quantiles produce a key collision.

    Notes:
        Performance: uses :func:`jnp.quantile` which sorts the input
        internally — ``O(N log N)``. For very large ``N`` under JIT, a
        sorted-array bisect fallback would be faster but the current
        implementation runs eagerly (called on already-materialised MC
        output), so the JIT-compile path is not a hot loop here. If a JIT
        caller needs to compute the summary inside a traced function, sort
        once with :func:`jnp.sort` and index by ``floor(q * (N - 1))`` to
        avoid recompiling :func:`jnp.quantile` for every shape.
    """
    arr = npa.asarray(samples)
    if arr.size == 0:
        raise ValueError("quantile_summary: samples must be non-empty.")
    if arr.ndim == 0:
        raise ValueError(
            "quantile_summary: samples must have at least one axis (the "
            "sample axis); got a 0-D scalar."
        )

    q_list = [float(q) for q in quantiles]
    for q in q_list:
        if not (0.0 < q < 1.0):
            raise ValueError(
                "quantile_summary: every quantile must lie in (0, 1); got "
                f"{q!r}."
            )
    labels = [_quantile_label(q) for q in q_list]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"quantile_summary: quantiles {q_list!r} collide on the qNN key "
            f"scheme: {labels!r}. Pick levels at least 1% apart."
        )

    out: dict[str, float | jnp.ndarray] = {}
    qs = npa.quantile(arr, npa.asarray(q_list), axis=0)
    for label, qv in zip(labels, qs):
        if arr.ndim == 1:
            out[label] = float(qv)
        else:
            out[label] = npa.asarray(qv)
    mean = npa.mean(arr, axis=0)
    std = npa.std(arr, axis=0)
    if arr.ndim == 1:
        out["mean"] = float(mean)
        out["std"] = float(std)
    else:
        out["mean"] = npa.asarray(mean)
        out["std"] = npa.asarray(std)
    return out


def value_at_risk(samples, alpha: float = 0.05) -> float:
    """One-sided alpha-quantile of a Monte Carlo output (Value-at-Risk).

    Standard risk-management VaR: the alpha-quantile of the loss distribution.
    With the convention that ``samples`` already encode the signed quantity
    of interest (positive = good, negative = loss), ``value_at_risk(samples,
    alpha=0.05)`` returns the 5th-percentile sample, i.e. the threshold below
    which only 5% of outcomes fall.

    Args:
        samples: ``(N,)`` 1-D array of MC samples.
        alpha: Probability level in ``(0, 1)``. Defaults to ``0.05`` (95%
            confidence VaR).

    Returns:
        Float value at the alpha-quantile.

    Raises:
        ValueError: If ``samples`` is empty or ``alpha`` is outside ``(0, 1)``.

    Notes:
        Matches the *left-tail* convention common in academic finance / safety
        literature (``VaR_alpha = -inf{x : P(X <= x) >= alpha}`` collapsed to
        the unsigned quantile). Practitioners who report VaR as a positive
        loss magnitude should negate the return value themselves; we stay in
        signed-sample space so :func:`conditional_value_at_risk` composes
        cleanly (CVaR is then the mean of the left tail without a sign flip).
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            "value_at_risk: alpha must lie in (0, 1); got "
            f"{alpha!r}."
        )
    arr = npa.asarray(samples)
    if arr.size == 0:
        raise ValueError("value_at_risk: samples must be non-empty.")
    if arr.ndim != 1:
        raise ValueError(
            "value_at_risk: samples must be 1-D (n_samples,); "
            f"got shape {arr.shape}."
        )
    return float(npa.quantile(arr, float(alpha)))


def conditional_value_at_risk(samples, alpha: float = 0.05) -> float:
    """Conditional Value-at-Risk (Expected Shortfall) of an MC output.

    Mean of the samples below the alpha-quantile threshold:

    .. math::
        \\mathrm{CVaR}_\\alpha = E[X \\mid X \\le \\mathrm{VaR}_\\alpha(X)].

    Captures the *severity* of the left tail beyond the VaR threshold, where
    VaR alone reports only the threshold itself. CVaR is coherent (subadditive,
    convex) whereas VaR is not, so it is the preferred risk measure for
    stress-testing.

    Args:
        samples: ``(N,)`` 1-D array of MC samples.
        alpha: Probability level in ``(0, 1)``. Defaults to ``0.05``.

    Returns:
        Float mean of the left-tail samples (``samples <= VaR_alpha``). If
        no sample is at-or-below the threshold (degenerate case at very small
        ``alpha`` and small ``N``), falls back to returning ``VaR_alpha``
        itself so the function never raises on a non-empty input.

    Raises:
        ValueError: If ``samples`` is empty or ``alpha`` is outside ``(0, 1)``.

    Notes:
        For a finite sample with ``K = floor(alpha * N)`` left-tail elements,
        the empirical CVaR equals the mean of the smallest ``K + 1`` samples
        (the ``+1`` includes the VaR-quantile sample itself). At small
        ``alpha * N`` (rare-tail estimation) the estimator has high variance;
        use :func:`importance_sample` to drive the proposal onto the tail
        first.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            "conditional_value_at_risk: alpha must lie in (0, 1); got "
            f"{alpha!r}."
        )
    arr = npa.asarray(samples)
    if arr.size == 0:
        raise ValueError(
            "conditional_value_at_risk: samples must be non-empty."
        )
    if arr.ndim != 1:
        raise ValueError(
            "conditional_value_at_risk: samples must be 1-D (n_samples,); "
            f"got shape {arr.shape}."
        )

    var_threshold = float(npa.quantile(arr, float(alpha)))
    # Use the .item() / float() coerced array path; npa.asarray handles both
    # numpy and jax dispatchers.
    arr_np = npa.asarray(arr)
    mask = arr_np <= var_threshold
    tail = arr_np[mask]
    if tail.size == 0:
        # Degenerate fallback (e.g. alpha so small no sample qualifies under
        # rounding). Honest answer: report the threshold itself.
        return var_threshold
    return float(npa.mean(tail))
