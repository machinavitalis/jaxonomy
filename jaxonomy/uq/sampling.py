# SPDX-License-Identifier: MIT

"""Parameter-distribution sampling for Monte Carlo / UQ.

Three named samplers:

* :func:`sample_parameters` — draw IID samples from a dict of distributions
  and return a ``param_batches``-shaped dict suitable for
  :func:`jaxonomy.simulate_batch`.
* :func:`latin_hypercube_sample` — Latin Hypercube Sampling (LHS) for
  stratified coverage of the unit cube; recommended over IID for small
  ``n_samples`` and as the input layer for screening methods.
* :func:`latin_hypercube_centered_sample` — *deterministic* LHS counterpart
  that places one sample at the *center* of every stratum (no within-stratum
  jitter; permutation across parameters still requires a key).  Useful for
  reproducible regression fixtures and as the deterministic baseline against
  which the jittered :func:`latin_hypercube_sample` is compared.

Example:

    >>> import jax
    >>> from jaxonomy.uq import sample_parameters, Uniform, Normal
    >>> dists = {"k": Uniform(0.5, 2.0), "tau": Normal(1.0, 0.1)}
    >>> batches = sample_parameters(dists, 256, jax.random.PRNGKey(0))
    >>> batches["k"].shape  # doctest: +SKIP
    (256,)
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp

from .distributions import Distribution

__all__ = [
    "sample_parameters",
    "latin_hypercube_sample",
    "latin_hypercube_centered_sample",
]


# ---------------------------------------------------------------------------
# IID sampling
# ---------------------------------------------------------------------------

def sample_parameters(
    distributions: Mapping[str, Distribution],
    n_samples: int,
    key,
) -> dict[str, jnp.ndarray]:
    """Draw ``n_samples`` IID samples from each distribution.

    Args:
        distributions: Mapping ``param_path -> Distribution``.  Keys are the
            same dot-paths accepted by :func:`jaxonomy.simulate_batch`'s
            ``param_batches`` argument.
        n_samples: Number of samples per parameter.
        key: ``jax.random.PRNGKey``.  A fresh subkey is split per parameter so
            that the marginals are independent.

    Returns:
        Dict ``{param_path: jnp.ndarray of shape (n_samples,)}``.
    """
    if n_samples <= 0:
        raise ValueError(f"sample_parameters: n_samples must be > 0, got {n_samples}")
    if not distributions:
        raise ValueError("sample_parameters: distributions dict must be non-empty.")

    keys = jax.random.split(key, len(distributions))
    return {
        name: dist.sample(k, (n_samples,))
        for k, (name, dist) in zip(keys, distributions.items())
    }


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sample(
    distributions: Mapping[str, Distribution],
    n_samples: int,
    key,
) -> dict[str, jnp.ndarray]:
    """Latin Hypercube samples mapped through each parameter's PPF.

    LHS partitions ``[0, 1]`` into ``n_samples`` equal strata per parameter and
    permutes one sample per stratum, producing more uniform 1-D coverage than
    IID at the same ``n_samples`` — a fixed-cost variance reduction useful for
    Monte Carlo when ``n_samples`` is in the low hundreds.  The PPF then maps
    each stratum into the parameter's native space.

    Args:
        distributions: Mapping ``param_path -> Distribution``.
        n_samples: Number of stratified samples per parameter.
        key: ``jax.random.PRNGKey``.  Used both for jittering within strata
            and for permuting them per parameter.

    Returns:
        Dict ``{param_path: jnp.ndarray of shape (n_samples,)}``.
    """
    if n_samples <= 0:
        raise ValueError(
            f"latin_hypercube_sample: n_samples must be > 0, got {n_samples}"
        )
    if not distributions:
        raise ValueError("latin_hypercube_sample: distributions dict must be non-empty.")

    d = len(distributions)
    keys = jax.random.split(key, 2 * d)
    edges = jnp.linspace(0.0, 1.0, n_samples + 1)
    lower, upper = edges[:-1], edges[1:]

    out: dict[str, jnp.ndarray] = {}
    for i, (name, dist) in enumerate(distributions.items()):
        jitter = jax.random.uniform(keys[i], shape=(n_samples,))
        u = lower + jitter * (upper - lower)
        u = jax.random.permutation(keys[d + i], u)
        out[name] = dist.ppf(u)
    return out


# ---------------------------------------------------------------------------
# Centered Latin Hypercube Sampling (deterministic counterpart)
# ---------------------------------------------------------------------------

def latin_hypercube_centered_sample(
    distributions: Mapping[str, Distribution],
    n_samples: int,
    key=None,
) -> dict[str, jnp.ndarray]:
    """Centered LHS: place one sample at the *center* of every stratum.

    The unit interval ``[0, 1]`` is partitioned into ``n_samples`` equal
    strata and the *midpoint* of each stratum is taken as the sample, giving
    the deterministic point set ``{(i + 0.5) / n_samples : 0 <= i < n}`` per
    parameter.  When ``key`` is provided each parameter's strata are
    independently permuted (the Latin-square property); when ``key=None``
    the strata are kept in increasing order, yielding a fully deterministic
    sample dictionary suitable for regression fixtures.

    Args:
        distributions: Mapping ``param_path -> Distribution``.
        n_samples: Number of stratified samples per parameter.
        key: Optional ``jax.random.PRNGKey``.  If provided, used to permute
            each parameter's stratum ordering independently (preserving the
            Latin-square property at the cost of determinism in ordering).
            If ``None``, samples are returned in monotonically increasing
            stratum order — fully deterministic but no Latin-square jitter
            across parameters.

    Returns:
        Dict ``{param_path: jnp.ndarray of shape (n_samples,)}``.
    """
    if n_samples <= 0:
        raise ValueError(
            "latin_hypercube_centered_sample: n_samples must be > 0, "
            f"got {n_samples}"
        )
    if not distributions:
        raise ValueError(
            "latin_hypercube_centered_sample: distributions dict must be non-empty."
        )

    # Stratum centers: (i + 0.5) / n for i = 0 .. n-1.
    centers = (jnp.arange(n_samples) + 0.5) / n_samples

    if key is None:
        out: dict[str, jnp.ndarray] = {}
        for name, dist in distributions.items():
            out[name] = dist.ppf(centers)
        return out

    keys = jax.random.split(key, len(distributions))
    out = {}
    for k, (name, dist) in zip(keys, distributions.items()):
        u = jax.random.permutation(k, centers)
        out[name] = dist.ppf(u)
    return out
