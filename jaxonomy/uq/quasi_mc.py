# SPDX-License-Identifier: MIT

"""Quasi-Monte-Carlo sampling via Sobol and Halton low-discrepancy sequences
(T-126 followup).

Quasi-Monte Carlo (QMC) replaces IID pseudo-random draws with a deterministic
*low-discrepancy* sequence (Sobol, Halton, ...).  For smooth integrands the
expected error converges as ``O(N^-1 (log N)^d)`` — effectively ``O(1/N)`` for
moderate ``d`` — versus IID MC's ``O(1/sqrt(N))``.  In practice QMC wins by an
order of magnitude or more once ``N >= 256`` and the integrand is reasonably
smooth.

This module exposes three helpers:

* :func:`sobol_sequence` — draw an ``(n_samples, n_dims)`` array of points in
  ``[0, 1)^n_dims`` from a (optionally scrambled) Sobol sequence.  Routes
  through :class:`scipy.stats.qmc.Sobol`.
* :func:`halton_sequence` — pure-Python Halton low-discrepancy sequence using
  the van der Corput construction with the first ``n_dims`` primes as bases
  (or user-supplied ``base_primes``).  No scipy required.
* :func:`quasi_monte_carlo` — high-level helper that generates a low-
  discrepancy grid (``sequence={"sobol","halton"}``) and pushes each column
  through the matching distribution's ``ppf`` (inverse-CDF), returning a
  ``{param_name: ndarray of shape (n_samples,)}`` dict suitable for
  :func:`decompose_variance`, :func:`sobol_indices`, or any QoI evaluator
  that accepts a ``param_batches``-shaped mapping.

All three return :mod:`jax.numpy` arrays (under the T-005 default-float64
policy) so the output plugs straight into the same kernel paths as
:func:`sample_parameters` / :func:`latin_hypercube_sample`.

``scipy`` is *required* for Sobol QMC.  Implementing the Sobol direction-
number tables in pure Python is a non-trivial body of code (Joe & Kuo 2008
supplies 21201 direction numbers); rather than ship a half-built fallback we
declare scipy as the Sobol backend and document it as the ``[uq-qmc]``
optional extra.  Halton, by contrast, is pure Python — no extra dependency.

**Sobol vs Halton.**  Sobol typically dominates Halton for ``n_dims < 8``: the
direction-number machinery gives it tighter projections onto every
2-D coordinate plane.  Halton's pure-prime-base construction starts to
develop visible correlations between dimensions once ``n_dims > 10`` (large
primes generate long, low-frequency stripes in 2-D projections) and is
generally recommended only with scrambling beyond ``n_dims = 8``.  This
module's unscrambled Halton is fine for screening / smoke tests up to ~10
dims; for production high-dimensional QMC prefer Sobol or a scrambled
Halton.

Example::

    >>> import jax  # doctest: +SKIP
    >>> from jaxonomy.uq import quasi_monte_carlo, Uniform, Normal  # doctest: +SKIP
    >>> dists = {"k": Uniform(0.5, 2.0), "tau": Normal(1.0, 0.1)}
    >>> samples = quasi_monte_carlo(dists, n_samples=1024, seed=0)  # doctest: +SKIP
    >>> samples["k"].shape  # doctest: +SKIP
    (1024,)
    >>> halton = quasi_monte_carlo(dists, n_samples=1024, sequence="halton")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as _np

from jaxonomy.backend import numpy_api as npa

from .distributions import Distribution

__all__ = [
    "sobol_sequence",
    "halton_sequence",
    "quasi_monte_carlo",
]


# ---------------------------------------------------------------------------
# Primes (used as default Halton bases).  The Halton sequence loses
# uniformity quickly past ~30 dims; we ship the first 50 primes which is
# more than enough headroom for any sane UQ setup.
# ---------------------------------------------------------------------------
_FIRST_PRIMES: tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
)


def _van_der_corput(n_samples: int, base: int, offset: int = 1) -> _np.ndarray:
    """Van der Corput sequence in the given integer base.

    Computes ``phi_base(i)`` for ``i = offset, offset+1, ..., offset + n - 1``
    by writing ``i`` in base ``base`` and reflecting the digits across the
    radix point.  Vectorised over the index axis so the ``n`` loop is in
    digit-position space (typically ~30 iterations for ``n <= 1e9``).

    The default ``offset=1`` skips the all-zero index 0 (which would map to
    exactly 0 in every base) — this matches the canonical Halton convention
    of starting at ``1/base``.
    """
    if base < 2:
        raise ValueError(f"_van_der_corput: base must be >= 2, got {base}")
    indices = _np.arange(offset, offset + n_samples, dtype=_np.int64)
    out = _np.zeros(n_samples, dtype=_np.float64)
    f = 1.0
    remaining = indices.copy()
    # Loop until every index is reduced to zero.  log_base(max_index) bounds
    # the number of iterations; ~30 for n=1e9 in base 2 (the worst case).
    while _np.any(remaining > 0):
        f /= base
        out += f * (remaining % base)
        remaining //= base
    return out


def _require_scipy_qmc():
    """Import scipy.stats.qmc or raise a clear error pointing at the extra."""
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - scipy is a hard test dep
        raise ImportError(
            "Quasi-Monte-Carlo sampling requires scipy. Install with "
            "`pip install scipy` or via the [uq-qmc] extra."
        ) from exc
    return qmc


def sobol_sequence(
    n_samples: int,
    n_dims: int,
    seed: Optional[int] = None,
    scramble: bool = True,
):
    """Draw an ``(n_samples, n_dims)`` Sobol low-discrepancy sample.

    Args:
        n_samples: Number of points to draw.  Sobol is most uniform when
            ``n_samples`` is a power of 2; non-power-of-2 sizes are
            accepted but lose some balance properties.
        n_dims: Number of dimensions.  scipy's Sobol generator supports
            up to ``n_dims=21201`` via the Joe-Kuo direction numbers.
        seed: Optional integer PRNG seed.  Only consumed when
            ``scramble=True``; an unscrambled Sobol sequence is fully
            deterministic regardless of the seed.
        scramble: If ``True`` (default), apply Owen scrambling.  This
            keeps the low-discrepancy property while randomising the
            exact point locations — variance estimates are unbiased and
            you can average over independent scrambles to estimate QMC
            error bars.  An unscrambled sequence starts at exactly 0,
            which can be a problem when feeding heavy-tailed inverse-CDFs.

    Returns:
        ``ndarray`` of shape ``(n_samples, n_dims)`` with all entries in
        ``[0, 1)``.  Backed by :mod:`jaxonomy.backend.numpy_api` so it
        plugs straight into the same kernel paths as the IID sampler.
    """
    if n_samples <= 0:
        raise ValueError(
            f"sobol_sequence: n_samples must be > 0, got {n_samples}"
        )
    if n_dims <= 0:
        raise ValueError(
            f"sobol_sequence: n_dims must be > 0, got {n_dims}"
        )

    qmc = _require_scipy_qmc()
    engine = qmc.Sobol(d=n_dims, scramble=scramble, seed=seed)
    # ``random`` returns a numpy float64 array in [0, 1) (the open
    # boundary is enforced by scipy's clamping).
    raw = engine.random(n_samples)
    return npa.asarray(raw)


def halton_sequence(
    n_samples: int,
    n_dims: int,
    base_primes: Optional[Sequence[int]] = None,
    scramble: bool = False,
    seed: Optional[int] = None,
):
    """Draw an ``(n_samples, n_dims)`` Halton low-discrepancy sample.

    The Halton sequence is the multidimensional generalisation of the van
    der Corput sequence: dimension ``d`` is filled by the radical-inverse
    expansion of ``i = 1, 2, ...`` in base ``base_primes[d]`` (default: the
    ``d``-th prime).  Unlike Sobol, Halton requires no precomputed
    direction tables — the implementation is a few lines of pure Python /
    numpy and so does not depend on scipy.

    Args:
        n_samples: Number of points to draw.  Halton has no power-of-2
            preference (unlike Sobol); any positive ``n_samples`` works.
        n_dims: Number of dimensions.  Bounded above by the length of
            ``base_primes`` (or the built-in 50-prime table if
            ``base_primes`` is left at the default).
        base_primes: Optional sequence of primes, one per dimension.
            ``None`` (default) uses the first ``n_dims`` primes:
            ``(2, 3, 5, 7, 11, 13, ...)``.  Supplied bases must be ``>= 2``
            and pairwise distinct — repeats produce perfectly correlated
            dimensions, which silently destroys the low-discrepancy
            property.
        scramble: If ``True``, apply a per-dimension *random-digit*
            scramble using ``seed``.  Specifically, each digit in the base
            expansion is permuted by a deterministic random permutation
            chosen per-dimension.  This eliminates the systematic
            high-dimensional correlations that vanilla Halton exhibits for
            ``n_dims > 8`` (large primes generate long, low-frequency
            stripes in 2-D projections).  Default is ``False`` so the
            sequence is fully deterministic; flip it on for high-dim
            screening.
        seed: PRNG seed for the digit scramble.  Ignored when
            ``scramble=False``.  Default seed is ``0`` for reproducibility.

    Returns:
        ``ndarray`` of shape ``(n_samples, n_dims)`` with all entries in
        ``[0, 1)``.  Backed by :mod:`jaxonomy.backend.numpy_api`.

    Notes:
        Halton's pairwise-correlation issue is well known: for
        ``n_dims > 10``, *unscrambled* Halton can be visibly worse than
        IID on some 2-D projections (Kocis & Whiten 1997).  Two
        mitigations: (1) ``scramble=True`` (cheap and effective), or
        (2) prefer :func:`sobol_sequence` for ``n_dims > 8``.  Both are
        documented here so callers can choose with eyes open.
    """
    if n_samples <= 0:
        raise ValueError(
            f"halton_sequence: n_samples must be > 0, got {n_samples}"
        )
    if n_dims <= 0:
        raise ValueError(
            f"halton_sequence: n_dims must be > 0, got {n_dims}"
        )

    if base_primes is None:
        if n_dims > len(_FIRST_PRIMES):
            raise ValueError(
                f"halton_sequence: default prime table covers "
                f"{len(_FIRST_PRIMES)} dims; got n_dims={n_dims}.  Pass "
                "``base_primes`` explicitly to extend, but note that "
                "Halton's correlation issues compound at higher dims — "
                "prefer sobol_sequence beyond ~8 dimensions."
            )
        bases = list(_FIRST_PRIMES[:n_dims])
    else:
        bases = [int(b) for b in base_primes]
        if len(bases) != n_dims:
            raise ValueError(
                f"halton_sequence: base_primes has {len(bases)} entries but "
                f"n_dims={n_dims}; must match."
            )
        if any(b < 2 for b in bases):
            raise ValueError(
                f"halton_sequence: every base must be >= 2; got {bases}."
            )
        if len(set(bases)) != len(bases):
            raise ValueError(
                f"halton_sequence: bases must be pairwise distinct; got "
                f"{bases}.  Repeated bases produce perfectly correlated "
                "dimensions and destroy the low-discrepancy property."
            )

    rng = _np.random.default_rng(0 if seed is None else seed)

    out = _np.empty((n_samples, n_dims), dtype=_np.float64)
    for d, base in enumerate(bases):
        if not scramble:
            out[:, d] = _van_der_corput(n_samples, base, offset=1)
            continue

        # Digit-scrambled Halton: build a deterministic random permutation
        # of {0, ..., base-1} for this dimension and apply it digit-wise.
        # Fix digit 0 -> 0 to keep the sequence in [0, 1).
        perm = _np.arange(base, dtype=_np.int64)
        # Shuffle entries [1, base) only — leaving 0 alone makes the
        # result fall in [0, 1) without truncation.
        if base > 2:
            tail = perm[1:].copy()
            rng.shuffle(tail)
            perm[1:] = tail

        indices = _np.arange(1, n_samples + 1, dtype=_np.int64)
        col = _np.zeros(n_samples, dtype=_np.float64)
        f = 1.0
        remaining = indices.copy()
        while _np.any(remaining > 0):
            f /= base
            digit = remaining % base
            col += f * perm[digit]
            remaining //= base
        out[:, d] = col

    return npa.asarray(out)


def quasi_monte_carlo(
    distributions: Mapping[str, Distribution],
    n_samples: int,
    seed: Optional[int] = None,
    scramble: bool = True,
    sequence: str = "sobol",
) -> dict:
    """Generate QMC samples and push each column through the matching
    distribution's inverse-CDF.

    The output is shaped exactly like :func:`sample_parameters` /
    :func:`latin_hypercube_sample` so it is a drop-in replacement when the
    QoI is smooth (linear systems, polynomial responses, lookup-table
    interpolations).

    **Smoothness caveat (T-126-followup-qmc-smoothness-doc).** QMC's
    asymptotic advantage over IID Monte Carlo (``O(N^{-1} (log N)^d)`` vs
    ``O(N^{-1/2})``) is guaranteed only for QoIs of *bounded Hardy-Krause
    variation* — intuitively, "smooth enough" functions of the unit cube
    input. On QoIs with discontinuities the guarantee fails and the rate
    can drop to (or below) IID:

    - **Event-triggered switches** (zero-crossing resets, hard saturations
      on the active path, conditional logic on the state) introduce
      step-function discontinuities in the QoI as a function of the
      parameter, so the variation is unbounded.
    - **Max-of-trajectory kinks** (``max(state(t))`` and similar
      ``argmax``-style QoIs) are non-smooth at the argmax flip; the
      Hardy-Krause variation is again unbounded at the kink.
    - **Hybrid systems with state-dependent mode switches** inherit the
      problem from the underlying switch.

    Practical guidance: prefer LHS or IID for hybrid / event-driven QoIs;
    use QMC for QoIs that are differentiable as a function of the
    parameter (e.g. RMS errors on linear plants, polynomial cost
    functions, smooth interpolation lookups). When in doubt, run a
    convergence check at ``N = 2^k`` for ``k = 6..12`` and confirm the
    error halves with each doubling — if it plateaus near ``N^{-1/2}``,
    the smoothness assumption has failed.

    Args:
        distributions: Mapping ``param_name -> Distribution``.  Each
            distribution must expose ``ppf(u)`` — continuous distributions
            (:class:`Uniform`, :class:`Normal`, :class:`LogNormal`,
            :class:`Triangular`, :class:`Exponential`, :class:`Beta`,
            :class:`Gamma`) all qualify.  Discrete distributions
            (:class:`Poisson`, :class:`Categorical`, :class:`Bernoulli`)
            do *not* expose ``ppf`` and are rejected at this surface:
            their step-function inverse-CDF would break the smoothness
            assumption that gives QMC its convergence advantage.
        n_samples: Number of QMC points; powers of 2 give the best
            uniformity for Sobol (Halton is power-of-2-agnostic).
        seed: Optional integer PRNG seed.  For Sobol it controls Owen
            scrambling; for Halton it controls digit-scramble permutations.
            Ignored when ``scramble=False``.
        scramble: If ``True`` (default), scramble the sequence.  For Sobol
            this is Owen scrambling (requires scipy); for Halton this is
            random-digit scrambling (pure Python).  See
            :func:`sobol_sequence` / :func:`halton_sequence` for details.
        sequence: ``"sobol"`` (default) or ``"halton"``.  Sobol typically
            dominates for ``n_dims < 8``; Halton is dependency-free but
            should be scrambled for ``n_dims > 8``.

    Returns:
        Dict ``{param_name: ndarray of shape (n_samples,)}``.
    """
    if n_samples <= 0:
        raise ValueError(
            f"quasi_monte_carlo: n_samples must be > 0, got {n_samples}"
        )
    if not distributions:
        raise ValueError(
            "quasi_monte_carlo: distributions dict must be non-empty."
        )
    # Reject distributions that do not expose a smooth inverse-CDF.  We
    # check by attribute lookup so any future continuous distribution
    # picks this up automatically.
    bad = [name for name, dist in distributions.items() if not hasattr(dist, "ppf")]
    if bad:
        raise ValueError(
            "quasi_monte_carlo: every distribution must expose 'ppf' "
            f"(inverse-CDF); these do not: {bad}.  Discrete distributions "
            "(Poisson, Categorical, Bernoulli) are deliberately excluded — "
            "their step-function inverse-CDF breaks the smoothness "
            "assumption QMC relies on."
        )

    n_dims = len(distributions)
    seq = sequence.lower()
    if seq == "sobol":
        u = sobol_sequence(n_samples, n_dims, seed=seed, scramble=scramble)
    elif seq == "halton":
        u = halton_sequence(
            n_samples, n_dims, scramble=scramble, seed=seed,
        )
    else:
        raise ValueError(
            f"quasi_monte_carlo: unknown sequence {sequence!r}; "
            "must be one of 'sobol', 'halton'."
        )

    out: dict = {}
    for i, (name, dist) in enumerate(distributions.items()):
        out[name] = dist.ppf(u[:, i])
    return out
