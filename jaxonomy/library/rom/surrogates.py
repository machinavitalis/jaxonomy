# SPDX-License-Identifier: MIT

"""Statistical surrogate models and matching feedthrough blocks (T-148..T-150).

Three response-surface / surrogate families for reduced-order modeling:

* Gaussian process (kriging) regression -- Rasmussen & Williams, *Gaussian
  Processes for Machine Learning*, MIT Press, 2006.
* Polynomial chaos expansion (PCE) in the Wiener--Askey scheme -- Xiu &
  Karniadakis, "The Wiener--Askey polynomial chaos for stochastic differential
  equations", SIAM J. Sci. Comput. 24(2):619--644, 2002.
* Radial basis function (RBF) interpolation -- Hardy, "Multiquadric equations
  of topography and other irregular surfaces", J. Geophys. Res. 76(8):1905--1915,
  1971; Wendland, *Scattered Data Approximation*, Cambridge Univ. Press, 2005.

Host-side fitting uses ordinary (jax) numpy and linear algebra; the ``predict``
methods and the ``LeafSystem`` output callbacks are written with ``jax.numpy``
so they stay jax-traceable (jit / grad / vmap). Fitted coefficients are exposed
as dynamic parameters on the blocks so a surrogate is differentiable in its
weights.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from ...framework import LeafSystem

__all__ = [
    # Gaussian process (kriging)
    "GPModel",
    "fit_gp",
    "GaussianProcess",
    # Polynomial chaos expansion
    "PCEModel",
    "fit_pce",
    "PolynomialChaos",
    # Radial basis function
    "RBFModel",
    "fit_rbf",
    "RadialBasisSurrogate",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _as2d(X) -> jnp.ndarray:
    """Return X as a 2-D (n_samples, n_features) float array."""
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim == 1:
        X = X[:, None]
    return X


def _row(u) -> jnp.ndarray:
    """Interpret a block input as a single feature vector, shape ``(1, d)``."""
    return jnp.atleast_2d(jnp.asarray(u, dtype=jnp.float64))


def _sqdist(Xa: jnp.ndarray, Xb: jnp.ndarray) -> jnp.ndarray:
    """Squared Euclidean distance matrix, shape (n_a, n_b). Clipped to >= 0."""
    a2 = jnp.sum(Xa * Xa, axis=1)[:, None]
    b2 = jnp.sum(Xb * Xb, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (Xa @ Xb.T)
    return jnp.clip(d2, 0.0, None)


# ===========================================================================
# Gaussian process / kriging  (T-148)
# ===========================================================================
def _gp_kernel(Xa, Xb, kernel, length_scale, signal_var, matern_nu):
    """Covariance matrix k(Xa, Xb). RBF (squared-exponential) or Matern."""
    d2 = _sqdist(Xa, Xb)
    if kernel in ("rbf", "squared_exponential", "se"):
        return signal_var * jnp.exp(-0.5 * d2 / (length_scale ** 2))
    if kernel in ("matern", "matern32", "matern52"):
        r = jnp.sqrt(jnp.clip(d2, 1e-36, None))
        if kernel == "matern32" or (kernel == "matern" and matern_nu == 1.5):
            arg = math.sqrt(3.0) * r / length_scale
            return signal_var * (1.0 + arg) * jnp.exp(-arg)
        # default Matern 5/2
        arg = math.sqrt(5.0) * r / length_scale
        return signal_var * (1.0 + arg + (arg ** 2) / 3.0) * jnp.exp(-arg)
    raise ValueError(f"Unknown GP kernel {kernel!r}")


class GPModel:
    """Fitted Gaussian-process regressor (kriging).

    Stores the training inputs, the pre-solved weight vector ``alpha = K^-1 y``,
    and the Cholesky factor of the (noisy) covariance matrix for variance
    prediction. See Rasmussen & Williams 2006, Algorithm 2.1.
    """

    def __init__(self, X, y, alpha, L, kernel, length_scale, signal_var,
                 noise, matern_nu):
        self.X_train = X
        self.y_train = y
        self.alpha = alpha
        self.L = L
        self.kernel = kernel
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise = float(noise)
        self.matern_nu = float(matern_nu)

    def predict(self, Xstar):
        """Posterior ``(mean, variance)`` at query points ``Xstar``.

        jax-traceable. ``Xstar`` may be 1-D (single point / single feature) or
        2-D ``(m, d)``. Returns arrays of shape ``(m,)``.
        """
        Xstar = _as2d(Xstar)
        Ks = _gp_kernel(Xstar, self.X_train, self.kernel, self.length_scale,
                        self.signal_var, self.matern_nu)  # (m, n)
        mean = Ks @ self.alpha
        v = jsla.solve_triangular(self.L, Ks.T, lower=True)  # (n, m)
        var = self.signal_var - jnp.sum(v * v, axis=0)
        var = jnp.clip(var, 0.0, None)
        return mean, var

    def log_marginal_likelihood(self):
        n = self.X_train.shape[0]
        data_fit = -0.5 * jnp.dot(self.y_train, self.alpha)
        complexity = -jnp.sum(jnp.log(jnp.diag(self.L)))
        return data_fit + complexity - 0.5 * n * math.log(2.0 * math.pi)


def _gp_solve(X, yc, kernel, length_scale, signal_var, noise, matern_nu):
    n = X.shape[0]
    K = _gp_kernel(X, X, kernel, length_scale, signal_var, matern_nu)
    K = K + (noise + 1e-12) * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    alpha = jsla.cho_solve((L, True), yc)
    return alpha, L


def fit_gp(X, y, kernel="rbf", length_scale=1.0, signal_var=1.0, noise=1e-8,
           optimize=False, n_restarts=0, lr=0.05, n_steps=200, matern_nu=2.5):
    """Fit a Gaussian-process (kriging) surrogate.

    Args:
        X: training inputs, shape ``(n,)`` or ``(n, d)``.
        y: training targets, shape ``(n,)``.
        kernel: ``"rbf"`` / ``"squared_exponential"`` or ``"matern"`` /
            ``"matern32"`` / ``"matern52"``.
        length_scale, signal_var, noise: kernel hyperparameters (initial values
            when ``optimize=True``).
        optimize: if True, maximize the marginal log-likelihood over
            ``(length_scale, signal_var, noise)`` by gradient ascent in
            log-space (Rasmussen & Williams 2006, Eq. 5.9).

    Returns:
        A :class:`GPModel`.
    """
    X = _as2d(X)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)

    ls = float(length_scale)
    sv = float(signal_var)
    nz = float(noise)

    if optimize:
        # Optimize in log-space to keep hyperparameters positive.
        theta0 = jnp.log(jnp.array([ls, sv, nz], dtype=jnp.float64))

        def neg_lml(theta):
            ell, s, z = jnp.exp(theta)
            alpha, L = _gp_solve(X, y, kernel, ell, s, z, matern_nu)
            n = X.shape[0]
            val = (0.5 * jnp.dot(y, alpha)
                   + jnp.sum(jnp.log(jnp.diag(L)))
                   + 0.5 * n * math.log(2.0 * math.pi))
            return val

        grad_fn = jax.jit(jax.grad(neg_lml))
        theta = theta0
        for _ in range(int(n_steps)):
            g = grad_fn(theta)
            theta = theta - lr * g
        ls, sv, nz = (float(v) for v in jnp.exp(theta))

    alpha, L = _gp_solve(X, y, kernel, ls, sv, nz, matern_nu)
    return GPModel(X, y, alpha, L, kernel, ls, sv, nz, matern_nu)


class GaussianProcess(LeafSystem):
    """Gaussian-process (kriging) surrogate as a feedthrough block.

    Input port 0 is the feature vector ``u``; output port 0 is the predictive
    mean and output port 1 the predictive variance. Training data and the
    Cholesky factor are stored statically on the block; the weight vector
    ``alpha`` and the kernel hyperparameters are dynamic parameters so the
    surrogate is differentiable in them (Rasmussen & Williams 2006).
    """

    def __init__(self, model: GPModel, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.declare_input_port()
        self.declare_dynamic_parameter("alpha", jnp.asarray(model.alpha))
        self.declare_dynamic_parameter(
            "length_scale", jnp.asarray(model.length_scale, dtype=jnp.float64))
        self.declare_dynamic_parameter(
            "signal_var", jnp.asarray(model.signal_var, dtype=jnp.float64))

        self._mean_port_idx = self.declare_output_port(
            self._mean, name="mean",
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )
        self._var_port_idx = self.declare_output_port(
            self._variance, name="variance",
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    def _mean(self, time, state, *inputs, **params):
        Xstar = _row(inputs[0])
        Ks = _gp_kernel(Xstar, self.model.X_train, self.model.kernel,
                        params["length_scale"], params["signal_var"],
                        self.model.matern_nu)
        return (Ks @ params["alpha"])[0]

    def _variance(self, time, state, *inputs, **params):
        Xstar = _row(inputs[0])
        Ks = _gp_kernel(Xstar, self.model.X_train, self.model.kernel,
                        params["length_scale"], params["signal_var"],
                        self.model.matern_nu)
        v = jsla.solve_triangular(self.model.L, Ks.T, lower=True)
        var = params["signal_var"] - jnp.sum(v * v, axis=0)
        return jnp.clip(var, 0.0, None)[0]


# ===========================================================================
# Polynomial chaos expansion  (T-149)
# ===========================================================================
def _total_degree_indices(dim: int, order: int) -> np.ndarray:
    """All multi-indices ``alpha`` (dim-tuples) with ``sum(alpha) <= order``.

    Total-degree truncation. The all-zero index (constant term) is first.
    """
    def _gen(d, o):
        if d == 0:
            return [()]
        out = []
        for i in range(o + 1):
            for rest in _gen(d - 1, o - i):
                out.append((i,) + rest)
        return out

    return np.array(_gen(dim, order), dtype=np.int64)


def _hermite_ortho(xi: jnp.ndarray, order: int) -> jnp.ndarray:
    """Orthonormal probabilists' Hermite polynomials evaluated at ``xi``.

    Returns ``(len(xi), order+1)``; column n is ``He_n(xi)/sqrt(n!)``, so the
    family is orthonormal w.r.t. the standard-normal weight (Xiu & Karniadakis
    2002, Table 4.1 -- Askey scheme, Gaussian -> Hermite).
    """
    cols = [jnp.ones_like(xi)]
    if order >= 1:
        cols.append(xi)
    for n in range(1, order):
        cols.append(xi * cols[n] - n * cols[n - 1])
    P = jnp.stack(cols, axis=-1)
    norms = jnp.array([math.sqrt(math.factorial(n)) for n in range(order + 1)],
                      dtype=jnp.float64)
    return P / norms


def _legendre_ortho(xi: jnp.ndarray, order: int) -> jnp.ndarray:
    """Orthonormal Legendre polynomials on ``[-1, 1]`` evaluated at ``xi``.

    Returns ``(len(xi), order+1)``; column n is ``P_n(xi)*sqrt(2n+1)``, so the
    family is orthonormal w.r.t. the uniform weight (Askey scheme,
    uniform -> Legendre).
    """
    cols = [jnp.ones_like(xi)]
    if order >= 1:
        cols.append(xi)
    for n in range(1, order):
        cols.append(((2 * n + 1) * xi * cols[n] - n * cols[n - 1]) / (n + 1))
    P = jnp.stack(cols, axis=-1)
    norms = jnp.array([math.sqrt(2 * n + 1) for n in range(order + 1)],
                      dtype=jnp.float64)
    return P * norms


def _parse_distributions(distributions):
    """Return (types, loc, scale) with loc/scale mapping x -> standard germ.

    normal (mu, sigma):   xi = (x - mu) / sigma
    uniform (a, b):       xi = 2*(x - a)/(b - a) - 1  == (x - mid)/half
    """
    types, loc, scale = [], [], []
    for spec in distributions:
        kind = spec[0].lower()
        if kind == "normal":
            _, mu, sigma = spec
            types.append("normal")
            loc.append(float(mu))
            scale.append(float(sigma))
        elif kind == "uniform":
            _, a, b = spec
            types.append("uniform")
            loc.append(0.5 * (a + b))
            scale.append(0.5 * (b - a))
        else:
            raise ValueError(f"Unsupported PCE distribution {kind!r}")
    return types, np.array(loc), np.array(scale)


def _pce_standardize(X, loc, scale):
    return (X - jnp.asarray(loc)) / jnp.asarray(scale)


def _pce_design(Xi, multi_indices, types, order):
    """Vandermonde-like matrix ``Psi[:, k] = prod_d psi_{alpha_k,d}(Xi_d)``."""
    m, dim = Xi.shape
    per_dim = []
    for d in range(dim):
        if types[d] == "normal":
            per_dim.append(_hermite_ortho(Xi[:, d], order))
        else:
            per_dim.append(_legendre_ortho(Xi[:, d], order))
    K = multi_indices.shape[0]
    Psi = jnp.ones((m, K), dtype=jnp.float64)
    for d in range(dim):
        Psi = Psi * per_dim[d][:, multi_indices[:, d]]
    return Psi


class PCEModel:
    """Fitted polynomial-chaos expansion ``y = sum_k c_k Psi_k(xi)``.

    The orthonormal basis (Askey scheme) gives closed-form statistics: the mean
    is the constant coefficient, the variance is the sum of squared non-constant
    coefficients, and Sobol indices follow from partitioning that sum by which
    inputs each basis term depends on (Xiu & Karniadakis 2002; Sudret,
    *Reliab. Eng. Syst. Saf.* 93(7):964--979, 2008).
    """

    def __init__(self, coeffs, multi_indices, types, loc, scale, order):
        self.coeffs = coeffs
        self.multi_indices = multi_indices  # np.ndarray (K, dim)
        self.types = types
        self.loc = loc
        self.scale = scale
        self.order = int(order)
        self.dim = multi_indices.shape[1]
        # index of the all-zero (constant) multi-index
        self._const_idx = int(np.argmin(multi_indices.sum(axis=1)))

    def predict(self, Xstar):
        """Surrogate response at ``Xstar`` (jax-traceable)."""
        Xstar = _as2d(Xstar)
        Xi = _pce_standardize(Xstar, self.loc, self.scale)
        Psi = _pce_design(Xi, self.multi_indices, self.types, self.order)
        return Psi @ self.coeffs

    def mean(self):
        """Analytic mean = constant-term coefficient."""
        return self.coeffs[self._const_idx]

    def variance(self):
        """Analytic variance = sum of squared non-constant coefficients."""
        mask = np.ones(self.coeffs.shape[0], dtype=bool)
        mask[self._const_idx] = False
        return jnp.sum(jnp.asarray(self.coeffs)[jnp.asarray(mask)] ** 2)

    def sobol_indices(self):
        """Main-effect (first-order) and total Sobol indices per input.

        Returns a dict ``{"first_order": (dim,), "total": (dim,)}``.
        """
        c2 = np.asarray(self.coeffs) ** 2
        idx = self.multi_indices
        nonconst = idx.sum(axis=1) > 0
        total_var = c2[nonconst].sum()

        first = np.zeros(self.dim)
        total = np.zeros(self.dim)
        for i in range(self.dim):
            involves_i = idx[:, i] > 0
            others_zero = (idx.sum(axis=1) == idx[:, i])
            main_mask = involves_i & others_zero
            first[i] = c2[main_mask].sum() / total_var
            total[i] = c2[involves_i].sum() / total_var
        return {"first_order": jnp.asarray(first), "total": jnp.asarray(total)}


def fit_pce(X, y, distributions: Sequence, order: int):
    """Fit a polynomial-chaos expansion by least-squares regression.

    Args:
        X: training inputs, shape ``(n,)`` or ``(n, d)``.
        y: training targets, shape ``(n,)``.
        distributions: per-dimension germ, e.g. ``[("normal", mu, sigma),
            ("uniform", a, b)]``. Hermite basis for normal, Legendre for uniform
            (Wiener--Askey scheme, Xiu & Karniadakis 2002).
        order: total-degree truncation.

    Returns:
        A :class:`PCEModel`.
    """
    X = _as2d(X)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    dim = X.shape[1]
    if len(distributions) != dim:
        raise ValueError(
            f"distributions has {len(distributions)} entries but X has {dim} "
            "feature dimensions")

    types, loc, scale = _parse_distributions(distributions)
    multi_indices = _total_degree_indices(dim, int(order))
    Xi = _pce_standardize(X, loc, scale)
    Psi = _pce_design(Xi, multi_indices, types, int(order))
    coeffs, *_ = jnp.linalg.lstsq(Psi, y, rcond=None)
    return PCEModel(coeffs, multi_indices, types, loc, scale, order)


class PolynomialChaos(LeafSystem):
    """Polynomial-chaos surrogate ``y = sum_k c_k Psi_k(u)`` as a feedthrough
    block. Input port 0 is the feature vector ``u``; the coefficients are a
    dynamic parameter (Xiu & Karniadakis 2002)."""

    def __init__(self, model: PCEModel, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.declare_input_port()
        self.declare_dynamic_parameter("coeffs", jnp.asarray(model.coeffs))
        self._output_port_idx = self.declare_output_port(
            self._eval_output, name="y",
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    def _eval_output(self, time, state, *inputs, **params):
        Xstar = _row(inputs[0])
        Xi = _pce_standardize(Xstar, self.model.loc, self.model.scale)
        Psi = _pce_design(Xi, self.model.multi_indices, self.model.types,
                          self.model.order)
        return (Psi @ params["coeffs"])[0]


# ===========================================================================
# Radial basis function  (T-150)
# ===========================================================================
def _rbf_phi(d2: jnp.ndarray, kernel: str, epsilon: float) -> jnp.ndarray:
    """RBF kernel evaluated from squared distances ``d2`` (Wendland 2005)."""
    e2 = epsilon ** 2
    if kernel == "multiquadric":
        return jnp.sqrt(1.0 + e2 * d2)
    if kernel in ("inverse_multiquadric", "inverse-multiquadric", "imq"):
        return 1.0 / jnp.sqrt(1.0 + e2 * d2)
    if kernel == "gaussian":
        return jnp.exp(-e2 * d2)
    if kernel in ("thin_plate_spline", "thin-plate-spline", "thin_plate", "tps"):
        # r^2 log(r) = 0.5 * d2 * log(d2), taken 0 at d2 = 0.
        return jnp.where(d2 > 0.0, 0.5 * d2 * jnp.log(jnp.clip(d2, 1e-36, None)),
                         0.0)
    raise ValueError(f"Unknown RBF kernel {kernel!r}")


def _rbf_monomials(X: jnp.ndarray, indices: np.ndarray) -> jnp.ndarray:
    """Polynomial-tail design matrix: column k is ``prod_d X_d^alpha_{k,d}``."""
    m = X.shape[0]
    P = jnp.ones((m, indices.shape[0]), dtype=jnp.float64)
    for d in range(X.shape[1]):
        P = P * X[:, d][:, None] ** jnp.asarray(indices[:, d])
    return P


class RBFModel:
    """Fitted radial-basis-function interpolant with optional polynomial tail.

    ``s(x) = sum_i w_i phi(||x - c_i||) + sum_k c_k p_k(x)`` (Hardy 1971;
    Wendland 2005).
    """

    def __init__(self, centers, weights, poly_coeffs, poly_indices, kernel,
                 epsilon):
        self.centers = centers
        self.weights = weights
        self.poly_coeffs = poly_coeffs      # None if no tail
        self.poly_indices = poly_indices    # None if no tail
        self.kernel = kernel
        self.epsilon = float(epsilon)

    def predict(self, Xstar):
        """Interpolant value at ``Xstar`` (jax-traceable)."""
        Xstar = _as2d(Xstar)
        d2 = _sqdist(Xstar, self.centers)
        y = _rbf_phi(d2, self.kernel, self.epsilon) @ self.weights
        if self.poly_indices is not None:
            y = y + _rbf_monomials(Xstar, self.poly_indices) @ self.poly_coeffs
        return y


def fit_rbf(X, y, kernel="multiquadric", epsilon=1.0, smoothing=0.0,
            poly_degree=None):
    """Fit a radial-basis-function surrogate.

    Args:
        X: training inputs, shape ``(n,)`` or ``(n, d)``.
        y: training targets, shape ``(n,)``.
        kernel: ``"multiquadric"``, ``"inverse_multiquadric"``, ``"gaussian"``,
            or ``"thin_plate_spline"``.
        epsilon: shape parameter (ignored by the thin-plate spline).
        smoothing: ridge regularization added to the kernel diagonal; ``0`` gives
            exact interpolation.
        poly_degree: if set, augment with a total-degree polynomial tail and
            solve the bordered saddle-point system (Wendland 2005, Ch. 8).

    Returns:
        An :class:`RBFModel`.
    """
    X = _as2d(X)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    n = X.shape[0]

    d2 = _sqdist(X, X)
    A = _rbf_phi(d2, kernel, epsilon) + smoothing * jnp.eye(n)

    if poly_degree is None:
        weights = jnp.linalg.solve(A, y)
        return RBFModel(X, weights, None, None, kernel, epsilon)

    poly_indices = _total_degree_indices(X.shape[1], int(poly_degree))
    P = _rbf_monomials(X, poly_indices)  # (n, m)
    m = P.shape[1]
    top = jnp.concatenate([A, P], axis=1)
    bot = jnp.concatenate([P.T, jnp.zeros((m, m), dtype=jnp.float64)], axis=1)
    M = jnp.concatenate([top, bot], axis=0)
    rhs = jnp.concatenate([y, jnp.zeros(m, dtype=jnp.float64)])
    sol = jnp.linalg.solve(M, rhs)
    weights = sol[:n]
    poly_coeffs = sol[n:]
    return RBFModel(X, weights, poly_coeffs, poly_indices, kernel, epsilon)


class RadialBasisSurrogate(LeafSystem):
    """RBF surrogate ``y = sum_i w_i phi(||u - c_i||) (+ poly tail)`` as a
    feedthrough block. Input port 0 is the feature vector ``u``; the RBF weights
    (and polynomial-tail coefficients) are dynamic parameters (Hardy 1971)."""

    def __init__(self, model: RBFModel, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.declare_input_port()
        self.declare_dynamic_parameter("weights", jnp.asarray(model.weights))
        if model.poly_indices is not None:
            self.declare_dynamic_parameter(
                "poly_coeffs", jnp.asarray(model.poly_coeffs))
        self._output_port_idx = self.declare_output_port(
            self._eval_output, name="y",
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    def _eval_output(self, time, state, *inputs, **params):
        Xstar = _row(inputs[0])
        d2 = _sqdist(Xstar, self.model.centers)
        y = _rbf_phi(d2, self.model.kernel, self.model.epsilon) @ params["weights"]
        if self.model.poly_indices is not None:
            y = y + (_rbf_monomials(Xstar, self.model.poly_indices)
                     @ params["poly_coeffs"])
        return y[0]
