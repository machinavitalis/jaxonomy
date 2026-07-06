import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import LTISystem, linearize, LinearizedSystem
from jaxonomy.framework import DiagramBuilder
from jaxonomy.models.pendulum import Pendulum

def test_linearize_linear_system_exact():
    """Linearization of a linear system returns exact A,B,C,D."""
    # Build a first-order LTI: dx = -x + u, y = x
    # A=-1, B=1, C=1, D=0
    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]])
    )
    
    # Linearize around x0=0, u0=0
    system.input_ports[0].fix_value(jnp.array([0.0]))
    context = system.create_context()
    
    result = linearize(system, context)
    
    assert isinstance(result, LinearizedSystem)
    assert np.allclose(result.A, [[-1.0]])
    assert np.allclose(result.B, [[1.0]])
    assert np.allclose(result.C, [[1.0]])
    assert np.allclose(result.D, [[0.0]])

def test_linearize_nonlinear_pendulum():
    """Linearization of pendulum at equilibrium."""
    # Simple pendulum: d2theta/dt2 = -(g/L)*sin(theta)
    system = Pendulum(m=1.0, L=1.0, b=0.0, input_port=True)
    system.input_ports[0].fix_value(jnp.array([0.0]))
    
    base_context = system.create_context()
    # Upward position (unstable)
    context_up = base_context.with_continuous_state(jnp.array([jnp.pi, 0.0]))
    res_up = linearize(system, context_up)
    
    # Downward position (stable boundary / oscillatory)
    context_down = base_context.with_continuous_state(jnp.array([0.0, 0.0]))
    res_down = linearize(system, context_down)
    
    assert isinstance(res_up, LinearizedSystem)
    
    g = 9.81
    L = 1.0
    # Upward eigenvalues: +/- sqrt(g/L)
    evals_up = res_up.eigenvalues()
    assert np.allclose(np.sort(np.real(evals_up)), [-np.sqrt(g/L), np.sqrt(g/L)], atol=1e-3)
    
    # Downward eigenvalues: +/- j*sqrt(g/L)
    evals_down = res_down.eigenvalues()
    assert np.allclose(np.real(evals_down), [0.0, 0.0], atol=1e-3)
    assert np.allclose(np.sort(np.imag(evals_down)), [-np.sqrt(g/L), np.sqrt(g/L)], atol=1e-3)

def test_linearize_returns_linearized_system():
    """Return type is LinearizedSystem."""
    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]])
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    context = system.create_context()
    
    result = linearize(system, context)
    assert isinstance(result, LinearizedSystem)
    assert hasattr(result, 'A')
    assert hasattr(result, 'eigenvalues')

def test_is_stable():
    """Stable system detected correctly."""
    # Stable system
    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]])
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    context = system.create_context()
    result = linearize(system, context)
    assert result.is_stable()

    # Unstable system
    system2 = LTISystem(
        A=jnp.array([[1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]])
    )
    system2.input_ports[0].fix_value(jnp.array([0.0]))
    context2 = system2.create_context()
    result2 = linearize(system2, context2)
    assert not result2.is_stable()

def test_to_lti():
    """Can convert to LTISystem."""
    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]])
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    context = system.create_context()
    
    result = linearize(system, context)
    lti = result.to_lti()
    
    assert isinstance(lti, LTISystem)


def test_linearized_system_to_lti():
    """to_lti() returns a usable LTISystem block."""
    # Build a simple integrator as LTI: dx/dt = u, y = x
    # A=0, B=1, C=1, D=0
    system = LTISystem(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = system.create_context()

    lin = linearize(system, ctx)
    assert isinstance(lin, LinearizedSystem)

    lti = lin.to_lti()
    # LTI block should be usable — put it in a builder
    b2 = DiagramBuilder()
    b2.add(lti)
    d2 = b2.build()
    assert d2 is not None


# ---------------------------------------------------------------------------
# New tests for to_scipy_lti() and equilibrium check
# ---------------------------------------------------------------------------

scipy_signal = pytest.importorskip("scipy.signal", reason="scipy not installed")


def test_to_scipy_lti_returns_state_space_siso():
    """to_scipy_lti() returns scipy.signal.StateSpace (not lti/TransferFunction)."""
    from scipy import signal

    system = LTISystem(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = system.create_context()

    lin = linearize(system, ctx)
    ss = lin.to_scipy_lti()

    assert isinstance(ss, signal.StateSpace)
    assert np.allclose(ss.A, [[-2.0]])
    assert np.allclose(ss.B, [[1.0]])
    assert np.allclose(ss.C, [[1.0]])
    assert np.allclose(ss.D, [[0.0]])


def test_to_scipy_lti_mimo():
    """to_scipy_lti() works for a 2-input, 2-output state-space system."""
    from scipy import signal

    # 2-state, 2-input, 2-output system
    A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
    B = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    C = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    D = jnp.zeros((2, 2))

    system = LTISystem(A=A, B=B, C=C, D=D)
    u0 = jnp.zeros(2)
    system.input_ports[0].fix_value(u0)
    ctx = system.create_context()

    lin = linearize(system, ctx)
    ss = lin.to_scipy_lti()

    assert isinstance(ss, signal.StateSpace)
    assert ss.A.shape == (2, 2)
    assert ss.B.shape == (2, 2)
    assert ss.C.shape == (2, 2)
    assert ss.D.shape == (2, 2)


def test_to_scipy_lti_bode_callable():
    """scipy.signal.StateSpace returned by to_scipy_lti() is usable with bode()."""
    from scipy import signal

    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = system.create_context()

    ss = linearize(system, ctx).to_scipy_lti()
    w, mag, phase = signal.bode(ss, w=np.logspace(-1, 2, 50))
    assert len(w) == 50
    assert np.all(np.isfinite(mag))


def test_linearize_non_equilibrium_warns():
    """linearize() emits a UserWarning when ẋ(x₀, u₀) ≠ 0."""
    # A first-order system dx/dt = -x + u, evaluated at x0=1, u0=0
    # → ẋ = -1 + 0 = -1 ≠ 0
    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = system.create_context().with_continuous_state(jnp.array([1.0]))

    with pytest.warns(UserWarning, match="equilibrium"):
        lin = linearize(system, ctx)
    # Jacobians should still be returned
    assert isinstance(lin, LinearizedSystem)


def test_linearize_nan_state_raises():
    """linearize() raises ValueError when the base context contains NaN state."""
    system = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    system.input_ports[0].fix_value(jnp.array([0.0]))
    ctx = system.create_context().with_continuous_state(jnp.array([float("nan")]))

    with pytest.raises(ValueError, match="non-finite"):
        linearize(system, ctx)
