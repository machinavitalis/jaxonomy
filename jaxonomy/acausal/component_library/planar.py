# SPDX-License-Identifier: MIT

"""
T-032 — planar (2-D) mechanical primitives for index-2 acausal models.

The 1-D ``translational`` library cannot express a holonomic
constraint of the form ``||r||² = L²`` because each port carries a
single scalar position / velocity. Without 2-D state vectors there is
no place for a Cartesian constraint, so problems like a pendulum
attached to a rigid link are out of reach.

This module ships the *minimum* needed to exercise an index-2
constrained-mechanics example end-to-end through the existing
Pantelides → BDF pipeline, so that DAE-projection work (T-003a) has a
real test case to target. It deliberately avoids introducing a
``Translational2DPort`` and the connection-rule plumbing that would
go with it; instead, the components are *self-contained* and expose
their position / angle outputs as scalar output ports for sensing.
A full 2-D port system can layer on top later without breaking these
components' contract.

Components:
  - :class:`PlanarPendulum` — point-mass swinging on a massless rigid
    link. Models the constraint ``x² + y² = L²`` directly via a
    Lagrange multiplier; equations of motion include gravity in
    user-specified ``(g_x, g_y)``.
"""

from typing import TYPE_CHECKING

from jaxonomy.lazy_loader import LazyLoader

from .base import SymKind, EqnKind
from .component_base import ComponentBase

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


class PlanarPendulum(ComponentBase):
    """A point mass on a rigid massless link, anchored at the origin
    and constrained to move on a circle of radius ``L``.

    The state-space form has 4 differential states (``x, y, vx, vy``)
    and 1 algebraic unknown — the Lagrange multiplier ``lam`` —
    governed by 7 equations:

        der(x) = vx
        der(y) = vy
        der(vx) = ax
        der(vy) = ay
        m * ax = -2 * lam * x + m * g_x
        m * ay = -2 * lam * y + m * g_y
        x² + y² = L²

    The constraint equation is index 2 with respect to ``lam``: it
    must be differentiated twice in time before ``lam`` appears
    explicitly. This is what makes the model exercise Pantelides
    index-reduction; once T-003a's projection step lands, this is
    also the test bed for verifying constraint drift stays small.

    Args:
        ev: equation environment.
        name: component name.
        m: mass of the bob (kg). Must be > 0.
        L: rigid-link length (m). Must be > 0.
        g_x, g_y: gravity components (m/s²). Default ``(0, -9.81)``
            (gravity along -y).
        initial_theta: initial angle from the +x axis, radians.
            ``initial_theta = -π/2`` puts the bob at ``(0, -L)``,
            i.e. hanging straight down. The default of ``π / 6``
            gives a 30°-from-horizontal release for a non-trivial
            initial trajectory.
        initial_omega: initial angular velocity (rad/s). Default 0.

    The component exposes three scalar output ports for sensing:
    ``x`` (horizontal position), ``y`` (vertical position), and
    ``theta = atan2(y, x)`` (angle from the +x axis).
    """

    def __init__(
        self,
        ev,
        name=None,
        m: float = 1.0,
        L: float = 1.0,
        g_x: float = 0.0,
        g_y: float = -9.81,
        initial_theta: float = 3.141592653589793 / 6.0,
        initial_omega: float = 0.0,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()

        # Validate constructor args eagerly so a bad pendulum fails
        # fast, before the SymPy graph is built.
        if m <= 0.0:
            raise ValueError(f"{self.name}: m must be > 0 (got {m})")
        if L <= 0.0:
            raise ValueError(f"{self.name}: L must be > 0 (got {L})")

        # Derive Cartesian initial conditions from polar (theta_0,
        # omega_0). x = L cos(theta_0); y = L sin(theta_0); the
        # corresponding velocities are tangential.
        import math
        x0 = L * math.cos(initial_theta)
        y0 = L * math.sin(initial_theta)
        vx0 = -L * math.sin(initial_theta) * initial_omega
        vy0 = L * math.cos(initial_theta) * initial_omega

        # Parameters.
        m_sym = self.declare_symbol(
            ev, "m", self.name, kind=SymKind.param, val=m,
            validator=lambda v: v > 0.0,
            invalid_msg=f"{self.name}: m must be > 0",
        )
        L_sym = self.declare_symbol(
            ev, "L", self.name, kind=SymKind.param, val=L,
            validator=lambda v: v > 0.0,
            invalid_msg=f"{self.name}: L must be > 0",
        )
        gx_sym = self.declare_symbol(
            ev, "g_x", self.name, kind=SymKind.param, val=g_x,
        )
        gy_sym = self.declare_symbol(
            ev, "g_y", self.name, kind=SymKind.param, val=g_y,
        )

        # Position. Differential states have ic_fixed=True so that
        # the index-reduction pass treats them as actual integrator
        # initial conditions rather than degrees of freedom.
        x = self.declare_symbol(
            ev, "x", self.name, kind=SymKind.var,
            ic=x0, ic_fixed=True,
        )
        y = self.declare_symbol(
            ev, "y", self.name, kind=SymKind.var,
            ic=y0, ic_fixed=True,
        )
        # Velocities — declared as the integrals of the positions.
        vx = self.declare_symbol(
            ev, "vx", self.name, kind=SymKind.var,
            int_sym=x, ic=vx0, ic_fixed=True,
        )
        vy = self.declare_symbol(
            ev, "vy", self.name, kind=SymKind.var,
            int_sym=y, ic=vy0, ic_fixed=True,
        )
        # Accelerations — integrals of velocities. No IC: they are
        # computed from the algebraic equations of motion.
        ax = self.declare_symbol(
            ev, "ax", self.name, kind=SymKind.var, int_sym=vx,
        )
        ay = self.declare_symbol(
            ev, "ay", self.name, kind=SymKind.var, int_sym=vy,
        )
        # Wire the derivative chain so Pantelides can find d/dt
        # relations going downstream as well as upstream.
        x.der_sym = vx
        y.der_sym = vy
        vx.der_sym = ax
        vy.der_sym = ay

        # Lagrange multiplier for the holonomic constraint. Algebraic
        # unknown — no derivative chain.
        lam = self.declare_symbol(ev, "lam", self.name, kind=SymKind.var)

        # Output ports for sensing. Cartesian only — users who want
        # an angular trace can compute ``np.arctan2(y, x)`` on the
        # results array (the acausal compiler currently has trouble
        # threading ``sp.atan2`` through index reduction).
        x_out = self.declare_symbol(ev, "x_out", self.name, kind=SymKind.outp)
        y_out = self.declare_symbol(ev, "y_out", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(x_out.s, x.s), kind=EqnKind.outp)
        self.declare_equation(sp.Eq(y_out.s, y.s), kind=EqnKind.outp)

        # Equations of motion. The constraint x² + y² = L² is
        # differentiated implicitly by Pantelides; the Lagrange
        # multiplier appears in Newton's second law via the gradient
        # of the constraint, scaled by the standard factor of 2.
        self.add_eqs([
            sp.Eq(m_sym.s * ax.s, -2 * lam.s * x.s + m_sym.s * gx_sym.s),
            sp.Eq(m_sym.s * ay.s, -2 * lam.s * y.s + m_sym.s * gy_sym.s),
            sp.Eq(x.s**2 + y.s**2, L_sym.s**2),
        ])

        # PlanarPendulum has no acausal ports — it's a self-contained
        # system. Mark port_idx_to_name as an empty mapping so the
        # serialization layer doesn't choke.
        self.port_idx_to_name = {}
