# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
from jaxonomy.lazy_loader import LazyLoader

from .base import SymKind, EqnKind
from .component_base import ComponentBase

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
1D mechanical translational components similar to Modelica Standard Library.

flow variable:Units = force:Newtons
potential variable:Units = velocity:meters/second
"""


class TranslationalOnePort(ComponentBase):
    """Partial component class for a translational component with one flange."""

    def __init__(
        self,
        ev,
        name,
        v_ic=0.0,
        v_ic_fixed=False,
        x_ic=0.0,
        x_ic_fixed=False,
        p="flange",
    ):
        super().__init__()
        self.f, self.x, self.v, self.a = self.declare_translational_port(
            ev,
            p,
            v_ic=v_ic,
            v_ic_fixed=v_ic_fixed,
            x_ic=x_ic,
            x_ic_fixed=x_ic_fixed,
        )
        self.port_idx_to_name = {-1: p}


class TranslationalTwoPort(ComponentBase):
    """Partial component class for an translational component with two
    flanges that can translate relative to each other.
    """

    def __init__(
        self,
        ev,
        name,
        x1_ic=0.0,
        x1_ic_fixed=False,
        v1_ic=0.0,
        v1_ic_fixed=False,
        x2_ic=0.0,
        x2_ic_fixed=False,
        v2_ic=0.0,
        v2_ic_fixed=False,
        p1="flange_a",
        p2="flange_b",
        include_force_equality=True,
    ):
        super().__init__()
        self.f1, self.x1, self.v1, self.a1 = self.declare_translational_port(
            ev,
            p1,
            v_ic=v1_ic,
            v_ic_fixed=v1_ic_fixed,
            x_ic=x1_ic,
            x_ic_fixed=x1_ic_fixed,
        )
        self.f2, self.x2, self.v2, self.a2 = self.declare_translational_port(
            ev,
            p2,
            v_ic=v2_ic,
            v_ic_fixed=v2_ic_fixed,
            x_ic=x2_ic,
            x_ic_fixed=x2_ic_fixed,
        )
        if include_force_equality:
            self.add_eqs([sp.Eq(0, self.f1.s + self.f2.s)])

        self.port_idx_to_name = {-1: p1, 1: p2}


class Damper(TranslationalTwoPort):
    """
    Ideal damper in translational domain. The characteristic equation is:
    force(t) = D*(v1(t) - v2(t)), where D is the damping coefficient in N/(m/s).

    Agrs:
        D (number):
            The damping coefficient.
        initial_velocity_A (number);
            initial velocity of flange_a.
        initial_position_A (number);
            initial position of flange_a.
        initial_velocity_B (number);
            initial velocity of flange_b.
        initial_position_B (number);
            initial position of flange_b.
    """

    def __init__(
        self,
        ev,
        name=None,
        D=1.0,
        initial_velocity_A=0.0,
        initial_velocity_A_fixed=False,
        initial_position_A=0.0,
        initial_position_A_fixed=False,
        initial_velocity_B=0.0,
        initial_velocity_B_fixed=False,
        initial_position_B=0.0,
        initial_position_B_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            v1_ic=initial_velocity_A,
            v1_ic_fixed=initial_velocity_A_fixed,
            x1_ic=initial_position_A,
            x1_ic_fixed=initial_position_A_fixed,
            v2_ic=initial_velocity_B,
            v2_ic_fixed=initial_velocity_B_fixed,
            x2_ic=initial_position_B,
            x2_ic_fixed=initial_position_B_fixed,
        )

        d = self.declare_symbol(
            ev,
            "D",
            self.name,
            kind=SymKind.param,
            val=D,
            validator=lambda D: D > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have D>0",
        )
        self.add_eqs([sp.Eq(self.f1.s, d.s * (self.v1.s - self.v2.s))])


class FixedPosition(TranslationalOnePort):
    """
    Rigid(non-moving) reference in mechanical translational domain.

    Agrs:
        initial_position (number);
            initial position of flange.
    """

    def __init__(self, ev, name=None, initial_position=0.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name, v_ic=0, x_ic=initial_position, x_ic_fixed=True)
        # not ideal to create a dummy parameter symbol in order to set the x symbol
        # to this fixed value, but this works for now. using 'initial_position' in the
        # IC equation below was tried, this results in equations processing errors. not
        # that these cannot be over come, but it's not a priority when the dummy param
        # solution works.
        self.add_eqs([sp.Eq(0, self.a.s), sp.Eq(0, self.v.s)])
        x_ic_param = self.declare_symbol(
            ev, "x_ic_param", self.name, kind=SymKind.param, val=initial_position
        )
        # IC equation
        self.add_eqs([sp.Eq(x_ic_param.s, self.x.s)])

        # this sucks. need this here because not all one port blocks
        # use the 'input' of the block.
        self.port_idx_to_name = {1: "flange"}


class ForceSensor(TranslationalTwoPort):
    """.
    Ideal force sensor in translational domain.
    Measures force between flange_a and flange_b.
    Tension is positive.
    """

    def __init__(
        self,
        ev,
        name=None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        f = self.declare_symbol(ev, "f", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(f.s, self.f1.s), kind=EqnKind.outp)
        self.add_eqs(
            [
                sp.Eq(self.x1.s, self.x2.s),
                sp.Eq(self.v1.s, self.v2.s),
                sp.Eq(self.a1.s, self.a2.s),
            ]
        )


class ForceSource(ComponentBase):
    """
    Ideal force source in translational domain.

    Args:
        f (number):
            Force value when enable_force_port=False.
        enable_force_port (bool):
            When true, the force value is from a input signal. When false,
            force value is from 'f'.
        enable_flange_b (bool):
            When flange_b enabled, applies force between flange_a and flange_b.
            when flange_b disbaled, applies absolute toqrue to flange_a.
    """

    def __init__(
        self,
        ev,
        name=None,
        f=0.0,
        enable_force_port=False,
        enable_flange_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        f1, x1, v1, a1 = self.declare_translational_port(ev, "flange_a")
        self.port_idx_to_name = {-1: "flange_a"}

        if enable_force_port:
            f = self.declare_symbol(ev, "f", self.name, kind=SymKind.inp)
        else:
            f = self.declare_symbol(ev, "f", self.name, kind=SymKind.param, val=f)

        self.add_eqs([sp.Eq(f.s, -f1.s)])
        if enable_flange_b:
            f2, x2, v2, a2 = self.declare_translational_port(ev, "flange_b")
            self.port_idx_to_name[1] = "flange_b"
            self.add_eqs([sp.Eq(f.s, f2.s)])


class Friction(TranslationalTwoPort):
    """
    Coulomb, Viscous and Stribeck friction in translational domain. Coulomb friction is always
    modeled, while Viscous and Stribeck can be optionally enabled as detailed below.

    In it's simplest form, Fbrk=None and C=None, the characteristic equation is:
        f = Fc*tanh(v/(Vbrk/10))

    With only Fbrk=None, the viscous term is included, and the characteristic equation is:
        f = Fc*tanh(v/(Vbrk/10)) +
            C*v

    With all friction sources enabled, the characteristic equation is:
        f = Fc*tanh(v/(Vbrk/10)) +
            C*v +
            sqrt(2*e)*(Fbrk - Fc)*exp(-(v/(Vbrk*sqrt(2)))**2)*(v/(Vbrk*sqrt(2)))

    where:
        f:v are relative force and velocity across the component ports
        Fbrk is break away friction
        Fc is Coulomb friction
        Vbrk is the break away friction velocity threshold
        C is the viscous friction coefficient

    Agrs:
        <blah> (number):
            The <blah> coefficient.
    """

    def __init__(self, ev, name=None, Fc=1.0, Vbrk=0.1, C=None, Fbrk=None):
        self.name = self.__class__.__name__ if name is None else name
        if Fc is None:
            Fc = 1.0
        if Vbrk is None:
            Vbrk = 0.1
        super().__init__(ev, self.name)

        Vbrk = self.declare_symbol(
            ev,
            "Vbrk",
            self.name,
            kind=SymKind.param,
            val=Vbrk,
            validator=lambda Vbrk: Vbrk > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Vbrk>0",
        )
        Fc = self.declare_symbol(
            ev,
            "Fc",
            self.name,
            kind=SymKind.param,
            val=Fc,
            validator=lambda Fc: Fc > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Fc>0",
        )
        frc_coul = self.declare_symbol(ev, "frc_coul", self.name, kind=SymKind.var)
        Vrel = self.declare_symbol(ev, "Vrel", self.name, kind=SymKind.var)
        self.add_eqs(
            [
                sp.Eq(Vrel.s, self.v1.s - self.v2.s),
                sp.Eq(frc_coul.s, Fc.s * sp.tanh(Vrel.s / (Vbrk.s / 10.0))),
            ]
        )
        frc_total = frc_coul.s

        if C is not None:
            # include viscous friction
            C = self.declare_symbol(
                ev,
                "C",
                self.name,
                kind=SymKind.param,
                val=C,
                validator=lambda C: C > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have C>0",
            )
            frc_total += C.s * Vrel.s

        if Fbrk is not None:
            Fbrk = self.declare_symbol(
                ev,
                "Fbrk",
                self.name,
                kind=SymKind.param,
                val=Fbrk,
                validator=lambda Fbrk: Fbrk >= Fc.val,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Fbrk>=Fc",
            )
            sqrt_2 = sp.sqrt(2.0)
            v_scaled = Vrel.s / (Vbrk.s * sqrt_2)
            stribeck_term = (
                sqrt_2
                * sp.sqrt(sp.E)
                * (Fbrk.s - Fc.s)
                * sp.exp(-(v_scaled**2))
                * v_scaled
            )
            frc_total += stribeck_term

        self.add_eqs([sp.Eq(self.f1.s, frc_total)])


class Mass(TranslationalOnePort):
    """
    Ideal mass in translational domain. The characteristic equation is:
    force(t) = M*a(t), where a=derivative(v(t)) and M is mass in kg.
    Agrs:
        M (number):
            The mass.
        initial_velocity (number);
            initial velocity of flange.
        initial_position (number);
            initial position of flange.
    """

    def __init__(
        self,
        ev,
        name=None,
        M=1.0,
        initial_velocity=0.0,
        initial_velocity_fixed=False,
        initial_position=0.0,
        initial_position_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            v_ic=initial_velocity,
            v_ic_fixed=initial_velocity_fixed,
            x_ic=initial_position,
            x_ic_fixed=initial_position_fixed,
        )

        m = self.declare_symbol(
            ev,
            "M",
            self.name,
            kind=SymKind.param,
            val=M,
            validator=lambda M: M > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have M>0",
        )
        self.add_eqs([sp.Eq(self.f.s, m.s * self.a.s)])


class MotionSensor(ComponentBase):
    """
    Ideal speed sensor in the translational domain.

    Agrs:
        enable_flange_b(bool):
            When flange_b enabled, measures between flange_a and flange_b.
            When flange_b disbaled, measures the absolute speed.
    """

    def __init__(
        self,
        ev,
        name=None,
        enable_flange_b=True,
        enable_position_port=False,
        enable_velocity_port=True,
        enable_acceleration_port=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        if not (
            enable_position_port or enable_velocity_port or enable_acceleration_port
        ):
            raise ValueError(
                f"SpeedSensor {self.name} must have one causal outport enabled."
            )

        super().__init__()
        f1, x1, v1, a1 = self.declare_translational_port(ev, "flange_a")
        self.port_idx_to_name = {-1: "flange_a"}

        if enable_position_port:
            x_rel = self.declare_symbol(ev, "x_rel", self.name, kind=SymKind.outp)
        if enable_velocity_port:
            v_rel = self.declare_symbol(ev, "v_rel", self.name, kind=SymKind.outp)
        if enable_acceleration_port:
            a_rel = self.declare_symbol(ev, "a_rel", self.name, kind=SymKind.outp)

        if enable_flange_b:
            f2, x2, v2, a2 = self.declare_translational_port(ev, "flange_b")
            self.port_idx_to_name[1] = "flange_b"

            self.add_eqs([sp.Eq(f1.s, 0), sp.Eq(f2.s, 0)])
            if enable_position_port:
                self.declare_equation(sp.Eq(x_rel.s, x1.s - x2.s), kind=EqnKind.outp)
            if enable_velocity_port:
                self.declare_equation(sp.Eq(v_rel.s, v1.s - v2.s), kind=EqnKind.outp)
            if enable_acceleration_port:
                self.declare_equation(sp.Eq(a_rel.s, a1.s - a2.s), kind=EqnKind.outp)
        else:
            self.add_eqs([sp.Eq(f1.s, 0)])
            if enable_position_port:
                self.declare_equation(sp.Eq(x_rel.s, x1.s), kind=EqnKind.outp)
            if enable_velocity_port:
                self.declare_equation(sp.Eq(v_rel.s, v1.s), kind=EqnKind.outp)
            if enable_acceleration_port:
                self.declare_equation(sp.Eq(a_rel.s, a1.s), kind=EqnKind.outp)


class SpeedSource(ComponentBase):
    """
    Ideal translational speed source.

    Args:
        v_ref (number):
            Speed value when enable_speed_port=False.
        enable_speed_port (bool):
            When true, the speed value is from a input signal. When false, speed
            value is from w_ref.
        enable_flange_b (bool):
            When flange_b enabled, applies speed between flange_a and flange_b.
            When flange_b disbaled, applies absolute speed to flange_a.
    """

    def __init__(
        self,
        ev,
        name=None,
        v_ref=0.0,
        enable_speed_port=False,
        enable_flange_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        f1, x1, v1, a1 = self.declare_translational_port(ev, "flange_a")
        self.port_idx_to_name = {-1: "flange_a"}

        if enable_speed_port:
            v_ref = self.declare_symbol(ev, "v_ref", self.name, kind=SymKind.inp)
        else:
            v_ref = self.declare_symbol(
                ev, "v_ref", self.name, kind=SymKind.param, val=v_ref
            )

        if enable_flange_b:
            f2, x2, v2, a2 = self.declare_translational_port(ev, "flange_b")
            self.port_idx_to_name[1] = "flange_b"
            self.add_eqs(
                [
                    # source does not store translational momentum
                    sp.Eq(0, f1.s + f2.s),
                    # prescribed relative speed
                    sp.Eq(v1.s - v2.s, v_ref.s),
                ]
            )
        else:
            # prescribed absolute speed
            self.add_eqs([sp.Eq(v1.s, v_ref.s)])


class Spring(TranslationalTwoPort):
    """
    Ideal spring in translational domain. The characteristic equation is:
    force(t) = K*(x1(t) - x2(t)), where K is the spring constant in N/m.

    Agrs:
        K (number):
            The stiffness of the spring.
        initial_velocity_A (number);
            initial velocity of flange_a.
        initial_position_A (number);
            initial position of flange_a.
        initial_velocity_B (number);
            initial velocity of flange_b.
        initial_position_B (number);
            initial position of flange_b.

    """

    def __init__(
        self,
        ev,
        name=None,
        K=1.0,
        initial_velocity_A=0.0,
        initial_velocity_A_fixed=False,
        initial_position_A=0.0,
        initial_position_A_fixed=False,
        initial_velocity_B=0.0,
        initial_velocity_B_fixed=False,
        initial_position_B=0.0,
        initial_position_B_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            v1_ic=initial_velocity_A,
            v1_ic_fixed=initial_velocity_A_fixed,
            x1_ic=initial_position_A,
            x1_ic_fixed=initial_position_A_fixed,
            v2_ic=initial_velocity_B,
            v2_ic_fixed=initial_velocity_B_fixed,
            x2_ic=initial_position_B,
            x2_ic_fixed=initial_position_B_fixed,
        )

        k = self.declare_symbol(
            ev,
            "K",
            self.name,
            kind=SymKind.param,
            val=K,
            validator=lambda K: K > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have K>0",
        )
        self.add_eqs([sp.Eq(self.f1.s, k.s * (self.x1.s - self.x2.s))])


class HardStop(TranslationalTwoPort):
    """
    Unilateral translational contact (hard stop) with linear spring/damper contact force.
    Contact engages when x1 - x2 >= gap.
    """

    def __init__(self, ev, name=None, gap=0.0, K=1e5, D=100.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        gap = self.declare_symbol(ev, "gap", self.name, kind=SymKind.param, val=gap)
        K = self.declare_symbol(
            ev,
            "K",
            self.name,
            kind=SymKind.param,
            val=K,
            validator=lambda x: x > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have K>0",
        )
        D = self.declare_symbol(
            ev,
            "D",
            self.name,
            kind=SymKind.param,
            val=D,
            validator=lambda x: x >= 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have D>=0",
        )
        penetration = self.declare_symbol(ev, "penetration", self.name, kind=SymKind.var)
        vrel = self.declare_symbol(ev, "vrel", self.name, kind=SymKind.var)
        f_contact = self.declare_symbol(ev, "f_contact", self.name, kind=SymKind.var)
        f_expr = self.declare_conditional(
            ev,
            penetration.s >= 0.0,
            K.s * penetration.s + D.s * vrel.s,
            0.0,
            cond_name="contact_force",
            non_bool_zc_expr=penetration.s,
        )
        self.add_eqs(
            [
                sp.Eq(penetration.s, self.x1.s - self.x2.s - gap.s),
                sp.Eq(vrel.s, self.v1.s - self.v2.s),
                sp.Eq(f_contact.s, f_expr.s),
                sp.Eq(self.f1.s, f_contact.s),
            ]
        )
