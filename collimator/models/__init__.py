# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from .pendulum import Pendulum, PendulumDiagram, animate_pendulum
from .lotka_volterra import LotkaVolterra
from .fitzhugh_nagumo import FitzHughNagumo
from .van_der_pol import VanDerPol
from .bouncing_ball import BouncingBall
from .planar_quadrotor import PlanarQuadrotor, animate_planar_quadrotor
from .rimless_wheel import RimlessWheel
from .compass_gait import CompassGait
from .compact_ev import CompactEV, DummyBlock
from .acrobot import Acrobot, animate_acrobot
from .cartpole import CartPole, animate_cartpole
from .battery_ecm import Battery
from .hairer import (
    EulerRigidBody,
    ArenstorfOrbit,
    Lorenz,
    Pleiades,
)

__all__ = [
    "Pendulum",
    "PendulumDiagram",
    "animate_pendulum",
    "LotkaVolterra",
    "FitzHughNagumo",
    "VanDerPol",
    "BouncingBall",
    "PlanarQuadrotor",
    "animate_planar_quadrotor",
    "RimlessWheel",
    "CompassGait",
    "CompactEV",
    "DummyBlock",
    "Acrobot",
    "animate_acrobot",
    "CartPole",
    "animate_cartpole",
    "Battery",
    "EulerRigidBody",
    "ArenstorfOrbit",
    "Lorenz",
    "Pleiades",
]
