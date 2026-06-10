"""
Dimensional rotor performance: convert non-dimensional coefficients into the
physical power [W], thrust [N] and torque [Nm] that the constraints are stated in.

All relations are the standard rotor-performance definitions used in the lecture
("Performance Off the Design Point"):

        P = 0.5 * rho * A * U^3 * C_P
        T = 0.5 * rho * A * U^2 * C_T
        Q = P / omega                      (torque = power / angular speed)

`A` is the rotor disk area and `U` is the flow speed seen by the rotor (the
equivalent free-air speed). One formula per function.
"""

from __future__ import annotations

import math


def rotor_disk_area(tip_radius: float) -> float:
    """Rotor swept area A = pi * R^2 (single formula)."""
    return math.pi * tip_radius * tip_radius


def aerodynamic_power(
    air_density: float,
    rotor_area: float,
    flow_speed: float,
    power_coefficient: float,
) -> float:
    """Aerodynamic power  P = 0.5 * rho * A * U^3 * C_P  [W]."""
    return 0.5 * air_density * rotor_area * flow_speed**3 * power_coefficient


def aerodynamic_thrust(
    air_density: float,
    rotor_area: float,
    flow_speed: float,
    thrust_coefficient: float,
) -> float:
    """Aerodynamic thrust  T = 0.5 * rho * A * U^2 * C_T  [N]."""
    return 0.5 * air_density * rotor_area * flow_speed**2 * thrust_coefficient


def aerodynamic_torque(power: float, omega: float) -> float:
    """
    Aerodynamic torque from power and angular speed:  Q = P / omega  [Nm].

    `omega` must be in [rad/s]; passing zero will raise ZeroDivisionError on
    purpose (a zero-speed rotor has no defined torque from this relation).
    """
    return power / omega
