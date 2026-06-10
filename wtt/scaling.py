"""
Scaling laws between the full-scale turbine and the scaled wind-tunnel model.

These are the relations from Lecture 1 ("Turbine Scaling"), which combine the
two base scale factors:

        n_L = model length / full-scale length        (length scale)
        n_T = model rotor speed / full-scale speed     (time / rotor-speed scale)

into derived scale factors for the quantities of interest:

    Quantity            Scale factor
    -----------------   -----------------
    Velocity            n_L * n_T
    Force               n_L^4 * n_T^2
    Power               n_L^5 * n_T^3
    Torque              n_L^5 * n_T^2
    Bending stiffness   n_L^6 * n_T^2

The TSR-similarity argument (lambda_model = lambda_full) fixes the model rotor
speed from the length scale and the chosen velocity scale; here we expose the
scale factors as pure functions plus a small dataclass to bundle the reference
full-scale machine. One formula per function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FullScaleTurbine:
    """
    Reference full-scale machine that the G1 model represents.

    Defaults are intentionally NOT provided; the concrete machine (the Samsung
    S7.0-171 used in the G1 lecture) is created in `samsung_s7_reference()`.
    """

    rotor_diameter: float  # [m]
    rotor_speed_rpm: float  # [rpm]
    rated_power: float  # [W]


def length_scale(model_length: float, full_scale_length: float) -> float:
    """Length scale  n_L = model_length / full_scale_length (single formula)."""
    return model_length / full_scale_length


def rotor_speed_scale(model_rotor_speed: float, full_scale_rotor_speed: float) -> float:
    """Rotor-speed (time) scale  n_T = model_speed / full_scale_speed."""
    return model_rotor_speed / full_scale_rotor_speed


def velocity_scale(length_scale_value: float, time_scale_value: float) -> float:
    """Velocity scale = n_L * n_T."""
    return length_scale_value * time_scale_value


def force_scale(length_scale_value: float, time_scale_value: float) -> float:
    """Force scale = n_L^4 * n_T^2."""
    return length_scale_value**4 * time_scale_value**2


def power_scale(length_scale_value: float, time_scale_value: float) -> float:
    """Power scale = n_L^5 * n_T^3."""
    return length_scale_value**5 * time_scale_value**3


def torque_scale(length_scale_value: float, time_scale_value: float) -> float:
    """Torque scale = n_L^5 * n_T^2."""
    return length_scale_value**5 * time_scale_value**2


def bending_stiffness_scale(length_scale_value: float, time_scale_value: float) -> float:
    """Bending-stiffness scale = n_L^6 * n_T^2."""
    return length_scale_value**6 * time_scale_value**2


def samsung_s7_reference() -> FullScaleTurbine:
    """
    Full-scale reference for the G1 model (Samsung S7.0-171), from Lecture 2.

    Rotor diameter 171.2 m, rotor speed 10.4 rpm, rated power 7.70e6 W.
    """
    return FullScaleTurbine(
        rotor_diameter=171.2,  # [m]
        rotor_speed_rpm=10.4,  # [rpm]
        rated_power=7.70e6,  # [W]
    )
