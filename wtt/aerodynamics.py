"""
Core aerodynamic formulas, each isolated in its own function.

Concepts implemented here:
    * velocity triangle at a blade section  -> relative velocity v_rel
    * Reynolds number from v_rel and chord
    * conversions between tip-speed-ratio, rotor angular speed and RPM

Keeping every formula in a separate function means a single equation can be
checked or corrected without touching anything else.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Tip-speed-ratio  <->  rotor speed
# ---------------------------------------------------------------------------
def omega_from_tsr(tsr: float, free_stream_speed: float, tip_radius: float) -> float:
    """
    Rotor angular speed from the TSR definition  lambda = omega * R / U.

    Solved for omega:  omega = lambda * U / R.
    Units: returns [rad / s].
    """
    return tsr * free_stream_speed / tip_radius


def tsr_from_omega(omega: float, free_stream_speed: float, tip_radius: float) -> float:
    """Inverse of `omega_from_tsr`:  lambda = omega * R / U."""
    return omega * tip_radius / free_stream_speed


def rpm_from_omega(omega: float) -> float:
    """Convert angular speed [rad/s] to revolutions per minute [rpm]."""
    return omega * 60.0 / (2.0 * math.pi)


def omega_from_rpm(rpm: float) -> float:
    """Convert revolutions per minute [rpm] to angular speed [rad/s]."""
    return rpm * 2.0 * math.pi / 60.0


# ---------------------------------------------------------------------------
# Velocity triangle at a blade section
# ---------------------------------------------------------------------------
def axial_velocity_component(
    free_stream_speed: float, axial_induction: float
) -> float:
    """
    Axial component of the local flow speed at the rotor plane.

    u_axial = U_inf * (1 - a)
    where `a` is the axial induction factor.
    """
    return free_stream_speed * (1.0 - axial_induction)


def tangential_velocity_component(
    tsr: float,
    free_stream_speed: float,
    tip_radius: float,
    section_radius: float,
    tangential_induction: float,
) -> float:
    """
    Tangential component of the local flow speed seen by a blade section.

    From the velocity triangle the blade-section tangential speed is the local
    rotational speed augmented by wake swirl:

        u_tangential = (lambda * U_inf / R) * r * (1 + a')

    where (lambda * U_inf / R) = omega, r is the section radius and a' is the
    tangential (angular) induction factor.
    """
    omega = tsr * free_stream_speed / tip_radius
    return omega * section_radius * (1.0 + tangential_induction)


def relative_velocity(
    tsr: float,
    free_stream_speed: float,
    tip_radius: float,
    section_radius: float,
    axial_induction: float,
    tangential_induction: float,
) -> float:
    """
    Magnitude of the relative velocity at a blade section (velocity triangle).

    v_rel = sqrt( (U_inf (1 - a))^2 + ( (lambda U_inf / R) r (1 + a') )^2 )

    This is exactly the expression given in the assignment, with r the midspan
    (reference) radius.
    """
    u_axial = axial_velocity_component(free_stream_speed, axial_induction)
    u_tangential = tangential_velocity_component(
        tsr=tsr,
        free_stream_speed=free_stream_speed,
        tip_radius=tip_radius,
        section_radius=section_radius,
        tangential_induction=tangential_induction,
    )
    return math.sqrt(u_axial * u_axial + u_tangential * u_tangential)


# ---------------------------------------------------------------------------
# Reynolds number
# ---------------------------------------------------------------------------
def reynolds_number(
    relative_speed: float, chord: float, kinematic_viscosity: float
) -> float:
    """
    Reynolds number convention used in the assignment:

        Re = (v_rel * chord) / nu

    evaluated with the reference (midspan) chord and the reference relative speed.
    """
    return relative_speed * chord / kinematic_viscosity
