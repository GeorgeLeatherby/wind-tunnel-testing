"""
Constraint checking for generated operating points (Group-2 limits).

Each limit from the assignment is checked by its own small predicate so a single
constraint can be inspected or relaxed independently. `check_row` aggregates them
and returns the list of violated-constraint messages (empty list == all good).

These checks are used to *verify* the generated matrix; per the no-fallback rule
the generators do not silently skip or clamp operating points to satisfy limits,
they build the requested points and the verification reports any breach.
"""

from __future__ import annotations

from .campaign_config import Group2Constraints
from .operating_point import OperatingPointResult


def is_tsr_within(result: OperatingPointResult, constraints: Group2Constraints) -> bool:
    """TSR must lie within [tsr_min, tsr_max]."""
    return constraints.tsr_min <= result.tsr <= constraints.tsr_max


def is_power_within(result: OperatingPointResult, constraints: Group2Constraints) -> bool:
    """Power must lie within [0, power_max_w]."""
    return 0.0 <= result.power_w <= constraints.power_max_w


def is_torque_within(result: OperatingPointResult, constraints: Group2Constraints) -> bool:
    """Torque must lie within [0, torque_max_nm]."""
    return 0.0 <= result.torque_nm <= constraints.torque_max_nm


def is_rotor_speed_within(
    result: OperatingPointResult, constraints: Group2Constraints
) -> bool:
    """Rotor speed must not exceed rotor_speed_max_rpm."""
    return result.rotor_speed_rpm <= constraints.rotor_speed_max_rpm


def is_reynolds_within(
    result: OperatingPointResult, constraints: Group2Constraints, target_reynolds: float
) -> bool:
    """Reynolds must be within +/- tolerance of the target."""
    return abs(result.reynolds - target_reynolds) <= constraints.reynolds_tolerance


def check_row(
    result: OperatingPointResult,
    constraints: Group2Constraints,
    target_reynolds: float,
    enforce_reynolds: bool,
) -> list[str]:
    """
    Return a list of human-readable violation messages for one operating point.

    `enforce_reynolds` is True for the performance task (constant-Re curves) and
    False for the wake task (fixed speed, Reynolds only reported).
    """
    violations: list[str] = []
    if not is_tsr_within(result, constraints):
        violations.append(f"TSR {result.tsr:.3f} outside [{constraints.tsr_min}, {constraints.tsr_max}]")
    if not is_power_within(result, constraints):
        violations.append(f"Power {result.power_w:.2f} W outside [0, {constraints.power_max_w}]")
    if not is_torque_within(result, constraints):
        violations.append(f"Torque {result.torque_nm:.3f} Nm outside [0, {constraints.torque_max_nm}]")
    if not is_rotor_speed_within(result, constraints):
        violations.append(
            f"Rotor speed {result.rotor_speed_rpm:.1f} rpm exceeds {constraints.rotor_speed_max_rpm}"
        )
    if enforce_reynolds and not is_reynolds_within(result, constraints, target_reynolds):
        violations.append(
            f"Reynolds {result.reynolds:.0f} outside {target_reynolds} +/- {constraints.reynolds_tolerance}"
        )
    return violations
