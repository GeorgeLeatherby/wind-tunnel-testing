"""
Scaling report: apply the Lecture-1 scaling laws to relate the G1 model to its
full-scale reference (Samsung S7.0-171).

The scaling laws combine two base factors (length scale n_L and rotor-speed /
time scale n_T) into derived factors for velocity, force, power, torque and
bending stiffness. This module computes those factors for the G1 model and
exposes them as labelled rows, so they can be printed and written into the Excel
workbooks (a compact "Scaling" sheet) — i.e. the scaling concept is actually
applied to the deliverable, not just defined.

n_L = R_model / R_full              (length scale)
n_T = omega_model / omega_full      (rotor-speed / time scale)
"""

from __future__ import annotations

from dataclasses import dataclass

from .scaling import (
    FullScaleTurbine,
    bending_stiffness_scale,
    force_scale,
    length_scale,
    power_scale,
    rotor_speed_scale,
    torque_scale,
    velocity_scale,
)


@dataclass(frozen=True)
class ScalingFactorRow:
    """One labelled scaling factor for the report."""

    quantity: str
    combination: str  # symbolic combination, e.g. "n_L^5 * n_T^3"
    value: float


class ScalingReport:
    """
    Computes the full set of model<->full-scale factors for the G1 model.

    `model_rotor_speed_rpm` is the representative model rotor speed used to define
    the time scale (the optimum operating rotor speed is a sensible choice).
    """

    def __init__(
        self,
        full_scale: FullScaleTurbine,
        model_rotor_diameter: float,
        model_rotor_speed_rpm: float,
    ) -> None:
        self._n_l = length_scale(model_rotor_diameter, full_scale.rotor_diameter)
        self._n_t = rotor_speed_scale(
            model_rotor_speed_rpm, full_scale.rotor_speed_rpm
        )

    def length_scale_value(self) -> float:
        """Base length scale n_L."""
        return self._n_l

    def time_scale_value(self) -> float:
        """Base rotor-speed / time scale n_T."""
        return self._n_t

    def rows(self) -> list[ScalingFactorRow]:
        """Return every scaling factor as a labelled row (model / full-scale)."""
        return [
            ScalingFactorRow("Length", "n_L", self._n_l),
            ScalingFactorRow("Rotor speed", "n_T", self._n_t),
            ScalingFactorRow(
                "Velocity", "n_L * n_T", velocity_scale(self._n_l, self._n_t)
            ),
            ScalingFactorRow(
                "Force", "n_L^4 * n_T^2", force_scale(self._n_l, self._n_t)
            ),
            ScalingFactorRow(
                "Power", "n_L^5 * n_T^3", power_scale(self._n_l, self._n_t)
            ),
            ScalingFactorRow(
                "Torque", "n_L^5 * n_T^2", torque_scale(self._n_l, self._n_t)
            ),
            ScalingFactorRow(
                "Bending stiffness",
                "n_L^6 * n_T^2",
                bending_stiffness_scale(self._n_l, self._n_t),
            ),
        ]
