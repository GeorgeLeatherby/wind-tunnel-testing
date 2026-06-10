"""
Reynolds-number calibration.

Goal: for a given operating point (TSR and blade pitch), find the equivalent
free-air wind speed U'_inf such that the midspan Reynolds number equals the
target value (75000).

The Reynolds number depends on the relative velocity at the midspan section,
which itself depends linearly on U'_inf (the velocity-triangle components both
scale with U'_inf at fixed TSR and induction factors). Because of this linearity
the calibration could be solved analytically, but we solve it with a robust
bracketed root finder (`brentq`) so the same routine keeps working unchanged if a
more complex (non-linear) velocity model is plugged in later.

No fallbacks: if the target Reynolds cannot be bracketed in the search interval,
`brentq` raises and the failure is visible.
"""

from __future__ import annotations

from scipy.optimize import brentq

from .aerodynamics import reynolds_number, relative_velocity


class ReynoldsCalibrator:
    """
    Solves for the free-air speed that yields a target midspan Reynolds number.

    The induction factors, geometry and viscosity are fixed at construction so
    that the residual used by the root finder is a clean function of speed only.
    """

    def __init__(
        self,
        tip_radius: float,
        midspan_radius: float,
        midspan_chord: float,
        kinematic_viscosity: float,
        search_speed_min: float,
        search_speed_max: float,
    ) -> None:
        self._tip_radius = tip_radius
        self._midspan_radius = midspan_radius
        self._midspan_chord = midspan_chord
        self._kinematic_viscosity = kinematic_viscosity
        self._search_speed_min = search_speed_min
        self._search_speed_max = search_speed_max

    def reynolds_at_speed(
        self,
        free_stream_speed: float,
        tsr: float,
        axial_induction: float,
        tangential_induction: float,
    ) -> float:
        """
        Midspan Reynolds number produced by a given free-air speed.

        Combines the velocity-triangle relative speed with the Re definition.
        """
        v_rel = relative_velocity(
            tsr=tsr,
            free_stream_speed=free_stream_speed,
            tip_radius=self._tip_radius,
            section_radius=self._midspan_radius,
            axial_induction=axial_induction,
            tangential_induction=tangential_induction,
        )
        return reynolds_number(
            relative_speed=v_rel,
            chord=self._midspan_chord,
            kinematic_viscosity=self._kinematic_viscosity,
        )

    def solve_speed_for_target_reynolds(
        self,
        target_reynolds: float,
        tsr: float,
        axial_induction: float,
        tangential_induction: float,
    ) -> float:
        """
        Find U'_inf so that the midspan Reynolds number equals `target_reynolds`.

        Uses a bracketed root find on the residual
            f(U) = Re(U) - target_reynolds
        over the configured speed search interval.
        """

        def residual(free_stream_speed: float) -> float:
            return (
                self.reynolds_at_speed(
                    free_stream_speed=free_stream_speed,
                    tsr=tsr,
                    axial_induction=axial_induction,
                    tangential_induction=tangential_induction,
                )
                - target_reynolds
            )

        return brentq(residual, self._search_speed_min, self._search_speed_max)
