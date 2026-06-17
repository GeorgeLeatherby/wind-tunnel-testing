"""
Task-1 (performance) test-matrix builder for Group 2.

Group 2 measures three performance curves at yaw = 0, +30 and -30 degrees, each
at Re in the range [70000, 80000] and the optimum blade pitch.

Design (current): the *requested tunnel speed* is set on a coarse grid of 0.5 m/s
steps. For every grid speed a range of TSR values is swept (0.3 step, e.g.,
[5.5, 5.8, 6.1, ..., 9.0]). For each (speed, TSR) pair all three yaw angles are
tested. Rows are kept if they satisfy power/torque/rpm limits and Re is in
[70000, 80000]. This yields ~30 points total (the hard maximum). The rows are
ordered by ascending requested wind speed, TSR, then yaw.
"""

from __future__ import annotations

import math

from .campaign_config import CampaignConfig
from .constraints import check_row
from .operating_point import OperatingPointResult, OperatingPointSolver
from .test_matrix_row import G1_MEASUREMENT_COLUMNS, TestMatrixRow


# Hard cap on the number of performance points (assignment maximum).
_MAX_POINTS: int = 30


class Task1PerformanceBuilder:
    """Builds the performance test matrix (constant Re, speed grid, 3 yaws)."""

    def __init__(self, config: CampaignConfig, solver: OperatingPointSolver) -> None:
        self._config = config
        self._solver = solver

    # ----------------------------------------------------------------------
    # Feasible requested-speed band (Re = 75000 reachable for every yaw)
    # ----------------------------------------------------------------------
    def _feasible_speed_band(self) -> tuple[float, float]:
        """
        Requested-speed range over which Re = 75000 is reachable for all yaws.

        At constant Re the required tunnel speed is highest at the lowest TSR and
        lowest at the highest TSR. Solving the Re calibration at the two TSR limits
        therefore brackets the achievable speed range for each yaw; the band common
        to every yaw is the intersection of those ranges.
        """
        pitch = self._config.targets.optimum_pitch_deg
        target_re = self._config.targets.target_reynolds
        tsr_min = self._config.constraints.tsr_min
        tsr_max = self._config.constraints.tsr_max
        yaws = self._config.yaw.yaw_angles_deg

        # Highest TSR -> lowest required speed (the low end of each yaw's range).
        low_ends = [
            self._solver.solve_for_target_reynolds(
                tsr=tsr_max, pitch_deg=pitch, yaw_deg=yaw, target_reynolds=target_re
            ).requested_tunnel_speed
            for yaw in yaws
        ]
        # Lowest TSR -> highest required speed (the high end of each yaw's range).
        high_ends = [
            self._solver.solve_for_target_reynolds(
                tsr=tsr_min, pitch_deg=pitch, yaw_deg=yaw, target_reynolds=target_re
            ).requested_tunnel_speed
            for yaw in yaws
        ]
        return max(low_ends), min(high_ends)

    def _speed_grid(self) -> list[float]:
        """
        Requested speeds on a 0.5 m/s grid, from the lowest feasible speed up.

        The first grid point is the smallest multiple of the step at or above the
        feasible band's lower edge; the last is the largest multiple at or below
        the upper edge.
        """
        band_low, band_high = self._feasible_speed_band()
        step = self._config.performance_speed_step
        first = math.ceil(band_low / step) * step

        speeds: list[float] = []
        speed = first
        while speed <= band_high + 1e-9:
            speeds.append(round(speed, 3))
            speed += step
        return speeds

    def _tsr_sweep(self) -> list[float]:
        """
        Generate TSR values to sweep at each tunnel speed for 30-point coverage.

        Returns a list of TSR values (e.g., [5.5, 5.8, 6.1, ..., 9.0]) that will be
        tested at each tunnel speed. With ~5 speeds and ~2 TSR per speed × 3 yaws,
        this yields ~30 rows total.
        """
        tsr_min = self._config.constraints.tsr_min
        tsr_max = self._config.constraints.tsr_max
        step = 0.3  # Sweep in 0.3 steps to yield ~2 TSR per speed per yaw -> 30 rows
        tsr_values = []
        tsr = tsr_min
        while tsr <= tsr_max + 1e-9:
            tsr_values.append(round(tsr, 2))
            tsr += step
        return tsr_values

    def _within_limits_relaxed(self, result: OperatingPointResult) -> bool:
        """
        True if a solved point satisfies every Group-2 limit except Re is allowed
        to vary in [70000, 80000] instead of being forced to 75000±2000.
        """
        return not check_row(
            result=result,
            constraints=self._config.constraints,
            target_reynolds=self._config.targets.target_reynolds,
            enforce_reynolds=False,  # Allow Re band; check later manually
        ) and 70000 <= result.reynolds <= 80000

    def build_results(self) -> list[OperatingPointResult]:
        """
        Solve every kept (speed, TSR, yaw) operating point up to 30 rows.

        For each grid speed and each TSR in the sweep, all three yaw points are
        solved. Rows are kept if they satisfy power/torque/rpm limits and have
        Re in [70000, 80000]. The result is sorted by requested speed and TSR,
        then yaw, and capped at the 30-point maximum.
        """
        pitch = self._config.targets.optimum_pitch_deg

        kept: list[OperatingPointResult] = []
        for speed in self._speed_grid():
            for tsr in self._tsr_sweep():
                for yaw in self._config.yaw.yaw_angles_deg:
                    result = self._solver.solve_for_fixed_tunnel_speed(
                        tunnel_speed=speed,
                        pitch_deg=pitch,
                        yaw_deg=yaw,
                        tsr=tsr,
                    )
                    if self._within_limits_relaxed(result):
                        kept.append(result)

        kept.sort(
            key=lambda r: (round(r.requested_tunnel_speed, 3), r.tsr, r.yaw_deg)
        )
        return kept[:_MAX_POINTS]

    def build_rows(self) -> list[TestMatrixRow]:
        """Convert the physics results into numbered test-matrix rows."""
        rows: list[TestMatrixRow] = []
        empty_measurements = {name: "" for name in G1_MEASUREMENT_COLUMNS}

        for index, result in enumerate(self.build_results(), start=1):
            rows.append(
                TestMatrixRow(
                    test_number=index,
                    requested_tunnel_speed=result.requested_tunnel_speed,
                    equivalent_free_air_speed=result.equivalent_free_air_speed,
                    blade_pitch_deg=result.pitch_deg,
                    rotor_speed_rpm=result.rotor_speed_rpm,
                    yaw_deg=result.yaw_deg,
                    reynolds=result.reynolds,
                    tsr=result.tsr,
                    expected_cp=result.cp,
                    expected_ct_thrust=result.ct_thrust,
                    expected_power_w=result.power_w,
                    expected_torque_nm=result.torque_nm,
                    expected_thrust_n=result.thrust_n,
                    measurements=dict(empty_measurements),
                )
            )
        return rows
