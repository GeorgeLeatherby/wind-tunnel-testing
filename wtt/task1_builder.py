"""
Task-1 (performance) test-matrix builder for Group 2.

Group 2 measures three performance curves at yaw = 0, +30 and -30 degrees, each
at Re = 75000 and the optimum blade pitch, sweeping the tip-speed ratio.

Design choice (documented): the dominant requirement is "Re = 75000 (+/- 2000)",
but the tunnel speed can only be SET in finite steps, so we drive the sweep by
the *requested tunnel speed* rather than by TSR. For each yaw the requested speed
is stepped in `performance_speed_step` (0.5 m/s) increments across the achievable
constant-Reynolds range, starting at the lowest feasible speed; for every speed
we re-solve the TSR that keeps the midspan Reynolds number at 75000. This makes
the "Requested wind speed" column contain only values the operator can dial in,
while Re stays exactly on target.

Because only a handful of 0.5 m/s steps are feasible, Task 1 yields fewer than 30
points (the assignment caps it at 30, and fewer is acceptable). Rows are finally
ordered by requested wind speed - the hardest parameter to set in the tunnel.
"""

from __future__ import annotations

import math

import numpy as np

from .campaign_config import CampaignConfig
from .operating_point import OperatingPointResult, OperatingPointSolver
from .test_matrix_row import G1_MEASUREMENT_COLUMNS, TestMatrixRow


class Task1PerformanceBuilder:
    """Builds the performance test matrix (constant Re, stepped tunnel speed)."""

    def __init__(self, config: CampaignConfig, solver: OperatingPointSolver) -> None:
        self._config = config
        self._solver = solver

    def _speed_sweep_for_yaw(self, yaw_deg: float) -> np.ndarray:
        """
        Requested tunnel speeds [m/s] for one yaw curve, in 0.5 m/s steps.

        The achievable constant-Reynolds speed range is bounded by the feasible
        TSR band: the lowest speed is the one commanded at the highest feasible TSR
        and the highest speed the one commanded at the lowest feasible TSR (the
        commanded speed decreases with TSR at fixed Re). We then take every
        `performance_speed_step` multiple inside that range, starting at the lowest
        feasible step. Raises if no step fits (a configuration error).
        """
        pitch = self._config.targets.optimum_pitch_deg
        target_re = self._config.targets.target_reynolds
        tsr_lo = self._config.performance_tsr_feasible_min
        tsr_hi = self._config.performance_tsr_feasible_max
        step = self._config.performance_speed_step

        lowest_speed = self._solver.solve_for_target_reynolds(
            tsr=tsr_hi, pitch_deg=pitch, yaw_deg=yaw_deg, target_reynolds=target_re
        ).requested_tunnel_speed
        highest_speed = self._solver.solve_for_target_reynolds(
            tsr=tsr_lo, pitch_deg=pitch, yaw_deg=yaw_deg, target_reynolds=target_re
        ).requested_tunnel_speed

        first = math.ceil(lowest_speed / step - 1e-9) * step
        last = math.floor(highest_speed / step + 1e-9) * step
        if last < first:
            raise ValueError(
                f"No {step} m/s tunnel-speed step is feasible for yaw {yaw_deg:+.0f} "
                f"(achievable range {lowest_speed:.2f}-{highest_speed:.2f} m/s)."
            )
        return np.arange(first, last + 0.5 * step, step)

    def build_results(self) -> list[OperatingPointResult]:
        """
        Solve every (yaw, requested-speed) operating point at constant Reynolds.

        For each requested tunnel speed the TSR is re-solved so the midspan
        Reynolds number equals the target; pitch is held at the optimum. Results
        are ordered by requested wind speed, then by yaw.
        """
        results: list[OperatingPointResult] = []
        pitch = self._config.targets.optimum_pitch_deg
        target_re = self._config.targets.target_reynolds
        tsr_lo = self._config.performance_tsr_feasible_min
        tsr_hi = self._config.performance_tsr_feasible_max

        for yaw in self._config.yaw.yaw_angles_deg:
            for speed in self._speed_sweep_for_yaw(yaw):
                result = self._solver.solve_for_target_reynolds_at_tunnel_speed(
                    requested_tunnel_speed=float(speed),
                    pitch_deg=pitch,
                    yaw_deg=yaw,
                    target_reynolds=target_re,
                    tsr_search_min=tsr_lo,
                    tsr_search_max=tsr_hi,
                )
                results.append(result)

        results.sort(key=lambda r: (round(r.requested_tunnel_speed, 3), r.yaw_deg))
        return results

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
