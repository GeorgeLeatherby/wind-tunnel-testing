"""
Task-2 (wake) test-matrix builder for Group 2.

Group 2 measures the wake shape at one downstream station for yaw = 0, +30 and
-30 degrees, at the prescribed wind speed (5.4 m/s), the optimum (0 deg) blade
pitch and the optimum TSR. A total of 70 hub-height points is required.

How the 70 points are placed
----------------------------
* All points are at hub height, i.e. Z = 0 (origin at the hub center, Z down).
* The streamwise station is fixed at X = 3D (single downstream location).
* The 70 points are split across the three yaw cases (23 / 23 / 24).
* For each yaw, the lateral wake center is predicted with the Jimenez deflection
  model and the points are spread (cosine-clustered, denser near the center)
  across a span sized from the local wake radius. This matches the guidance to
  "distribute measurement points around the expected center of the wake".

Reference frame (assignment): X downwind, Z down, origin at hub center; Y is the
lateral horizontal axis. Positions are reported in millimetres.
"""

from __future__ import annotations

import math

import numpy as np

from .campaign_config import CampaignConfig
from .operating_point import OperatingPointResult, OperatingPointSolver
from .test_matrix_row import G1_MEASUREMENT_COLUMNS, TestMatrixRow
from .wake_model import (
    centered_lateral_positions,
    fitted_half_width,
    wake_center_offset,
    wake_radius,
)


# Fraction of the wake radius used as half-width of the lateral survey span.
# 1.5 means the points reach 1.5 wake-radii either side of the center, which
# straddles the wake edges where the deficit gradient (and hence the center
# information) is strongest.
_SPAN_FACTOR: float = 1.5

# Convert metres to millimetres for the reported probe positions.
_M_TO_MM: float = 1000.0


class Task2WakeBuilder:
    """Builds the 70-row wake-survey test matrix."""

    def __init__(self, config: CampaignConfig, solver: OperatingPointSolver) -> None:
        self._config = config
        self._solver = solver

    # ----------------------------------------------------------------------
    # Distribute the total point budget across the yaw cases as evenly as
    # possible (e.g. 70 over 3 -> 24, 23, 23).
    # ----------------------------------------------------------------------
    def _points_per_yaw(self) -> list[int]:
        total = self._config.wake.total_points
        n_yaw = len(self._config.yaw.yaw_angles_deg)
        base = total // n_yaw
        remainder = total % n_yaw
        # Give the first `remainder` yaw cases one extra point.
        return [base + (1 if i < remainder else 0) for i in range(n_yaw)]

    def _optimum_tsr(self) -> float:
        """Optimum TSR taken from the look-up table's peak-C_P point."""
        return self._solver._table.optimum_operating_point().tsr  # noqa: SLF001

    # ----------------------------------------------------------------------
    # Lateral Y positions for one yaw case, centred on the deflected wake center.
    # ----------------------------------------------------------------------
    def _lateral_positions_for_yaw(
        self, yaw_deg: float, thrust_coefficient: float, number_of_points: int
    ) -> np.ndarray:
        yaw_rad = math.radians(yaw_deg)
        diameter = self._config.rotor_diameter
        distance = self._config.wake.downstream_distance
        k_d = self._config.wake.wake_expansion_coefficient

        center_y = wake_center_offset(
            thrust_coefficient=thrust_coefficient,
            yaw_angle_rad=yaw_rad,
            wake_expansion_coefficient=k_d,
            rotor_diameter=diameter,
            downstream_distance=distance,
        )
        # Desired span from the wake physics, then shrunk so it fits the traverse
        # limits around the (possibly offset) center -> no point gets clipped.
        desired_half_width = _SPAN_FACTOR * wake_radius(
            wake_expansion_coefficient=k_d,
            rotor_diameter=diameter,
            downstream_distance=distance,
        )
        half_width = fitted_half_width(
            wake_center_y=center_y,
            desired_half_width=desired_half_width,
            lower_limit=self._config.wake.y_min,
            upper_limit=self._config.wake.y_max,
        )

        return centered_lateral_positions(
            wake_center_y=center_y,
            half_width=half_width,
            number_of_points=number_of_points,
        )

    # ----------------------------------------------------------------------
    # Build raw results: one operating point per yaw (constant during the sweep)
    # plus the list of Y positions for that yaw.
    # ----------------------------------------------------------------------
    def build_results(self) -> list[tuple[OperatingPointResult, np.ndarray]]:
        """
        For each yaw, solve the (constant) operating point and the Y positions.

        Returns a list of (operating_point_result, y_positions_array), one entry
        per yaw case.
        """
        pitch = self._config.targets.optimum_pitch_deg
        free_air_speed = self._config.wake.free_stream_speed
        tsr = self._optimum_tsr()

        results: list[tuple[OperatingPointResult, np.ndarray]] = []
        for yaw, n_points in zip(
            self._config.yaw.yaw_angles_deg, self._points_per_yaw()
        ):
            result = self._solver.solve_for_fixed_speed(
                tsr=tsr,
                pitch_deg=pitch,
                yaw_deg=yaw,
                free_air_speed=free_air_speed,
            )
            y_positions = self._lateral_positions_for_yaw(
                yaw_deg=yaw,
                thrust_coefficient=result.ct_thrust,
                number_of_points=n_points,
            )
            results.append((result, y_positions))
        return results

    def build_rows(self) -> list[TestMatrixRow]:
        """Convert the per-yaw results and Y positions into numbered rows."""
        rows: list[TestMatrixRow] = []
        empty_measurements = {name: "" for name in G1_MEASUREMENT_COLUMNS}

        # Fixed streamwise / vertical positions (mm), hub-centered frame.
        x_mm = self._config.wake.downstream_distance * _M_TO_MM
        z_mm = 0.0  # hub height

        test_number = 1
        for result, y_positions in self.build_results():
            for y_m in y_positions:
                rows.append(
                    TestMatrixRow(
                        test_number=test_number,
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
                        probe_x=x_mm,
                        probe_y=float(y_m) * _M_TO_MM,
                        probe_z=z_mm,
                        measurements=dict(empty_measurements),
                    )
                )
                test_number += 1
        return rows
