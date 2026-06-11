"""
Task-2 (wake) test-matrix builder for Group 2.

Group 2 measures the wake in a vertical y-z plane at ONE downstream station, for
yaw = 0, +30 and -30 degrees, at the prescribed tunnel speed (5.4 m/s), optimum
pitch and optimum TSR. Per yaw the plane holds 23 points (3 x 23 = 69 total).

How the plane points are placed (per yaw)
-----------------------------------------
All points share the streamwise station X = 2D behind the rotor. Centred on the
yaw-deflected wake centre (Y = delta from the Jimenez model, Z = 0 at hub
height) we place:

* a hub-height horizontal LINE (Z = 0), cosine-clustered in Y - the lateral
  deficit profile used to calibrate the wake-centre model (Jimenez / Cacciola);
* a set of concentric CIRCLES - real 2-D samples used to validate / graph the
  wake shape in the plane.

Both groups use at most `measuring_area_fraction` (85%) of the traverse extent
reachable around the centre, leaving a margin to the limits / tunnel walls. The
vertical reach (+600 / -700 mm about hub) is smaller than the wake, so the
circles capture the wake core while the hub line captures the full lateral width.

Traverse reference frame (Lecture 5)
------------------------------------
Home X=Y=Z=0 sits at hub height, `traverse_home_offset` (0.5D) behind the rotor,
with X growing downwind. Hence the reported X = downstream_distance - home_offset;
Y and Z share their origin with the rotor centre (so Y, Z are the lateral /
vertical offsets from the hub). Positions are reported in millimetres; the
turntable yaw angle A equals the rotor yaw for each block.
"""

from __future__ import annotations

import math

from .campaign_config import CampaignConfig
from .operating_point import OperatingPointResult, OperatingPointSolver
from .test_matrix_row import G1_MEASUREMENT_COLUMNS, TestMatrixRow
from .wake_model import (
    centered_lateral_positions,
    concentric_ring_positions,
    reachable_radius,
    wake_center_offset,
    wake_radius,
)


# Convert metres to millimetres for the reported probe positions.
_M_TO_MM: float = 1000.0


class Task2WakeBuilder:
    """Builds the wake-survey test matrix (a y-z plane per yaw)."""

    def __init__(self, config: CampaignConfig, solver: OperatingPointSolver) -> None:
        self._config = config
        self._solver = solver

    def _optimum_tsr(self) -> float:
        """Optimum TSR taken from the look-up table's peak-C_P point."""
        return self._solver._table.optimum_operating_point().tsr  # noqa: SLF001

    def _traverse_x_mm(self) -> float:
        """Streamwise probe position in the traverse frame [mm] (constant)."""
        wake = self._config.wake
        return (wake.downstream_distance - wake.traverse_home_offset) * _M_TO_MM

    # ----------------------------------------------------------------------
    # (Y, Z) plane positions for one yaw, centred on the deflected wake centre.
    # ----------------------------------------------------------------------
    def _plane_positions_for_yaw(
        self, yaw_deg: float, thrust_coefficient: float
    ) -> list[tuple[float, float]]:
        """
        Build the (Y, Z) measurement positions [m] for one yaw case.

        Returns the hub-height line points (Z = 0) followed by the concentric-ring
        points, all centred on the Jimenez wake centre (Y = delta, Z = 0). Both
        groups use at most `measuring_area_fraction` of the reachable extent so a
        margin to the traverse limits (and tunnel walls) is preserved.
        """
        wake = self._config.wake
        yaw_rad = math.radians(yaw_deg)

        center_y = wake_center_offset(
            thrust_coefficient=thrust_coefficient,
            yaw_angle_rad=yaw_rad,
            wake_expansion_coefficient=wake.wake_expansion_coefficient,
            rotor_diameter=self._config.rotor_diameter,
            downstream_distance=wake.downstream_distance,
        )
        r_w = wake_radius(
            wake_expansion_coefficient=wake.wake_expansion_coefficient,
            rotor_diameter=self._config.rotor_diameter,
            downstream_distance=wake.downstream_distance,
        )

        # Hub-height horizontal line (Z = 0): reach ~hubline_span_factor wake radii,
        # capped at a fraction of the lateral room around the (offset) centre.
        available_half_y = min(center_y - wake.y_min, wake.y_max - center_y)
        hub_half_width = min(
            wake.hubline_span_factor * r_w,
            wake.measuring_area_fraction * available_half_y,
        )
        hub_line_y = centered_lateral_positions(
            wake_center_y=center_y,
            half_width=hub_half_width,
            number_of_points=wake.hubline_points,
        )
        line_positions = [(float(y), 0.0) for y in hub_line_y]

        # Concentric circles centred on the wake centre, sized to the reachable
        # window (the vertical reach is the binding limit) with the same margin.
        max_radius = wake.measuring_area_fraction * reachable_radius(
            center_y=center_y,
            center_z=0.0,
            y_min=wake.y_min,
            y_max=wake.y_max,
            z_min=wake.z_min,
            z_max=wake.z_max,
        )
        ring_positions = concentric_ring_positions(
            center_y=center_y,
            center_z=0.0,
            max_radius=max_radius,
            points_per_ring=wake.circle_rings_points,
        )

        return line_positions + ring_positions

    # ----------------------------------------------------------------------
    def build_results(
        self,
    ) -> list[tuple[OperatingPointResult, list[tuple[float, float]]]]:
        """
        For each yaw, solve the (constant) operating point and the (Y, Z) positions.

        The wake task SETS the tunnel speed (5.4 m/s); the rotor runs at the
        optimum TSR and pitch. Returns one (result, [(y, z), ...]) entry per yaw.
        """
        pitch = self._config.targets.optimum_pitch_deg
        tunnel_speed = self._config.wake.requested_tunnel_speed
        tsr = self._optimum_tsr()

        results: list[tuple[OperatingPointResult, list[tuple[float, float]]]] = []
        for yaw in self._config.yaw.yaw_angles_deg:
            result = self._solver.solve_for_fixed_tunnel_speed(
                tsr=tsr,
                pitch_deg=pitch,
                yaw_deg=yaw,
                requested_tunnel_speed=tunnel_speed,
            )
            positions = self._plane_positions_for_yaw(
                yaw_deg=yaw,
                thrust_coefficient=result.ct_thrust,
            )
            results.append((result, positions))
        return results

    def build_rows(self) -> list[TestMatrixRow]:
        """Convert the per-yaw results and plane positions into numbered rows.

        Rows are ordered by requested wind speed (constant here), then by yaw (the
        turntable setting), then swept vertically (Z) and laterally (Y) so the
        traverse moves smoothly through the plane.
        """
        x_mm = self._traverse_x_mm()
        empty_measurements = {name: "" for name in G1_MEASUREMENT_COLUMNS}

        flattened: list[tuple[OperatingPointResult, float, float]] = []
        for result, positions in self.build_results():
            for y_m, z_m in positions:
                flattened.append((result, y_m, z_m))

        flattened.sort(
            key=lambda item: (
                round(item[0].requested_tunnel_speed, 3),
                item[0].yaw_deg,
                item[2],  # Z
                item[1],  # Y
            )
        )

        rows: list[TestMatrixRow] = []
        for index, (result, y_m, z_m) in enumerate(flattened, start=1):
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
                    probe_x=x_mm,
                    probe_y=y_m * _M_TO_MM,
                    probe_z=z_m * _M_TO_MM,
                    measurements=dict(empty_measurements),
                )
            )
        return rows
