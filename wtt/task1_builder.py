"""
Task-1 (performance) test-matrix builder for Group 2.

Group 2 measures three performance curves at yaw = 0, +30 and -30 degrees, each
at Re = 75000 and the optimum (here: 0 deg) blade pitch, sweeping the tip-speed
ratio. A total of 30 measurement points is required, i.e. 10 TSR points per yaw.

Design choice (documented): the dominant requirement is "Re = 75000 (+/- 2000)".
To honour it we build *constant-Reynolds* curves: for every TSR point the wind
speed is re-solved so the midspan Reynolds number equals 75000, and the rotor
speed follows from TSR. Because the relative velocity grows with TSR, the
calibrated wind speed decreases as TSR increases, which automatically produces a
spread of wind speeds across the sweep (checked afterwards against the
"avoid wind-speed changes < 0.5 m/s" guideline and the rpm / power / torque
limits).

The TSR sweep is centred on the target TSR (7.05) and kept inside the Group-2
band [5.5, 9].
"""

from __future__ import annotations

import numpy as np

from .campaign_config import CampaignConfig
from .operating_point import OperatingPointResult, OperatingPointSolver
from .test_matrix_row import G1_MEASUREMENT_COLUMNS, TestMatrixRow


class Task1PerformanceBuilder:
    """Builds the 30-row performance test matrix."""

    def __init__(self, config: CampaignConfig, solver: OperatingPointSolver) -> None:
        self._config = config
        self._solver = solver

    def _tsr_sweep(self) -> np.ndarray:
        """
        TSR values for one yaw curve.

        Evenly spaced across the verified *feasible* TSR band, with
        `performance_points_per_yaw` samples. The feasible band is narrower than
        the raw limit band so that every constant-Reynolds point respects the
        power / torque / rotor-speed limits, while still bracketing the optimum
        TSR (7.05).
        """
        return np.linspace(
            self._config.performance_tsr_sweep_min,
            self._config.performance_tsr_sweep_max,
            self._config.performance_points_per_yaw,
        )

    def build_results(self) -> list[OperatingPointResult]:
        """
        Solve every (yaw, TSR) operating point and return the raw physics results.

        Pitch is held at the configured optimum value; Reynolds is calibrated to
        the target for each point.
        """
        results: list[OperatingPointResult] = []
        pitch = self._config.targets.optimum_pitch_deg
        target_re = self._config.targets.target_reynolds

        for yaw in self._config.yaw.yaw_angles_deg:
            for tsr in self._tsr_sweep():
                result = self._solver.solve_for_target_reynolds(
                    tsr=float(tsr),
                    pitch_deg=pitch,
                    yaw_deg=yaw,
                    target_reynolds=target_re,
                )
                results.append(result)
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
