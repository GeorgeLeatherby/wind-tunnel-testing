"""
Configuration objects for the G1 wind-tunnel test campaign.

These are plain immutable data containers (frozen dataclasses) that hold the
fixed facts of the experiment: the wind-tunnel geometry, the operating targets
(Reynolds number, target TSR, optimum pitch), the Group-2 constraints, and the
wake-survey settings.

NOTE on the "no defaults" rule: the dataclasses below intentionally do NOT give
default values to physically meaningful quantities. The concrete numbers for the
actual campaign live in `build_group2_campaign()` at the bottom, in ONE place,
so they are easy to find and change. If you forget to pass one, construction
fails on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindTunnelGeometry:
    """
    Test-section geometry of the wind tunnel.

    The blockage correction only needs the cross-sectional area that the flow
    passes through (height x width). The length is kept for documentation and
    for sanity-checking the downstream measurement positions.
    """

    length: float  # [m]  streamwise length of the test section
    width: float  # [m]   horizontal (cross-stream) width
    height: float  # [m]  vertical height

    def cross_section_area(self) -> float:
        """Cross-sectional area seen by the flow: width * height (single formula)."""
        return self.width * self.height


@dataclass(frozen=True)
class OperatingTargets:
    """Targets that define the desired operating point of the rotor."""

    target_reynolds: float  # [-]  Reynolds number to calibrate to (75000)
    target_tsr: float  # [-]       tip-speed ratio to set at the design point (7.05)
    optimum_pitch_deg: float  # [deg] collective blade pitch (0 now, changeable)


@dataclass(frozen=True)
class Group2Constraints:
    """Hard limits the test matrix must respect (Group 2)."""

    tsr_min: float  # [-]
    tsr_max: float  # [-]
    power_max_w: float  # [W]
    torque_max_nm: float  # [Nm]
    rotor_speed_max_rpm: float  # [rpm]
    reynolds_tolerance: float  # [-] allowed +/- band around target Reynolds


@dataclass(frozen=True)
class WakeSurveySettings:
    """Settings for the Task-2 wake survey (hub-height horizontal line)."""

    free_stream_speed: float  # [m/s]   prescribed wind speed for the wake task (5.4)
    downstream_distance: float  # [m]    single streamwise station X
    wake_expansion_coefficient: float  # [-] Jimenez k_d
    total_points: int  # [-]            total wake measurement points (70)
    # Traverse limits of the wake probe (from the lecture), in metres.
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class YawSchedule:
    """The set of yaw misalignment angles to test (Group 2: 0, +30, -30)."""

    yaw_angles_deg: tuple[float, ...]


@dataclass(frozen=True)
class YawModelSettings:
    """
    Parameters of the yaw-dependent performance correction (see `wtt.yaw_model`).

    The cosine exponents follow from actuator-disk momentum theory applied to the
    rotor-normal wind component U cos(gamma): power scales with cos^3 and thrust
    with cos^2. `rotational_asymmetry` (k_gamma) scales the advancing/retreating-
    blade coupling that makes +gamma and -gamma differ for a fixed rotation sense;
    its sign and magnitude are modelling parameters to be calibrated against
    measurement (no hidden default — it must be set explicitly here).
    """

    power_cosine_exponent: float  # [-]  p_P (momentum theory: 3)
    thrust_cosine_exponent: float  # [-] p_T (momentum theory: 2)
    rotational_asymmetry: float  # [-]   k_gamma sensitivity (calibrate to data)


@dataclass(frozen=True)
class CampaignConfig:
    """Top-level container grouping every configuration block for one campaign."""

    tunnel: WindTunnelGeometry
    targets: OperatingTargets
    constraints: Group2Constraints
    wake: WakeSurveySettings
    yaw: YawSchedule
    yaw_model: YawModelSettings
    lookup_table_path: str
    rotor_diameter: float  # [m]  G1 rotor diameter (derived from the lookup geometry)
    performance_points_per_yaw: int  # [-] number of TSR points per yaw in Task 1
    # Feasible TSR band actually swept in Task 1. This is narrower than the raw
    # [tsr_min, tsr_max] limit band: at constant Re = 75000 the calibrated wind
    # speed (and therefore power ~ U^3 and torque) grows towards low TSR, so the
    # lowest part of the allowed band would exceed the 75 W / 1 Nm limits. The
    # band below was chosen and verified so every swept point stays inside ALL
    # Group-2 limits while still bracketing the optimum TSR (7.05).
    performance_tsr_sweep_min: float  # [-]
    performance_tsr_sweep_max: float  # [-]


def build_group2_campaign(
    lookup_table_path: str,
    rotor_diameter: float,
) -> CampaignConfig:
    """
    Assemble the concrete configuration for the Group-2 campaign.

    This is the single place where the campaign numbers are written down. The
    rotor diameter is passed in (it is read from the lookup-table geometry, so we
    do not hard-code it here) which keeps geometry and configuration consistent.
    """
    return CampaignConfig(
        tunnel=WindTunnelGeometry(
            length=4.5,  # [m]
            width=2.7,  # [m]
            height=1.8,  # [m]
        ),
        targets=OperatingTargets(
            target_reynolds=75000.0,
            target_tsr=7.05,
            optimum_pitch_deg=0.4,  # [deg]  small positive pitch
        ),
        constraints=Group2Constraints(
            tsr_min=5.5,
            tsr_max=9.0,
            power_max_w=75.0,
            torque_max_nm=1.0,
            rotor_speed_max_rpm=850.0,
            reynolds_tolerance=2000.0,
        ),
        wake=WakeSurveySettings(
            free_stream_speed=5.4,  # [m/s] prescribed for Group 2 wake task
            downstream_distance=3.0 * rotor_diameter,  # X = 3D
            wake_expansion_coefficient=0.07,
            total_points=70,
            # Traverse limits from Lecture 5 (converted mm -> m).
            x_min=0.0,
            x_max=3.850,
            y_min=-1.0,
            y_max=1.0,
            z_min=-0.7,
            z_max=0.6,
        ),
        yaw=YawSchedule(
            yaw_angles_deg=(0.0, 30.0, -30.0),
        ),
        yaw_model=YawModelSettings(
            # Momentum-theory cosine exponents (rotor-normal wind U cos gamma):
            # power ~ cos^3, thrust ~ cos^2 (Lecture 3 actuator-disk theory).
            power_cosine_exponent=3.0,
            thrust_cosine_exponent=2.0,
            # Rotational +/- yaw asymmetry sensitivity (advance-ratio coupling).
            # Nominal small value; sign/magnitude to be calibrated against the
            # measured yaw sweep. Set to 0.0 to recover a symmetric cosine law.
            rotational_asymmetry=0.10,
        ),
        lookup_table_path=lookup_table_path,
        rotor_diameter=rotor_diameter,
        performance_points_per_yaw=10,
        # Verified feasible band (see comment on the field above): power, torque,
        # rotor speed and Re = 75000 all satisfied across [6.0, 9.0].
        performance_tsr_sweep_min=6.0,
        performance_tsr_sweep_max=9.0,
    )
