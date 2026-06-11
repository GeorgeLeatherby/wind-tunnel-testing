"""
Wake-center deflection and wake-survey point distribution for Task 2.

Physical picture (Lecture 3, "Wake Deflection by Yaw Misalignment", and the
Cacciola/Bottasso paper on wake-center tracking): when the turbine is yawed, the
wake is deflected laterally. To place the wake-probe points usefully we must
distribute them *around the expected wake center*, which shifts in the Y
(lateral) direction depending on the yaw angle and the downstream distance.

We use the classical Jimenez (2010) analytical wake-deflection model to predict
the lateral position of the wake center, and then build a measurement plane
around it: concentric circles centred on the wake center (for 2-D wake-shape
validation) plus a denser hub-height line of Y positions (for calibrating the
wake-center model). Cosine clustering keeps points dense where the deficit
gradient is steepest.

Reference frame: X is the downwind direction, Z is vertical and Y is the lateral
horizontal axis, with the origin at the hub center (matching the traverse frame
once the streamwise home offset is applied). The circle radii are sized to the
reachable traverse window so no point is clipped.

Each formula is its own function.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Jimenez wake-center deflection
# ---------------------------------------------------------------------------
def initial_skew_angle(thrust_coefficient: float, yaw_angle_rad: float) -> float:
    """
    Initial wake skew angle at the rotor plane (Jimenez):

        xi0 = 0.5 * cos^2(gamma) * sin(gamma) * C_T

    `gamma` is the yaw misalignment, `C_T` the rotor THRUST coefficient. The sign
    follows the sign of the yaw angle, so a positive yaw gives a positive skew.
    """
    return 0.5 * math.cos(yaw_angle_rad) ** 2 * math.sin(yaw_angle_rad) * thrust_coefficient


def wake_center_offset(
    thrust_coefficient: float,
    yaw_angle_rad: float,
    wake_expansion_coefficient: float,
    rotor_diameter: float,
    downstream_distance: float,
) -> float:
    """
    Lateral (Y) offset of the wake center at a downstream distance x.

    The local deflection angle decays as the wake expands,
        xi(x) = xi0 / (1 + 2 k_d x / D)^2 ,
    and the lateral offset is the integral of tan(xi) ~ xi along x, which has the
    closed form

        delta(x) = (xi0 * D / (2 k_d)) * (1 - 1 / (1 + 2 k_d x / D))

    Returns the offset in metres. At zero yaw xi0 = 0, so delta = 0.
    """
    xi0 = initial_skew_angle(thrust_coefficient, yaw_angle_rad)
    expansion = 1.0 + 2.0 * wake_expansion_coefficient * downstream_distance / rotor_diameter
    return (xi0 * rotor_diameter / (2.0 * wake_expansion_coefficient)) * (
        1.0 - 1.0 / expansion
    )


def wake_radius(
    wake_expansion_coefficient: float,
    rotor_diameter: float,
    downstream_distance: float,
) -> float:
    """
    Approximate wake radius at distance x: r_w = R + k_d * x.

    Used to size the lateral span over which the survey points are spread.
    """
    rotor_radius = rotor_diameter / 2.0
    return rotor_radius + wake_expansion_coefficient * downstream_distance


# ---------------------------------------------------------------------------
# Survey-point distribution (hub-height Y line centred on the wake center)
# ---------------------------------------------------------------------------
def centered_lateral_positions(
    wake_center_y: float,
    half_width: float,
    number_of_points: int,
) -> np.ndarray:
    """
    Build `number_of_points` Y positions centred on `wake_center_y`.

    Cosine spacing is used so that points cluster near the center (where the wake
    gradient is steepest and the center is best identified) and thin out towards
    the edges. The positions span [center - half_width, center + half_width].
    """
    if number_of_points < 2:
        raise ValueError("Need at least two lateral points to span the wake.")
    # Parameter t in [0, pi]; cos(t) gives endpoints at +/-1 with clustering at 0.
    t = np.linspace(0.0, math.pi, number_of_points)
    # -cos(t) maps to [-1, 1] increasing; multiply by half_width and shift.
    normalized = -np.cos(t)
    return wake_center_y + half_width * normalized


def clip_to_limits(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clamp an array of positions into the allowed traverse range [lower, upper]."""
    return np.clip(values, lower, upper)


def reachable_radius(
    center_y: float,
    center_z: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> float:
    """
    Largest circle radius around (center_y, center_z) that fits the traverse window.

    The probe can move inside the rectangle [y_min, y_max] x [z_min, z_max]; the
    biggest centred circle is bounded by the nearest wall:

        r = min(center_y - y_min, y_max - center_y,
                center_z - z_min, z_max - center_z)

    Raises if the centre lies outside the window (a configuration error).
    """
    if not (y_min <= center_y <= y_max and z_min <= center_z <= z_max):
        raise ValueError(
            f"Wake centre ({center_y:.3f}, {center_z:.3f}) m lies outside the "
            f"traverse window [{y_min}, {y_max}] x [{z_min}, {z_max}] m."
        )
    return min(
        center_y - y_min,
        y_max - center_y,
        center_z - z_min,
        z_max - center_z,
    )


def concentric_ring_positions(
    center_y: float,
    center_z: float,
    max_radius: float,
    points_per_ring: tuple[int, ...],
) -> list[tuple[float, float]]:
    """
    (Y, Z) points on concentric circles centred on the wake centre.

    `points_per_ring` gives the number of points on each ring from the innermost
    to the outermost; its length is the number of rings. Ring radii use
    equal-AREA spacing

        r_i = max_radius * sqrt(i / n_rings),   i = 1 .. n_rings

    so the rings sample the disk roughly uniformly per unit area. Each ring's
    points are spread evenly in angle with a half-step offset, which keeps every
    point off the Z = 0 axis (so the rings never duplicate the hub-height line
    points). Use an even number of points per ring to guarantee this.
    """
    n_rings = len(points_per_ring)
    if n_rings < 1:
        raise ValueError("Need at least one ring.")
    positions: list[tuple[float, float]] = []
    for ring_index, point_count in enumerate(points_per_ring, start=1):
        if point_count < 1:
            raise ValueError("Each ring needs at least one point.")
        radius = max_radius * math.sqrt(ring_index / n_rings)
        for k in range(point_count):
            angle = (k + 0.5) * 2.0 * math.pi / point_count
            y = center_y + radius * math.cos(angle)
            z = center_z + radius * math.sin(angle)
            positions.append((y, z))
    return positions
