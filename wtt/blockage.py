"""
Wind-tunnel wall-blockage correction (Glauert approach).

Background (from the lecture slides, "Blockage Correction Approaches"):
the tunnel walls reduce the area available to the flow, which speeds up the flow
near the model. The Glauert wake-blockage relation links the (higher) equivalent
free-air speed U'_inf to the actual speed U_inf measured near the model:

        U'_inf / U_inf = 1 + (alpha / 4) * C_T / (1 - C_T)

with
        C_T   = thrust coefficient  =  T / (0.5 rho A_R U_inf^2)
        alpha = blockage ratio      =  A_rotor / A_tunnel

Direction of use in THIS project
--------------------------------
We first decide the *equivalent free-air* wind speed `U'_inf` that the rotor
should "feel" (this is the speed that calibrates the Reynolds number, because the
velocity triangle and the coefficient tables are defined in free-air terms).
The wind tunnel, however, must be commanded to a *lower* speed `U_inf`, because
its walls will accelerate the flow back up to the equivalent free-air value at
the model. Hence:

        U_inf (commanded) = U'_inf / (1 + (alpha/4) * C_T / (1 - C_T))

Each formula lives in its own function so it can be checked independently.
"""

from __future__ import annotations


def blockage_ratio(rotor_area: float, tunnel_area: float) -> float:
    """
    Blockage ratio alpha = A_rotor / A_tunnel (single formula).

    Should stay below ~10 % for the correction (and the test) to be meaningful.
    """
    return rotor_area / tunnel_area


def glauert_speed_ratio(thrust_coefficient: float, blockage_ratio_value: float) -> float:
    """
    Glauert factor  U'_inf / U_inf = 1 + (alpha / 4) * C_T / (1 - C_T).

    `thrust_coefficient` must be the rotor THRUST coefficient (not torque).
    """
    return 1.0 + (blockage_ratio_value / 4.0) * (
        thrust_coefficient / (1.0 - thrust_coefficient)
    )


def commanded_tunnel_speed(
    equivalent_free_air_speed: float,
    thrust_coefficient: float,
    blockage_ratio_value: float,
) -> float:
    """
    Wind-tunnel speed to command so the model feels `equivalent_free_air_speed`.

    U_inf = U'_inf / (Glauert factor)
    """
    ratio = glauert_speed_ratio(
        thrust_coefficient=thrust_coefficient,
        blockage_ratio_value=blockage_ratio_value,
    )
    return equivalent_free_air_speed / ratio


def equivalent_free_air_speed(
    commanded_tunnel_speed_value: float,
    thrust_coefficient: float,
    blockage_ratio_value: float,
) -> float:
    """
    Inverse of `commanded_tunnel_speed`: free-air speed felt by the model given
    the speed actually set in the tunnel.

    U'_inf = U_inf * (Glauert factor)
    """
    ratio = glauert_speed_ratio(
        thrust_coefficient=thrust_coefficient,
        blockage_ratio_value=blockage_ratio_value,
    )
    return commanded_tunnel_speed_value * ratio
