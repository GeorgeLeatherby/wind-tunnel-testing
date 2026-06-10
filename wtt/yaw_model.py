"""
Yaw-dependent performance correction.

The bare look-up table gives the rotor coefficients C_P and C_T as functions of
(TSR, pitch) only — it contains no yaw dependence. Physically, yawing the rotor
must reduce the extracted power and thrust, because the rotor disk only processes
the wind component **normal** to its plane:

    U_perp = U_inf * cos(gamma)

Starting from the 1-D actuator-disk momentum theory of Lecture 3 (1-D Annular
Stream Tube Theory) and simply replacing the freestream speed by its normal
component, the dimensional loads become

    T = 1/2 * rho * A * (U cos gamma)^2 * 4a(1-a)      ->  C_T(gamma) = C_T(0) cos^2(gamma)
    P = 1/2 * rho * A * (U cos gamma)^3 * 4a(1-a)^2    ->  C_P(gamma) = C_P(0) cos^3(gamma)

i.e. a *cosine loss* cos^p(gamma) with the momentum-theory exponents p = 2 for
thrust and p = 3 for power. (Empirically-fitted exponents are often slightly
lower; the exponents are therefore configurable.)

Sign asymmetry (+gamma vs -gamma)
---------------------------------
The cosine loss alone is symmetric in the yaw sign (cos is even), so it would
predict identical performance at +30 deg and -30 deg. In reality the rotor spins
in a *fixed* direction, so the two yaw signs are not equivalent: the in-plane
wind component

    U_t = U_inf * sin(gamma)

sweeps across a clockwise- (or counter-clockwise-) turning rotor differently for
+gamma and -gamma, producing an advancing/retreating-blade asymmetry analogous to
a helicopter rotor in forward flight. The governing non-dimensional group is the
advance ratio

    mu = U_t / (Omega R) = sin(gamma) / lambda ,

which is **odd** in gamma (mu(+gamma) = -mu(-gamma)). To leading order the
azimuth-averaged loads pick up a correction linear in mu:

    asymmetry(gamma, lambda) = 1 + k_gamma * sin(gamma) / lambda ,

with k_gamma a dimensionless rotational-coupling sensitivity. At gamma = 0 the
factor is exactly 1 (no asymmetry). The magnitude and sign of k_gamma depend on
the rotor and the yaw-sign convention and should be **calibrated against
measurement**; it is exposed as an explicit configuration parameter (no hidden
default).

Combined factors applied to the table coefficients:

    f_P(gamma, lambda) = cos^{p_P}(gamma) * (1 + k_gamma * sin(gamma)/lambda)
    f_T(gamma, lambda) = cos^{p_T}(gamma) * (1 + k_gamma * sin(gamma)/lambda)

Each elementary formula is its own function.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Elementary formulas
# ---------------------------------------------------------------------------
def cosine_loss(yaw_angle_rad: float, exponent: float) -> float:
    """
    Cosine-law loss factor cos^exponent(gamma).

    Derived from replacing the freestream speed by its rotor-normal component
    U cos(gamma) in actuator-disk momentum theory (Lecture 3). Use exponent = 3
    for power and exponent = 2 for thrust.
    """
    return math.cos(yaw_angle_rad) ** exponent


def advance_ratio(yaw_angle_rad: float, tsr: float) -> float:
    """
    Rotor advance ratio mu = sin(gamma) / lambda (single formula).

    Ratio of the in-plane wind component U sin(gamma) to the blade tip speed
    Omega R = lambda U. Odd in gamma, so it carries the +/- yaw sign that breaks
    the symmetry for a fixed rotation direction.
    """
    return math.sin(yaw_angle_rad) / tsr


def rotational_asymmetry_factor(
    yaw_angle_rad: float, tsr: float, sensitivity: float
) -> float:
    """
    Sign-dependent asymmetry factor 1 + k_gamma * mu.

    `sensitivity` (k_gamma) scales the advancing/retreating-blade coupling. The
    factor equals 1 at zero yaw and differs between +gamma and -gamma because the
    advance ratio is odd in gamma.
    """
    return 1.0 + sensitivity * advance_ratio(yaw_angle_rad, tsr)


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------
class YawPerformanceModel:
    """
    Multiplicative yaw correction for the power and thrust coefficients.

    Holds the (configurable) cosine exponents and the rotational-asymmetry
    sensitivity, and exposes one factor for power and one for thrust. The factors
    are meant to multiply the yaw-free look-up-table coefficients.
    """

    def __init__(
        self,
        power_cosine_exponent: float,
        thrust_cosine_exponent: float,
        rotational_asymmetry: float,
    ) -> None:
        self._power_exponent = power_cosine_exponent
        self._thrust_exponent = thrust_cosine_exponent
        self._asymmetry = rotational_asymmetry

    def power_factor(self, yaw_angle_rad: float, tsr: float) -> float:
        """Combined power-coefficient factor cos^{p_P}(gamma) * (1 + k_gamma mu)."""
        return cosine_loss(yaw_angle_rad, self._power_exponent) * (
            rotational_asymmetry_factor(yaw_angle_rad, tsr, self._asymmetry)
        )

    def thrust_factor(self, yaw_angle_rad: float, tsr: float) -> float:
        """Combined thrust-coefficient factor cos^{p_T}(gamma) * (1 + k_gamma mu)."""
        return cosine_loss(yaw_angle_rad, self._thrust_exponent) * (
            rotational_asymmetry_factor(yaw_angle_rad, tsr, self._asymmetry)
        )
