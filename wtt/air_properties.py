"""
Physical constants and air properties for the wind-tunnel test-matrix generator.

Strict rule for this whole project: NO fallbacks, NO silent defaults.
Every quantity that a calculation needs must be supplied explicitly. If a value
is missing, the code is expected to raise (KeyError / TypeError) on purpose so
that the mistake is caught immediately instead of producing a wrong number.

The two "looked-up" standard quantities below (kinematic viscosity of air and
the specific gas constant of dry air) are physical constants of nature, not
tuning defaults, so they are defined here once as named constants. They are the
values for dry air at 20 [degC] and standard atmospheric pressure.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- Standard physical constants (dry air, 20 degC, 101325 Pa) -----------------

# Kinematic viscosity of air at 20 degC and 1 atm.
# Source: standard engineering tables (nu = mu / rho), nu_air(20C) ~ 1.516e-5 m^2/s.
KINEMATIC_VISCOSITY_AIR_20C: float = 1.516e-5  # [m^2 / s]

# Specific gas constant for dry air.
SPECIFIC_GAS_CONSTANT_DRY_AIR: float = 287.058  # [J / (kg K)]

# Reference ambient conditions used to derive the air density via the ideal-gas law.
REFERENCE_TEMPERATURE_CELSIUS: float = 20.0  # [degC]
REFERENCE_PRESSURE_PA: float = 101325.0  # [Pa]

# Conversion helpers (named, so the intent is explicit at the call site).
CELSIUS_TO_KELVIN_OFFSET: float = 273.15  # [K]
RAD_PER_SECOND_TO_RPM: float = 60.0 / (2.0 * 3.141592653589793)  # [rpm / (rad/s)]


def celsius_to_kelvin(temperature_celsius: float) -> float:
    """Convert a temperature from degrees Celsius to Kelvin (single formula)."""
    return temperature_celsius + CELSIUS_TO_KELVIN_OFFSET


def air_density_ideal_gas(
    pressure_pa: float,
    temperature_celsius: float,
    specific_gas_constant: float,
) -> float:
    """
    Air density from the ideal-gas law: rho = p / (R * T).

    All three inputs are mandatory; passing None will raise on purpose.
    """
    temperature_kelvin = celsius_to_kelvin(temperature_celsius)
    return pressure_pa / (specific_gas_constant * temperature_kelvin)


@dataclass(frozen=True)
class AirProperties:
    """
    Immutable bundle of the air properties needed by the calculations.

    Built once from the reference ambient conditions and then passed around, so
    that every formula uses exactly the same density / viscosity. Using a frozen
    dataclass makes accidental mutation impossible.
    """

    density: float  # [kg / m^3]
    kinematic_viscosity: float  # [m^2 / s]

    @classmethod
    def at_reference_conditions(cls) -> "AirProperties":
        """Create the air properties for the documented reference state (20 degC, 1 atm)."""
        density = air_density_ideal_gas(
            pressure_pa=REFERENCE_PRESSURE_PA,
            temperature_celsius=REFERENCE_TEMPERATURE_CELSIUS,
            specific_gas_constant=SPECIFIC_GAS_CONSTANT_DRY_AIR,
        )
        return cls(
            density=density,
            kinematic_viscosity=KINEMATIC_VISCOSITY_AIR_20C,
        )
