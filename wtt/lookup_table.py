"""
Loader and interpolator for the G1 aerodynamic look-up table (LookUpTable_G1.mat).

This module is the single source of truth for everything that comes out of the
.mat file: blade geometry (radial stations and chord) and the aerodynamic
coefficient surfaces as functions of (TSR, blade pitch).

IMPORTANT field-naming clarification (verified against the data):
    * `Cp`    -> power coefficient            C_P(TSR, pitch)
    * `Cf`    -> THRUST coefficient           C_T(TSR, pitch)   (peaks ~0.88)
    * `Ct`    -> TORQUE coefficient           C_Q(TSR, pitch)   (== Cp / TSR)
    * `Axial` -> axial induction factor a     a(TSR, pitch)
The names in the .mat file are misleading: the field literally called `Ct` is
the torque coefficient, while the true thrust coefficient is the field `Cf`.
This was confirmed numerically (Cf*... gives 4a(1-a); Ct*TSR == Cp).

We use index 2 of the `LookUp` array, which corresponds to Reynolds = 75000
(the `RN` vector is [50000, 60000, 75000, 90000]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.io
from scipy.interpolate import RegularGridInterpolator


# Index into Reynolds.LookUp that corresponds to Re = 75000 (RN[2]).
_REYNOLDS_75000_INDEX: int = 2


@dataclass(frozen=True)
class OptimumOperatingPoint:
    """The (TSR, pitch) at which the power coefficient is maximal in the table."""

    tsr: float
    pitch_deg: float
    cp: float


class G1LookUpTable:
    """
    Wraps the G1 look-up table and provides clean accessors.

    Construction loads the file and extracts the Re = 75000 slice. All heavy data
    (grids and coefficient surfaces) are stored once; interpolators are built
    lazily on first use.
    """

    def __init__(self, file_path: str) -> None:
        # Exactly the access pattern prescribed by the assignment.
        mat_data = scipy.io.loadmat(
            file_path, squeeze_me=True, struct_as_record=False
        )
        reynolds_struct = mat_data["Reynolds"]

        # Use index 2 for the high-Re case (Re = 75,000).
        data = reynolds_struct.LookUp[_REYNOLDS_75000_INDEX]

        # --- Blade geometry -------------------------------------------------
        self._radial_stations = np.asarray(data.rm, dtype=float)  # [m]
        self._chord = np.asarray(data.cord, dtype=float)  # [m]

        # --- Coefficient grid axes -----------------------------------------
        self._tsr_axis = np.asarray(data.TSR, dtype=float)  # [-]
        self._pitch_axis = np.asarray(data.Pitch, dtype=float)  # [deg]

        # --- Coefficient surfaces, indexed [TSR, pitch] ---------------------
        self._cp_surface = np.asarray(data.Cp, dtype=float)  # power coeff
        self._thrust_surface = np.asarray(data.Cf, dtype=float)  # THRUST coeff
        self._torque_surface = np.asarray(data.Ct, dtype=float)  # torque coeff
        self._axial_surface = np.asarray(data.Axial, dtype=float)  # axial induction a

        # Lazily-created interpolators (built on first request).
        self._cp_interp: RegularGridInterpolator | None = None
        self._thrust_interp: RegularGridInterpolator | None = None
        self._torque_interp: RegularGridInterpolator | None = None
        self._axial_interp: RegularGridInterpolator | None = None

    # ----------------------------------------------------------------------
    # Geometry accessors
    # ----------------------------------------------------------------------
    def tip_radius(self) -> float:
        """Rotor tip radius = outermost radial station (single formula)."""
        return float(self._radial_stations[-1])

    def rotor_diameter(self) -> float:
        """Rotor diameter = 2 * tip radius."""
        return 2.0 * self.tip_radius()

    def midspan_radius(self) -> float:
        """Radius at r/R = 0.5 (nearest tabulated station)."""
        target = 0.5 * self.tip_radius()
        index = int(np.argmin(np.abs(self._radial_stations - target)))
        return float(self._radial_stations[index])

    def midspan_chord(self) -> float:
        """
        Chord at the r/R = 0.5 station.

        This is the reference chord used for the Reynolds-number calibration, in
        agreement with the assignment example (chord ~ 5 cm at r/R = 0.5).
        """
        target = 0.5 * self.tip_radius()
        index = int(np.argmin(np.abs(self._radial_stations - target)))
        return float(self._chord[index])

    # ----------------------------------------------------------------------
    # Interpolator construction (one private helper, reused for every surface)
    # ----------------------------------------------------------------------
    def _make_interpolator(self, surface: np.ndarray) -> RegularGridInterpolator:
        """
        Build a 2-D interpolator on the (TSR, pitch) grid for one surface.

        `bounds_error=True` is deliberate: asking for a point outside the
        tabulated range must fail loudly rather than silently extrapolating.
        """
        return RegularGridInterpolator(
            (self._tsr_axis, self._pitch_axis),
            surface,
            method="linear",
            bounds_error=True,
        )

    # ----------------------------------------------------------------------
    # Coefficient look-ups (TSR, pitch) -> coefficient
    # ----------------------------------------------------------------------
    def power_coefficient(self, tsr: float, pitch_deg: float) -> float:
        """Interpolated power coefficient C_P(TSR, pitch)."""
        if self._cp_interp is None:
            self._cp_interp = self._make_interpolator(self._cp_surface)
        return float(self._cp_interp((tsr, pitch_deg)))

    def thrust_coefficient(self, tsr: float, pitch_deg: float) -> float:
        """Interpolated THRUST coefficient C_T(TSR, pitch) (field `Cf`)."""
        if self._thrust_interp is None:
            self._thrust_interp = self._make_interpolator(self._thrust_surface)
        return float(self._thrust_interp((tsr, pitch_deg)))

    def torque_coefficient(self, tsr: float, pitch_deg: float) -> float:
        """Interpolated TORQUE coefficient C_Q(TSR, pitch) (field `Ct`)."""
        if self._torque_interp is None:
            self._torque_interp = self._make_interpolator(self._torque_surface)
        return float(self._torque_interp((tsr, pitch_deg)))

    def axial_induction(self, tsr: float, pitch_deg: float) -> float:
        """Interpolated axial induction factor a(TSR, pitch) (field `Axial`)."""
        if self._axial_interp is None:
            self._axial_interp = self._make_interpolator(self._axial_surface)
        return float(self._axial_interp((tsr, pitch_deg)))

    # ----------------------------------------------------------------------
    # Optimum search
    # ----------------------------------------------------------------------
    def optimum_operating_point(self) -> OptimumOperatingPoint:
        """
        Locate the (TSR, pitch) of maximum power coefficient on the table grid.

        Returns the tabulated grid point (no interpolation) so the result is
        exactly reproducible from the raw data.
        """
        flat_index = int(np.nanargmax(self._cp_surface))
        i_tsr, i_pitch = np.unravel_index(flat_index, self._cp_surface.shape)
        return OptimumOperatingPoint(
            tsr=float(self._tsr_axis[i_tsr]),
            pitch_deg=float(self._pitch_axis[i_pitch]),
            cp=float(self._cp_surface[i_tsr, i_pitch]),
        )

    # Bounds, exposed so callers can keep their TSR sweeps inside the table.
    def tsr_bounds(self) -> tuple[float, float]:
        """Smallest and largest tabulated TSR values."""
        return float(self._tsr_axis[0]), float(self._tsr_axis[-1])
