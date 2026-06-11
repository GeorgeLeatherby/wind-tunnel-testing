"""
Traverse-position file writer for Task 2 (Lecture 5 input format).

The wake-probe traversing system is fed a TAB-separated text file listing the
X / Y / Z positions to visit (millimetres, traverse frame) plus the turntable
yaw angle A (degrees). See Lecture 5, "Required Data for Wake Measurement
(Input)": the home position X=Y=Z=0 is at hub height, 0.5D behind the rotor, and
the table must be TAB-separated.

One line per measurement point, written in the same order as the Task-2 matrix so
the traverse visits the points in that sequence. No fallbacks: a row without a
probe position raises immediately.
"""

from __future__ import annotations

from .test_matrix_row import TestMatrixRow


# Header for the TAB-separated file (Lecture 5 convention: x y z, plus the
# turntable yaw A which differs per yaw block).
_HEADER = "x\ty\tz\tA"


class TraversePositionWriter:
    """Writes the Task-2 probe positions as a TAB-separated .txt file."""

    def write(self, rows: list[TestMatrixRow], file_path: str) -> None:
        """
        Write `rows` (which must carry probe_x/y/z) to a TAB-separated file.

        Columns: x, y, z [mm] and A [deg], where A is the turntable yaw of the
        point. Positions are written with 0.1 mm resolution; A as an integer.
        """
        lines = [_HEADER]
        for row in rows:
            if row.probe_x is None or row.probe_y is None or row.probe_z is None:
                raise ValueError(
                    f"Row {row.test_number} has no probe position; cannot write "
                    "the traverse file."
                )
            lines.append(
                f"{row.probe_x:.1f}\t{row.probe_y:.1f}\t"
                f"{row.probe_z:.1f}\t{row.yaw_deg:.0f}"
            )
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
