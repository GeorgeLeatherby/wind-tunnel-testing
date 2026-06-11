"""
wtt — Wind-Tunnel Testing test-matrix generator for the G1 model (Group 2).

Public building blocks are imported here for convenience. The end-to-end
generation is driven by `generate_test_matrices.py` in the project root.
"""

from __future__ import annotations

from .air_properties import AirProperties
from .campaign_config import CampaignConfig, build_group2_campaign
from .lookup_table import G1LookUpTable
from .operating_point import OperatingPointSolver
from .reynolds_calibration import ReynoldsCalibrator
from .task1_builder import Task1PerformanceBuilder
from .task2_builder import Task2WakeBuilder
from .excel_writer import TestMatrixExcelWriter
from .traverse_writer import TraversePositionWriter

__all__ = [
    "AirProperties",
    "CampaignConfig",
    "build_group2_campaign",
    "G1LookUpTable",
    "OperatingPointSolver",
    "ReynoldsCalibrator",
    "Task1PerformanceBuilder",
    "Task2WakeBuilder",
    "TestMatrixExcelWriter",
    "TraversePositionWriter",
]
