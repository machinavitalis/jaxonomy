# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""Non-linear Model Predictive Control (NMPC) blocks.

A few of these blocks are implemented using CyIPOPT, which may require
additional installation steps:
- Install CyIPOPT on your system (eg. `sudo apt install coinor-libipopt-dev` on
  Ubuntu)
- `pip install 'pycollimator[nmpc]'`
"""

from .direct_shooting_ipopt_nmpc import DirectShootingNMPC
from .direct_transcription_ipopt_nmpc import DirectTranscriptionNMPC
from .hermite_simpson_ipopt_nmpc import HermiteSimpsonNMPC
from .trajectory_optimization import trajopt


__all__ = [
    "DirectShootingNMPC",
    "DirectTranscriptionNMPC",
    "HermiteSimpsonNMPC",
    "trajopt",
]
