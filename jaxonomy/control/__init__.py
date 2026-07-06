# SPDX-License-Identifier: MIT

"""Control sub-namespace for Jaxonomy.

This package collects control-flavoured helpers that compose with the
simulation primitives in :mod:`jaxonomy.simulation`. Currently it exposes
:mod:`jaxonomy.control.dpc` — a small differentiable-predictive-control
toolkit. Other control utilities (LQR, LQG, observers) live in
:mod:`jaxonomy.library` alongside the standard-library blocks rather than
here, by long-standing convention.
"""

from . import dpc

__all__ = ["dpc"]
