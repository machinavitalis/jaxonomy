# SPDX-License-Identifier: MIT

from .framework import (
    DistributionConfig,
    Evosax,
    Optax,
    OptaxWithStochasticVars,
    Optimizable,
    OptimizableWithStochasticVars,
    OptimizationResult,
    Scipy,
    NLopt,
    IPOPT,
    Transform,
    CompositeTransform,
    IdentityTransform,
    LogTransform,
    LogitTransform,
    NegativeNegativeLogTransform,
    NormalizeTransform,
)
from .multi_start import MultiStart, MultiStartResult
from .sensitivity import compute_sensitivity, SensitivityResult
from .confidence import compute_confidence_intervals, ConfidenceIntervalResult
from .objectives import (
    ise_objective,
    lqr_objective,
    tracking_mse,
    weighted_sum,
)
from .pid_autotuning import AutoTuner
from .training import Trainer
from .parameter_tuning import tune_parameters, TuningResult
from .implicit import implicit_solver

# RLEnv requires optional heavy dependencies (flax, brax).
# Import lazily so missing deps don't break the rest of jaxonomy.optimization.
try:
    from .rl_env import RLEnv
except ImportError:  # flax / brax not installed
    RLEnv = None  # type: ignore[assignment,misc]

__all__ = [
    "Trainer",
    "Optimizable",
    "OptimizableWithStochasticVars",
    "OptimizationResult",
    "Optax",
    "OptaxWithStochasticVars",
    "Scipy",
    "Evosax",
    "NLopt",
    "IPOPT",
    "DistributionConfig",
    "AutoTuner",
    "tune_parameters",
    "TuningResult",
    "implicit_solver",
    "MultiStart",
    "MultiStartResult",
    "compute_sensitivity",
    "SensitivityResult",
    "compute_confidence_intervals",
    "ConfidenceIntervalResult",
    "ise_objective",
    "lqr_objective",
    "tracking_mse",
    "weighted_sum",
    "Transform",
    "CompositeTransform",
    "IdentityTransform",
    "LogTransform",
    "LogitTransform",
    "NegativeNegativeLogTransform",
    "NormalizeTransform",
    "RLEnv",
]
