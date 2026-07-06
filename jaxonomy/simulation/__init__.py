# SPDX-License-Identifier: MIT

from .batch import (
    attach_provenance_to_batch,
    BatchSimulationResults,
    simulate_batch,
)
from .fast_restart import FastRestartSimulator, fast_restart
from .simulate_distributed import simulate_distributed
from .simulator import (
    estimate_max_major_steps,
    Simulator,
    simulate,
    simulate_jacfwd,
    scalar_cost_simulate,
)
from .simulate_variants import simulate_variant_sweep
from .static_sweep import simulate_static_sweep
from .cloud_runner import simulate_cloud
from .testing_systems import Decay
from ..backend import ODESolver, ODESolverOptions


from .dae_drift import (
    algebraic_row_mask,
    compute_constraint_residual,
    constraint_residual_norm,
)
from .errors import SimulationError
from .event_gradient import (
    event_time_gradient,
    event_time_jacobian,
    event_times_gradient,
    multi_event_time_gradient,
    simulate_with_event_time_grad,
    vmap_event_time_gradient,
    vmap_event_times_gradient,
)
from .lazy_results import LazyResults
from .provenance import (
    bundle_results,
    compare_manifests,
    compute_provenance,
    load_manifest,
    ManifestMismatch,
    ProvenanceManifest,
    ResultsWithProvenance,
    verify_manifest,
)
from .types import (
    ResultsOptions,
    SimulationResults,
    SimulatorOptions,
    ResultsMode,
)

__all__ = [
    "algebraic_row_mask",
    "attach_provenance_to_batch",
    "BatchSimulationResults",
    "bundle_results",
    "compare_manifests",
    "compute_constraint_residual",
    "compute_provenance",
    "constraint_residual_norm",
    "estimate_max_major_steps",
    "event_time_gradient",
    "event_time_jacobian",
    "event_times_gradient",
    "fast_restart",
    "FastRestartSimulator",
    "Decay",
    "load_manifest",
    "ManifestMismatch",
    "multi_event_time_gradient",
    "ODESolver",
    "ODESolverOptions",
    "LazyResults",
    "ProvenanceManifest",
    "ResultsOptions",
    "ResultsMode",
    "ResultsWithProvenance",
    "SimulationError",
    "SimulationResults",
    "simulate",
    "simulate_batch",
    "simulate_distributed",
    "simulate_cloud",
    "simulate_jacfwd",
    "scalar_cost_simulate",
    "simulate_static_sweep",
    "simulate_variant_sweep",
    "simulate_with_event_time_grad",
    "Simulator",
    "SimulatorOptions",
    "verify_manifest",
    "vmap_event_time_gradient",
    "vmap_event_times_gradient",
]
