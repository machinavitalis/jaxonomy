# SPDX-License-Identifier: MIT

"""The registry of shipped notebooks and how each one is executed.

`docs/**` is a shippable surface: the notebooks under it are the first thing a
new user runs, and their committed outputs are what the MkDocs site publishes.
Nothing re-executes them, so an API rename can leave a notebook broken while its
stale outputs still look correct on the docs site. This manifest is the single
source of truth for the execution gate that closes that hole.

Every notebook on disk must appear here exactly once — `test_notebook_manifest`
fails otherwise, so a newly added notebook is red until someone classifies it.

Tiers
-----
``SMOKE``   Fast, high-traffic notebooks. Unmarked, so they run in the default
            (pull-request) tier. Keep the whole tier under ~60 s: this is the
            gate that catches a core API rename on the PR that causes it.
``WEEKLY``  Everything else. Carries the ``notebook`` marker, which
            ``pytest.ini`` deselects by default, so these run only in the weekly
            CI job (and locally via ``pytest -m notebook``).

Fields
------
``timeout``  Per-notebook seconds. Required because ``pytest.ini`` caps every
             test at 180 s globally and several notebooks legitimately exceed
             it; the value here is roughly 3x the measured local runtime, to
             leave headroom on slower CI hardware.
``requires`` Modules to ``importorskip`` before executing. Optional heavy deps
             (torch, mujoco, cyipopt, ...) are installed in the weekly job but
             not in the default test extra, so a notebook needing one skips
             cleanly rather than failing where it cannot run.
``binaries`` External programs the notebook shells out to (``ffmpeg`` for the
             animation writers). Skipped when absent from ``PATH``.
``artifacts`` Files the notebook reads but does not itself produce, relative to
             its run directory. A notebook that needs one of these cannot run
             from a clean checkout; the entry documents that rather than hiding
             it behind a generic failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPO_ROOT / "docs"

SMOKE = "smoke"
WEEKLY = "weekly"
TIERS = (SMOKE, WEEKLY)


@dataclass(frozen=True)
class Notebook:
    """How to execute one shipped notebook."""

    path: str
    tier: str
    timeout: int
    requires: tuple[str, ...] = field(default_factory=tuple)
    binaries: tuple[str, ...] = field(default_factory=tuple)
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"{self.path}: unknown tier {self.tier!r}")

    @property
    def full_path(self) -> Path:
        return REPO_ROOT / self.path

    @property
    def run_dir(self) -> Path:
        """Every notebook executes from its own directory.

        That is the convention the corpus already assumes: 31 notebooks read
        ``media/...`` relative to themselves, ``conservation_laws_as_ci``
        resolves the repo root as ``getcwd()/../..``, and
        ``f1_part_6_drivaerml_hero`` asserts outright that it was run from
        ``docs/examples/``. Running from the repository root instead breaks all
        three groups.
        """
        return self.full_path.parent


def discovered_notebooks() -> list[str]:
    """Every notebook under ``docs/``, as repo-relative posix paths."""
    return sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in NOTEBOOK_ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )


MANIFEST: tuple[Notebook, ...] = (
    Notebook(
        "docs/examples/01_quadcopter_modelling.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/02_quadcopter_trajectory_generation.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/03_quadcopter_nonlinear_mpc.ipynb",
        WEEKLY,
        timeout=240,
    ),
    Notebook(
        "docs/examples/04_flip_trajectory_optimization.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/MLP_training.ipynb",
        WEEKLY,
        timeout=180,
        requires=("sklearn",),
    ),
    Notebook(
        "docs/examples/actuator_delay_identification.ipynb",
        WEEKLY,
        timeout=240,
    ),
    Notebook(
        "docs/examples/aerospace_adcs_ekf_wheels.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/aleatoric_vs_epistemic_uq.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/artificial_pancreas_mpc.ipynb",
        WEEKLY,
        timeout=720,
    ),
    Notebook(
        "docs/examples/battery_pack_10k_scaling.ipynb",
        WEEKLY,
        timeout=180,
        requires=("psutil",),
    ),
    Notebook(
        "docs/examples/battery_pack_thermal.ipynb",
        WEEKLY,
        timeout=420,
    ),
    Notebook(
        "docs/examples/battery_part_1_ecm_model.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/battery_part_2_parameter_estimation_synthetic_data.ipynb",
        WEEKLY,
        timeout=900,
    ),
    Notebook(
        "docs/examples/battery_part_3_parameter_estimation_real_data.ipynb",
        WEEKLY,
        timeout=1320,
    ),
    Notebook(
        "docs/examples/battery_part_4_data_driven_models_DMDc.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/battery_part_5_data_driven_models_eDMDc.ipynb",
        WEEKLY,
        timeout=240,
    ),
    Notebook(
        "docs/examples/battery_part_6_data_driven_models_SINDyc.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pysindy",),
        note=(
            "Passes on the pinned pysindy ~=1.7.5. Fails on pysindy >=2, which "
            "rejects the t=None that jaxonomy.library.Sindy forwards for "
            "discrete-time fits — a library incompatibility, not a notebook one."
        ),
    ),
    Notebook(
        "docs/examples/battery_part_7_data_driven_models_Neural_Networks.ipynb",
        WEEKLY,
        timeout=480,
    ),
    Notebook(
        "docs/examples/booster_part_1_modeling.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/booster_part_2_mpc_and_render.ipynb",
        WEEKLY,
        timeout=360,
        requires=("PIL", "cyipopt", "mujoco"),
    ),
    Notebook(
        "docs/examples/booster_part_3_atmosphere_and_phases.ipynb",
        WEEKLY,
        timeout=3900,
        requires=("cyipopt",),
    ),
    Notebook(
        "docs/examples/booster_part_4_high_fidelity_propulsion.ipynb",
        WEEKLY,
        timeout=7440,
        requires=("cyipopt",),
    ),
    Notebook(
        "docs/examples/booster_part_5_sensing_and_estimation.ipynb",
        WEEKLY,
        timeout=2100,
        requires=("cyipopt",),
    ),
    Notebook(
        "docs/examples/booster_part_6_gnc_validation_and_analysis.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/bouncing_ball.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/examples/conservation_laws_as_ci.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/container_blocks_tour.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/examples/custom_block_authoring.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/dae_projection_pendulum.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/dfig_wind_turbine.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/diagram_visualization.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/differentiable_audio_dsp.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/dpc_two_tank_reference_tracking.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/ebike_part2_optimization.ipynb",
        WEEKLY,
        timeout=240,
    ),
    Notebook(
        "docs/examples/ebike_part3_pybamm_battery.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pybamm",),
    ),
    Notebook(
        "docs/examples/ebike_part4_mujoco_multibody.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/ebike_part5_openfoam_cfd.ipynb",
        WEEKLY,
        timeout=420,
    ),
    Notebook(
        "docs/examples/ebike_part1_smart_cargo.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/energy_shaping_and_lqr.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/engine_map_fitting_to_mpc.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/f1_part_1_lap_time_simulator.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/f1_part_2_setup_optimization.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/f1_part_3_aero_map_fitting.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/f1_part_4_sobol_cfd_budget.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/f1_part_5_naca_su2_cosim.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/f1_part_6_drivaerml_hero.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/fast_restart_and_batched_sweeps.ipynb",
        WEEKLY,
        timeout=660,
    ),
    Notebook(
        "docs/examples/fmi_export_roundtrip.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pythonfmu",),
    ),
    Notebook(
        "docs/examples/grid_forming_microgrid.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/hl20_glide_autopilot.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/examples/hybrid_ml_physics_predictor.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/hybrid_trajopt_through_events.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/limit_cycles.ipynb",
        WEEKLY,
        timeout=180,
        requires=("cyipopt",),
    ),
    Notebook(
        "docs/examples/linear_mpc.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/linearization_workflow.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/lqr.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/examples/motor_part_1_pmsm_modeling.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/motor_part_2_field_oriented_control.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/motor_part_3_thermal_and_derating.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/motor_part_4_system_identification.ipynb",
        WEEKLY,
        timeout=180,
        requires=("jaxterity",),
        note=(
            "Imports the downstream jaxterity package. Runs wherever that is "
            "installed (it passes locally); skips on CI, which does not install "
            "it — jaxonomy sits upstream and its CI must not depend on a repo "
            "further down the stack."
        ),
    ),
    Notebook(
        "docs/examples/motor_part_5_design_margins.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/motor_part_6_embedded_deployment.ipynb",
        WEEKLY,
        timeout=180,
        requires=("casadi", "jaxility"),
        note=(
            "Imports the downstream jaxility package; same situation as "
            "motor_part_4 above — runs locally, skips on CI."
        ),
    ),
    Notebook(
        "docs/examples/mujoco/pick_and_place.ipynb",
        WEEKLY,
        timeout=180,
        requires=("mediapy", "mujoco"),
    ),
    Notebook(
        "docs/examples/multi_domain_hvac.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/multirate_controller.ipynb",
        WEEKLY,
        timeout=180,
        requires=("graphviz",),
    ),
    Notebook(
        "docs/examples/neural_dae_pendulum_drag.ipynb",
        WEEKLY,
        timeout=720,
    ),
    Notebook(
        "docs/examples/openmodelica_plant_fmu_cosim.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pythonfmu",),
    ),
    Notebook(
        "docs/examples/pid_2dof_classical_tuning.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/pid_autotuning_interactive.ipynb",
        WEEKLY,
        timeout=180,
        # `%matplotlib widget` needs ipympl, not just ipywidgets, and it
        # overrides the inline backend the fixture sets.
        requires=("ipympl", "ipywidgets"),
    ),
    Notebook(
        "docs/examples/pid_tuning.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/pinn_across_stacks_part_1_policy_export.ipynb",
        WEEKLY,
        timeout=180,
        requires=("jaxonnxruntime", "neuromancer", "onnx", "onnxruntime", "torch"),
    ),
    Notebook(
        "docs/examples/pinn_across_stacks_part_2_neural_dae.ipynb",
        WEEKLY,
        timeout=180,
        requires=("neuromancer", "torch"),
    ),
    Notebook(
        "docs/examples/pinn_across_stacks_part_3_fmi_cosim.ipynb",
        WEEKLY,
        timeout=180,
        requires=("onnxruntime", "pythonfmu", "torch"),
    ),
    Notebook(
        "docs/examples/primitives.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/examples/product_family_variants.ipynb",
        WEEKLY,
        timeout=180,
    ),
    # The quanser/ series is hardware-in-the-loop: every notebook sets
    # HARDWARE = True and drives a physical QUBE-Servo through
    # library.QuanserHAL, which imports Quanser's proprietary `pal` SDK from
    # ~/Quanser/libraries/python. They cannot run on a CI runner at all, and the
    # `pal` guard is what expresses that -- it is deliberately conservative,
    # since the notebooks put the SDK on sys.path themselves, so these skip
    # unless `pal` is importable outright.
    Notebook(
        "docs/examples/quanser/01-plant-model.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pal",),
        binaries=("ffmpeg",),
    ),
    Notebook(
        "docs/examples/quanser/02-lqg.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pal",),
    ),
    Notebook(
        "docs/examples/quanser/03-energy-shaping.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pal",),
        binaries=("ffmpeg",),
        note=(
            "Sets HARDWARE = False, but still constructs library.QuanserHAL, whose "
            "__init__ imports the `pal` SDK unconditionally, so the `pal` gate is "
            "what keeps it out of CI. Its control.dlqr step used to fail on top of "
            "that, because discretize_forward_zoh returned an infinite B_d for this "
            "singular-A plant; that bug is fixed and covered by "
            "test/library/test_t_109_phase4_discretize.py."
        ),
    ),
    Notebook(
        "docs/examples/quanser/04-trajopt.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pal",),
        binaries=("ffmpeg",),
    ),
    Notebook(
        "docs/examples/quanser/05-nn-control.ipynb",
        WEEKLY,
        timeout=180,
        requires=("pal",),
        binaries=("ffmpeg",),
        artifacts=("models/swingup.pkl",),
        note=(
            "Loads a pre-trained policy that no cell in the notebook produces and "
            "that the repo does not track; it is written by running qube_nn.py as a "
            "script (1000 training epochs). Unreproducible from a clean checkout "
            "until the artifact is committed or a training cell is added."
        ),
    ),
    Notebook(
        "docs/examples/quantitative_model_slicing.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/realtime_fixed_step_controller.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/reproducibility_manifest.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/rl_environment_from_diagram.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/rom_balanced_truncation_thermal.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/rom_dmdc_koopman_mpc.ipynb",
        WEEKLY,
        timeout=180,
        requires=("osqp",),
    ),
    Notebook(
        "docs/examples/rom_pod_deim_reaction_diffusion.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/state_estimation_with_Kalman_filters.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/stiff_robertson_bdf.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/trajectory_optimization_and_stabilization.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/truth_table_gear_logic.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/examples/ude_and_sr_lotka_volterra.ipynb",
        WEEKLY,
        timeout=180,
        requires=("gplearn", "pysindy"),
    ),
    Notebook(
        "docs/examples/unit_safe_wiring.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/examples/vehicle_handling_autodiff.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/tutorials/01-getting-started.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/tutorials/02-creating-custom-blocks.ipynb",
        SMOKE,
        timeout=90,
    ),
    Notebook(
        "docs/tutorials/03-creating-custom-acausal-components.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/tutorials/04-wrappers.ipynb",
        WEEKLY,
        timeout=180,
    ),
    Notebook(
        "docs/tutorials/05-automatic-differentiation-optimization.ipynb",
        WEEKLY,
        timeout=180,
        requires=("cyipopt",),
    ),
)


def by_tier(tier: str) -> list[Notebook]:
    return [nb for nb in MANIFEST if nb.tier == tier]


def manifest_paths() -> list[str]:
    return sorted(nb.path for nb in MANIFEST)
