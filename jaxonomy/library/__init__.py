# SPDX-License-Identifier: MIT

from jaxonomy.backend.backend import IS_JAXLITE

# T-117-followup-bus-units: re-export BusUnit alongside BusCreator /
# BusSelector so callers can write ``from jaxonomy.library import
# BusUnit`` next to the bus blocks themselves.
from ..framework.units import BusUnit

from .generic import (
    SourceBlock,
    FeedthroughBlock,
    ReduceBlock,
)
from .primitives import (
    Abs,
    Arithmetic,
    Adder,
    Backlash,
    BandLimitedNoise,
    BusCreator,
    BusMerge,
    BusPassthrough,
    BusSelector,
    BusUpdate,
    bus_fields,
    flatten_bus,
    merge_buses,
    unflatten_bus,
    Chirp,
    Clock,
    Comparator,
    Constant,
    Counter,
    CrossProduct,
    DeadZone,
    DeadZoneInverse,
    Demultiplexer,
    Demux,
    DerivativeDiscrete,
    DiscreteClock,
    DiscreteInitializer,
    DotProduct,
    EdgeDetection,
    Exponent,
    FilterDiscrete,
    Gain,
    IfThenElse,
    Integrator,
    IntegratorDiscrete,
    IOPort,
    LeadLag,
    Logarithm,
    LogicalOperator,
    LogicalReduce,
    LookupTable1d,
    LookupTable2d,
    LookupTableND,
    LowPassDiscrete,
    MatrixConcatenation,
    MatrixInversion,
    MatrixMultiplication,
    MatrixTransposition,
    MinMax,
    Multiplexer,
    MultiPortSwitch,
    Mux,
    Notch,
    Offset,
    PIDController2DOF,
    PIDDiscrete,
    Power,
    Prelookup,
    PrelookupInverse,
    InterpolationUsingPrelookup,
    PRBS,
    PRBSLFSR,
    Product,
    ProductOfElements,
    Pulse,
    Quantizer,
    Ramp,
    RandomSource,
    RateLimiter,
    RateTransition,
    Decimator,
    Reciprocal,
    Relay,
    Saturate,
    Sawtooth,
    SoftRateLimiter,
    SoftSaturate,
    soft_dead_zone,
    soft_saturate,
    ScalarBroadcast,
    Sine,
    SignalDatatypeConversion,
    Slice,
    SquareRoot,
    Stack,
    Step,
    Stop,
    SumOfElements,
    Switch,
    TableSearch,
    TransportDelay,
    VariableTransportDelay,
    Trigonometric,
    TruthTable,
    TruthTableBuilder,
    UniformRandomNumber,
    UnitDelay,
    ZeroOrderHold,
)
from .battery_cell import BatteryCell
from .conditional import Conditional, WhenDisabled
from .replicated import ReplicatedFunction
# T-120 phase 1: Container Blocks (EnabledSubsystem, TriggeredSubsystem).
# The implementations live in jaxonomy/framework/containers.py to avoid
# touching jaxonomy/library/primitives.py (parallel work in T-112/T-121).
from ..framework.containers import (
    EnabledSubsystem,
    TriggeredSubsystem,
    ForEach,
    EnabledMode,
    EnabledStateMode,
    TriggerEdge,
)
from .custom import (
    CustomJaxBlock,
    CustomPythonBlock,
)
from .wrappers import (
    ode_block,
    feedthrough_block,
)
from .linear_system import (
    LTISystem,
    TransferFunction,
    linearize,
    PID,
    PIDContinuous,
    Derivative,
    LTISystemDiscrete,
    TransferFunctionDiscrete,
    LinearizedSystem,
)
from .linearize_container import linearize_to_lti

# T-124 phase 1 — differentiable lookup-table fitting.  Lives in its
# own module so the helper does not require a primitives.py edit (the
# parallel T-127 task owns those changes).  T-124-followup-grid-optimization
# adds ``fit_table_1d_with_grid`` which jointly optimises the grid
# placement AND the table values.
from .lookup_table_fitting import (
    fit_lookup_table_1d,
    fit_lookup_table_2d,
    fit_lookup_table_nd,
    fit_table_1d_with_grid,
    fit_table_2d,
    fit_table_nd,
)

from .neural_dae import add_neural_correction, NeuralDAEBlock

from .linearization_workflow import (
    FrequencyResponse,
    OperatingPoint,
    bode_data,
    discretize,
    estimate_frequency_response,
    findop,
    frequency_response,
    impulse_response,
    nyquist_data,
    pole_zero_map,
    step_response,
    with_observer,
)

from .random import (
    RandomNumber,
    WhiteNoise,
)

from .rotations import (
    CoordinateRotation,
    CoordinateRotationConversion,
    RigidBody,
)

from .data_source import (
    DataSource,
    SimulationResultsSource,
)
from .state_machine import (
    StateMachine,
)

from .reference_subdiagram import ReferenceSubdiagram

from .delay import (
    ShiftRegister,
    MaskedDelayBuffer,
)

if not IS_JAXLITE:
    from .mpc import (
        LinearDiscreteTimeMPC,
        LinearDiscreteTimeMPC_OSQP,
    )

    from .nmpc import (
        DirectShootingNMPC,
        DirectTranscriptionNMPC,
        HermiteSimpsonNMPC,
    )

    from .lqr import (
        LinearQuadraticRegulator,
        DiscreteTimeLinearQuadraticRegulator,
        FiniteHorizonLinearQuadraticRegulator,
    )

    from .lqg import LinearQuadraticGaussian

    from .mujoco import (
        MJX,
        MuJoCo,
    )

    from .nn import (
        MLP,
    )

    from .costs_and_losses import (
        QuadraticCost,
    )

    from .fmu_import import (
        ModelicaFMU,
    )

    from .sindy import (
        Sindy,
    )

    # Reduced-order modeling & statistical surrogates (T-143..T-151).
    # Grouped here alongside the other analysis-heavy blocks because linear
    # MOR pulls in scipy.linalg and the reduced/​surrogate blocks are JAX
    # LeafSystems. See docs/scope/rom.md.
    from .rom import (
        ReducedOrderModel,
        reduce,
        SnapshotData,
        collect_snapshots,
        relative_error,
        retained_energy,
        projection_error,
        controllability_gramian,
        observability_gramian,
        hankel_singular_values,
        balanced_realization,
        balanced_truncation,
        balred,
        minimal_realization,
        minreal,
        modal_truncation,
        residualize,
        pod_basis,
        galerkin_reduce,
        deim,
        deim_galerkin_reduce,
        DMDResult,
        DMDcResult,
        ERAResult,
        dmd,
        dmdc,
        era,
        DMDForecaster,
        identity_dictionary,
        polynomial_dictionary,
        rbf_dictionary,
        EDMDResult,
        edmd,
        KoopmanPredictor,
        GPModel,
        fit_gp,
        GaussianProcess,
        PCEModel,
        fit_pce,
        PolynomialChaos,
        RBFModel,
        fit_rbf,
        RadialBasisSurrogate,
    )

    from .ansys import (
        PyTwin,
    )

    from .ros2 import (
        Ros2Publisher,
        Ros2Subscriber,
    )

    from .state_estimators import (
        KalmanFilter,
        InfiniteHorizonKalmanFilter,
        ContinuousTimeInfiniteHorizonKalmanFilter,
        ExtendedKalmanFilter,
        UnscentedKalmanFilter,
        RecursiveLeastSquares,
        AugmentedStateEKF,
        Luenberger,
    )

    from .predictor import (
        PyTorch,
        TensorFlow,
        PyTorchPredictor,
        TensorFlowPredictor,
    )

    from .onnx_block import ONNX
    from .onnx_jax_block import ONNXJax

    from .fmu_export import (
        model_description_xml,
        write_model_description,
    )

    from .video import VideoSink, VideoSource

    from .quanser import QuanserHAL, QubeServoModel

else:
    # NOTE We could improve this by defining a different list based on Emscripten
    # vs. full environment. For now, we just raise an error at runtime. Much simpler.

    class JaxliteNotSupportedError(RuntimeError):
        pass

    class _InvalidBlock:
        def __init__(self, _class: str, *args, **kwargs):
            raise JaxliteNotSupportedError(
                f"Block not available with jaxlite: {_class}"
            )

    def _invalid(name):
        return lambda *args, **kwargs: _InvalidBlock(name, *args, **kwargs)

    LinearDiscreteTimeMPC = _invalid("LinearDiscreteTimeMPC")
    LinearDiscreteTimeMPC_OSQP = _invalid("LinearDiscreteTimeMPC_OSQP")
    DirectShootingNMPC = _invalid("DirectShootingNMPC")
    DirectTranscriptionNMPC = _invalid("DirectTranscriptionNMPC")
    HermiteSimpsonNMPC = _invalid("HermiteSimpsonNMPC")
    LinearQuadraticRegulator = _invalid("LinearQuadraticRegulator")
    DiscreteTimeLinearQuadraticRegulator = _invalid(
        "DiscreteTimeLinearQuadraticRegulator"
    )
    FiniteHorizonLinearQuadraticRegulator = _invalid(
        "FiniteHorizonLinearQuadraticRegulator"
    )
    LinearQuadraticGaussian = _invalid("LinearQuadraticGaussian")
    MJX = _invalid("MJX")
    MuJoCo = _invalid("MuJoCo")
    MLP = _invalid("MLP")
    QuadraticCost = _invalid("QuadraticCost")
    ModelicaFMU = _invalid("ModelicaFMU")
    Sindy = _invalid("Sindy")
    PyTwin = _invalid("PyTwin")
    Ros2Publisher = _invalid("Ros2Publisher")
    Ros2Subscriber = _invalid("Ros2Subscriber")
    KalmanFilter = _invalid("KalmanFilter")
    InfiniteHorizonKalmanFilter = _invalid("InfiniteHorizonKalmanFilter")
    ContinuousTimeInfiniteHorizonKalmanFilter = _invalid(
        "ContinuousTimeInfiniteHorizonKalmanFilter"
    )
    ExtendedKalmanFilter = _invalid("ExtendedKalmanFilter")
    UnscentedKalmanFilter = _invalid("UnscentedKalmanFilter")
    RecursiveLeastSquares = _invalid("RecursiveLeastSquares")
    AugmentedStateEKF = _invalid("AugmentedStateEKF")
    PyTorch = _invalid("PyTorch")
    TensorFlow = _invalid("TensorFlow")
    PyTorchPredictor = _invalid("PyTorchPredictor")
    TensorFlowPredictor = _invalid("TensorFlowPredictor")
    ONNX = _invalid("ONNX")
    ONNXJax = _invalid("ONNXJax")
    model_description_xml = _invalid("model_description_xml")
    write_model_description = _invalid("write_model_description")
    VideoSink = _invalid("VideoSink")
    VideoSource = _invalid("VideoSource")
    QuanserHAL = _invalid("QuanserHAL")
    QubeServoModel = _invalid("QubeServoModel")


__all__ = [
    "Arithmetic",
    "SourceBlock",
    "FeedthroughBlock",
    "ReduceBlock",
    "Abs",
    "Backlash",
    "Constant",
    "Sine",
    "BatteryCell",
    "BusCreator",
    "BusMerge",
    "BusPassthrough",
    "BusSelector",
    "BusUnit",
    "BusUpdate",
    "merge_buses",
    "Conditional",
    "WhenDisabled",
    "ReplicatedFunction",
    "EnabledSubsystem",
    "TriggeredSubsystem",
    "ForEach",
    "EnabledMode",
    "EnabledStateMode",
    "TriggerEdge",
    "Clock",
    "Comparator",
    "CoordinateRotation",
    "CoordinateRotationConversion",
    "Counter",
    "CrossProduct",
    "CustomJaxBlock",
    "CustomPythonBlock",
    "DataSource",
    "SimulationResultsSource",
    "DeadZone",
    "Derivative",
    "DerivativeDiscrete",
    "DiscreteInitializer",
    "DotProduct",
    "DiscreteClock",
    "EdgeDetection",
    "Exponent",
    "FilterDiscrete",
    "Gain",
    "IfThenElse",
    "Offset",
    "Reciprocal",
    "LogicalOperator",
    "LogicalReduce",
    "MatrixConcatenation",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "ModelicaFMU",
    "MinMax",
    "Multiplexer",
    "MultiPortSwitch",
    "Mux",
    "Notch",
    "Demultiplexer",
    "Demux",
    "Adder",
    "PID",
    "PIDContinuous",
    "Product",
    "ProductOfElements",
    "Power",
    "Integrator",
    "IntegratorDiscrete",
    "IOPort",
    "LeadLag",
    "Logarithm",
    "LookupTable1d",
    "LookupTable2d",
    "LookupTableND",
    "Prelookup",
    "PrelookupInverse",
    "InterpolationUsingPrelookup",
    "LowPassDiscrete",
    "Chirp",
    "Pulse",
    "PRBS",
    "PRBSLFSR",
    "Quantizer",
    "RandomNumber",
    "RandomSource",
    "Relay",
    "RigidBody",
    "Sawtooth",
    "ScalarBroadcast",
    "Sindy",
    "SumOfElements",
    "Slice",
    "StateMachine",
    "Stack",
    "Step",
    "Stop",
    "Switch",
    "SquareRoot",
    "Ramp",
    "RateLimiter",
    "RateTransition",
    "Decimator",
    "Saturate",
    "SoftRateLimiter",
    "SoftSaturate",
    "soft_dead_zone",
    "soft_saturate",
    "PIDController2DOF",
    "PIDDiscrete",
    "WhiteNoise",
    "ZeroOrderHold",
    "UnitDelay",
    "TableSearch",
    "TransportDelay",
    "VariableTransportDelay",
    "ode_block",
    "feedthrough_block",
    "LTISystem",
    "LTISystemDiscrete",
    "TransferFunction",
    "TransferFunctionDiscrete",
    "linearize",
    "linearize_to_lti",
    "LinearizedSystem",
    "FrequencyResponse",
    "OperatingPoint",
    "bode_data",
    "estimate_frequency_response",
    "findop",
    "frequency_response",
    "impulse_response",
    "nyquist_data",
    "discretize",
    "with_observer",
    "Luenberger",
    "pole_zero_map",
    "step_response",
    "LinearDiscreteTimeMPC",
    "LinearDiscreteTimeMPC_OSQP",
    "DirectShootingNMPC",
    "DirectTranscriptionNMPC",
    "HermiteSimpsonNMPC",
    "MJX",
    "MuJoCo",
    "MLP",
    "QuadraticCost",
    "Trigonometric",
    "TruthTable",
    "TruthTableBuilder",
    "UniformRandomNumber",
    "ReferenceSubdiagram",
    "KalmanFilter",
    "InfiniteHorizonKalmanFilter",
    "ContinuousTimeInfiniteHorizonKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "RecursiveLeastSquares",
    "AugmentedStateEKF",
    "LinearQuadraticRegulator",
    "DiscreteTimeLinearQuadraticRegulator",
    "FiniteHorizonLinearQuadraticRegulator",
    "LinearQuadraticGaussian",
    "PyTwin",
    "PyTorch",
    "TensorFlow",
    "PyTorchPredictor",
    "TensorFlowPredictor",
    "ONNX",
    "ONNXJax",
    "model_description_xml",
    "write_model_description",
    "VideoSink",
    "VideoSource",
    "Ros2Publisher",
    "Ros2Subscriber",
    "SignalDatatypeConversion",
    "QubeServoModel",
    "QuanserHAL",
    "ShiftRegister",
    "MaskedDelayBuffer",
    "fit_lookup_table_1d",
    "fit_lookup_table_2d",
    "fit_lookup_table_nd",
    "fit_table_1d_with_grid",
    "fit_table_2d",
    "fit_table_nd",
    # Reduced-order modeling & statistical surrogates (jaxonomy.library.rom),
    # listed unconditionally like the other IS_JAXLITE-guarded blocks
    # (MPC, LQR, Sindy, MLP) above.
    "ReducedOrderModel",
    "reduce",
    "SnapshotData",
    "collect_snapshots",
    "relative_error",
    "retained_energy",
    "projection_error",
    "controllability_gramian",
    "observability_gramian",
    "hankel_singular_values",
    "balanced_realization",
    "balanced_truncation",
    "balred",
    "minimal_realization",
    "minreal",
    "modal_truncation",
    "residualize",
    "pod_basis",
    "galerkin_reduce",
    "deim",
    "deim_galerkin_reduce",
    "DMDResult",
    "DMDcResult",
    "ERAResult",
    "dmd",
    "dmdc",
    "era",
    "DMDForecaster",
    "identity_dictionary",
    "polynomial_dictionary",
    "rbf_dictionary",
    "EDMDResult",
    "edmd",
    "KoopmanPredictor",
    "GPModel",
    "fit_gp",
    "GaussianProcess",
    "PCEModel",
    "fit_pce",
    "PolynomialChaos",
    "RBFModel",
    "fit_rbf",
    "RadialBasisSurrogate",
]
