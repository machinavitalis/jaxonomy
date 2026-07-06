# SPDX-License-Identifier: MIT

"""Provenance / reproducibility manifest (T-110 Phase 1).

Captures a snapshot of the environment, library versions, simulator
options, and a deterministic system fingerprint at the time
:func:`jaxonomy.simulate` is called.  The manifest is meant to be
attached to :class:`~jaxonomy.simulation.types.SimulationResults` so
that a recorded simulation can later be reproduced or its conditions
audited.

Default-off path is byte-equivalent: the manifest is gathered only when
``SimulatorOptions.record_provenance=True``.  Computation runs entirely
in Python before/after the JIT-traced ``_wrapped_simulate`` kernel —
nothing is captured inside the JAX trace.

Public surface:

* :class:`ProvenanceManifest` — frozen dataclass with ``to_dict()`` and
  ``to_json()`` serialisers.
* :func:`compute_provenance` — gather all fields from a system + options
  pair.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import subprocess
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..framework import SystemBase
    from .types import SimulatorOptions


__all__ = [
    "ProvenanceManifest",
    "ResultsWithProvenance",
    "bundle_results",
    "canonicalize_for_hash",
    "compute_provenance",
    "gather_git_info",
]


# ---------------------------------------------------------------------
# Canonical PyTree serialiser (T-110-followup-pytree-stable-hash)
# ---------------------------------------------------------------------
#
# ``json.dumps(..., sort_keys=True, default=repr)`` already gave us a
# Python-version-stable hash for primitive option fields, but:
#
# * ``repr(np.ndarray)`` truncates long arrays with ellipses and is not
#   guaranteed byte-stable across numpy releases.
# * ``repr(jax.Array)`` includes a device tag that flips between hosts
#   (``cuda:0`` vs ``cpu``) and across JAX versions.
# * Dataclass / NamedTuple ``repr`` includes the declaration order of
#   fields, so a cosmetic field re-order in a future Jaxonomy version
#   would silently break the hash even when semantics are unchanged.
#
# :func:`canonicalize_for_hash` walks a value and returns a structure
# composed only of ``(None, bool, int, float, str, list, dict)``.  By
# the time ``json.dumps(..., sort_keys=True)`` sees the structure,
# every dict key has been stringified and every container is sorted
# (for sets/frozensets) or ordered by declaration (for sequences),
# so the byte output is identical on Python 3.10 / 3.11 / 3.12 for
# the same input semantics.
#
# Canonical form per type (documented for external verifiers):
#
# * ``None / bool / int / str`` → unchanged.
# * ``float``                   → ``repr(float)`` (handles ``nan`` /
#                                 ``inf`` deterministically, since
#                                 ``json`` rejects them by default).
# * ``bytes / bytearray``       → ``["__bytes__", hex_digest]``.
# * ``list / tuple``            → ``["__seq__", [canon(x), ...]]``.
# * ``set / frozenset``         → ``["__set__", sorted_list_of_canon]``.
# * ``dict``                    → ``["__map__", sorted_kv_pairs]``
#                                 (key coerced to ``str`` first).
# * ``np.ndarray``              → ``["__ndarray__", shape, dtype_str,
#                                    sha256_of_tobytes]``.
# * jax ``Array``               → ``["__jax_array__", shape, dtype_str,
#                                    sha256_of_tobytes]``.
# * PRNG key array              → ``["__jax_prng_key__", shape,
#                                    dtype_str, sha256_of_tobytes]``
#                                 (detected by ``uint32`` dtype + the
#                                  conventional ``(2,)`` shape).
# * np / jax scalar dtype       → ``["__dtype__", str(dtype)]``.
# * Dataclass instance          → ``["__dataclass__", qualname,
#                                    sorted_field_dict]``.
# * NamedTuple instance         → ``["__namedtuple__", qualname,
#                                    sorted_field_dict]``.
# * Anything else               → ``["__repr__", repr(value)]``
#                                 (honest fallback for closures,
#                                 lambdas, JAX tracers, etc.).
#
# The fixed wrapper tags ensure two different value categories never
# collide just because they happen to canonicalise to similar nested
# shapes.

_HASH_TAG_BYTES = "__bytes__"
_HASH_TAG_SEQ = "__seq__"
_HASH_TAG_SET = "__set__"
_HASH_TAG_MAP = "__map__"
_HASH_TAG_NDARRAY = "__ndarray__"
_HASH_TAG_JAX_ARRAY = "__jax_array__"
_HASH_TAG_JAX_PRNG_KEY = "__jax_prng_key__"
_HASH_TAG_DTYPE = "__dtype__"
_HASH_TAG_DATACLASS = "__dataclass__"
_HASH_TAG_NAMEDTUPLE = "__namedtuple__"
_HASH_TAG_REPR = "__repr__"


def _is_namedtuple_instance(value: Any) -> bool:
    """Return ``True`` if ``value`` quacks like a ``typing.NamedTuple``.

    Duck-typing avoids importing ``typing.NamedTuple`` at module load
    time and stays compatible with both ``collections.namedtuple`` and
    ``typing.NamedTuple`` flavours.
    """
    return (
        isinstance(value, tuple)
        and hasattr(value, "_fields")
        and hasattr(value, "_asdict")
    )


def _qualname(value: Any) -> str:
    """Return ``module.qualname`` for a class instance, used as a stable
    identifier in the canonical form for dataclasses / NamedTuples."""
    cls = type(value)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else str(qualname)


def _array_digest(arr: Any) -> tuple[list[int], str, str]:
    """Return ``(shape_list, dtype_str, sha256_hex_of_bytes)`` for a
    numpy/JAX-like array.

    Falls back gracefully when ``tobytes`` is unavailable (extremely
    rare — every real ndarray-shaped value we care about has it).
    """
    try:
        import numpy as _np
        # ``np.asarray`` materialises JAX arrays without copying when
        # they live on the host, and forces a host copy otherwise — both
        # outcomes give us a deterministic byte stream.
        host = _np.asarray(arr)
        shape = [int(s) for s in host.shape]
        dtype_str = str(host.dtype)
        digest = hashlib.sha256(host.tobytes()).hexdigest()
        return shape, dtype_str, digest
    except Exception as exc:  # pragma: no cover - defensive
        return [], f"<error: {exc!r}>", ""


def canonicalize_for_hash(value: Any) -> Any:
    """Return a JSON-friendly canonical form of ``value`` for hashing.

    The output is composed exclusively of ``None``, ``bool``, ``int``,
    ``float``, ``str``, ``list``, and ``dict``, so ``json.dumps(
    canonicalize_for_hash(v), sort_keys=True)`` is byte-stable across
    Python 3.10 / 3.11 / 3.12 for any input whose runtime types are in
    the documented set above.  See the module-level comment for the
    full type → canonical-form mapping.
    """
    # --- primitives -------------------------------------------------
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        # ``repr(float)`` is byte-stable since 3.1 (PEP 3101 short repr)
        # and survives nan/inf, which raw json.dumps rejects.
        return ["__float__", repr(float(value))]
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return [_HASH_TAG_BYTES, hashlib.sha256(bytes(value)).hexdigest()]

    # --- numpy / jax arrays ----------------------------------------
    # We check arrays before generic sequences because ndarrays satisfy
    # ``__iter__`` but should not be treated as plain lists.
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            shape, dtype_str, digest = _array_digest(value)
            return [_HASH_TAG_NDARRAY, shape, dtype_str, digest]
        if isinstance(value, _np.generic):
            # 0-d numpy scalar — canonicalise its Python value + dtype.
            return [
                "__np_scalar__",
                str(value.dtype),
                canonicalize_for_hash(value.item()),
            ]
        if isinstance(value, _np.dtype):
            return [_HASH_TAG_DTYPE, str(value)]
    except Exception:  # pragma: no cover - numpy import failure shouldn't break the hash
        pass

    try:
        import jax
        if isinstance(value, jax.Array):
            shape, dtype_str, digest = _array_digest(value)
            # PRNG keys conventionally have ``uint32`` dtype + shape (2,)
            # in the legacy key format; we tag them separately so a
            # plain ``jnp.array([0, 0], dtype=uint32)`` doesn't collide
            # with an actual PRNG key (which it semantically is anyway
            # under that representation — the tag is purely for human
            # readability of the canonical form).
            tag = (
                _HASH_TAG_JAX_PRNG_KEY
                if dtype_str == "uint32" and shape == [2]
                else _HASH_TAG_JAX_ARRAY
            )
            return [tag, shape, dtype_str, digest]
    except Exception:  # pragma: no cover - jax import failure shouldn't break the hash
        pass

    # --- dataclasses / NamedTuples ---------------------------------
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # ``sorted(...)`` on field names is what makes the hash
        # robust to dataclass-field re-orderings across jaxonomy
        # minor versions: the declaration order changes but the
        # canonical form does not.
        fields = sorted(dataclasses.fields(value), key=lambda f: f.name)
        return [
            _HASH_TAG_DATACLASS,
            _qualname(value),
            [[f.name, canonicalize_for_hash(getattr(value, f.name))]
             for f in fields],
        ]
    if _is_namedtuple_instance(value):
        # ``_asdict`` is order-preserving in Py3.8+, but we sort
        # anyway so field re-orderings don't perturb the hash.
        items = sorted(value._asdict().items(), key=lambda kv: kv[0])
        return [
            _HASH_TAG_NAMEDTUPLE,
            _qualname(value),
            [[k, canonicalize_for_hash(v)] for k, v in items],
        ]

    # --- sequences / mappings / sets -------------------------------
    if isinstance(value, (list, tuple)):
        return [_HASH_TAG_SEQ, [canonicalize_for_hash(v) for v in value]]
    if isinstance(value, (set, frozenset)):
        # Sets have no insertion order; sort by their canonical JSON
        # form so the hash is invariant to construction order.
        items = [canonicalize_for_hash(v) for v in value]
        items_sorted = sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
        return [_HASH_TAG_SET, items_sorted]
    if isinstance(value, dict):
        # Sort by stringified key — JSON has no notion of non-string
        # keys, and ``sort_keys=True`` on the final ``json.dumps`` only
        # sorts the *top* level of each dict; we want recursive sort
        # for the deterministic-byte guarantee.
        items = sorted(
            ((str(k), canonicalize_for_hash(v)) for k, v in value.items()),
            key=lambda kv: kv[0],
        )
        return [_HASH_TAG_MAP, [[k, v] for k, v in items]]

    # --- honest fallback -------------------------------------------
    # Closures, lambdas, JAX tracers, opaque C extensions: ``repr`` is
    # the best we can do and is documented as such.  This is the only
    # path that's *not* fully byte-stable across Python versions.
    return [_HASH_TAG_REPR, repr(value)]


# Fields of ``SimulatorOptions`` worth recording for reproducibility.
# Excluded: callables (``major_step_callback``), recorded-signal dicts
# (port objects are not JSON-friendly and are recorded separately on
# ``SimulationResults``), and internal underscore-prefixed bookkeeping
# fields (``_explicit_max_major_steps``).
_RECORDED_OPTION_FIELDS: tuple[str, ...] = (
    "math_backend",
    "enable_tracing",
    "enable_autodiff",
    "max_major_steps",
    "max_major_step_length",
    "buffer_length",
    "ode_solver_method",
    "rtol",
    "atol",
    "min_minor_step_size",
    "max_minor_step_size",
    "max_checkpoints",
    "save_time_series",
    "return_context",
    "validate",
    "zc_bisection_loop_count",
    "int_time_scale",
    "lower_triangular_discrete_update",
    "check_rate_transitions",
    "dae_projection_enabled",
    "dae_projection_tol",
    "dae_projection_max_iter",
    "dae_drift_threshold",
    "bdf_condition_warning_threshold",
    "zeno_protection_enabled",
    "zeno_tolerance",
    "zeno_recovery_period",
    "per_signal_timestamps",
    "per_signal_timestamps_atol",
    "per_signal_timestamps_mode",
    "record_solver_states",
    "record_provenance",
)


def _json_safe(value: Any) -> Any:
    """Coerce ``value`` to a JSON-serialisable Python type.

    Falls back to ``repr(value)`` for anything that doesn't naturally
    round-trip — keeps the manifest informative without exploding on
    exotic types (callables, JAX tracers, etc.).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_safe(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    # Final fallback — string representation always round-trips through json.
    return repr(value)


def _capture_options(options: Optional["SimulatorOptions"]) -> dict[str, Any]:
    if options is None:
        return {}
    captured: dict[str, Any] = {}
    for name in _RECORDED_OPTION_FIELDS:
        if hasattr(options, name):
            captured[name] = _json_safe(getattr(options, name))
    return captured


def _capture_precision_info() -> dict[str, Any]:
    try:
        from ..precision import precision_info
        info = precision_info()
        return {
            "x64_enabled": bool(info.x64_enabled),
            "default_float_dtype": str(info.default_float_dtype),
            "machine_eps": float(info.machine_eps),
            "integer_time_dtype": str(info.integer_time_dtype),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": repr(exc)}


def _capture_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from ..version import __version__ as jaxonomy_version
        versions["jaxonomy"] = str(jaxonomy_version)
    except Exception as exc:  # pragma: no cover - defensive
        versions["jaxonomy"] = f"<error: {exc!r}>"
    try:
        import jax
        versions["jax"] = str(jax.__version__)
    except Exception as exc:  # pragma: no cover - defensive
        versions["jax"] = f"<error: {exc!r}>"
    try:
        import numpy as _np
        versions["numpy"] = str(_np.__version__)
    except Exception as exc:  # pragma: no cover - defensive
        versions["numpy"] = f"<error: {exc!r}>"
    return versions


_GIT_SUBPROCESS_TIMEOUT_SECONDS = 2.0


def _run_git(args: list[str], cwd: str) -> Optional[str]:
    """Run a git command in ``cwd`` and return stdout (stripped).

    Returns ``None`` if git isn't on PATH, the command fails, or the
    timeout fires.  Centralised so :func:`gather_git_info` can stay
    readable and unit tests have a single seam to mock.
    """
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def gather_git_info(start: Optional[str] = None) -> dict[str, Any]:
    """Best-effort capture of git HEAD metadata for the repo at ``start``.

    Returns a dict with four keys — ``sha`` (short HEAD SHA, e.g.
    ``"abc1234"``), ``branch`` (current branch name, may be
    ``"HEAD"`` in a detached state), ``dirty`` (``True`` when
    ``git status --porcelain`` is non-empty), and ``commit_time``
    (ISO-8601 UTC timestamp of the HEAD commit).  Every field is
    ``None`` when the corresponding git call fails, when ``git`` is
    unavailable, or when the call times out.  Designed as a single
    seam tests can monkeypatch (``provenance.gather_git_info``).
    """
    cwd = start if start is not None else os.getcwd()
    info: dict[str, Any] = {
        "sha": None,
        "branch": None,
        "dirty": None,
        "commit_time": None,
    }
    sha_raw = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    if sha_raw is None:
        # Not a git checkout / git unavailable — everything stays None.
        return info
    sha = sha_raw.strip()
    info["sha"] = sha or None

    branch_raw = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch_raw is not None:
        branch = branch_raw.strip()
        info["branch"] = branch or None

    status_raw = _run_git(["status", "--porcelain"], cwd)
    if status_raw is not None:
        # Non-empty porcelain output means uncommitted changes.
        info["dirty"] = bool(status_raw.strip())

    time_raw = _run_git(["log", "-1", "--format=%ct", "HEAD"], cwd)
    if time_raw is not None:
        time_str = time_raw.strip()
        if time_str:
            try:
                ts = int(time_str)
                info["commit_time"] = (
                    datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    .isoformat()
                )
            except (ValueError, OSError, OverflowError):
                info["commit_time"] = None
    return info


def _capture_git_head(start: Optional[str] = None) -> Optional[str]:
    """Return the short git HEAD SHA for the repo containing ``start``.

    Kept as a thin compatibility wrapper around :func:`gather_git_info`;
    older code (and the original Phase 1 manifest ``git_head`` field)
    only wanted the SHA.  Returns ``None`` when not inside a git
    checkout, when ``git`` is unavailable, or when the call times out.
    """
    return gather_git_info(start).get("sha")


def _structural_fingerprint(system: "SystemBase") -> tuple:
    """Recursive structural fingerprint of ``system``.

    Returns an immutable, JSON-friendly tuple that captures the
    structural shape of ``system`` *without* dependence on the
    per-process ``system_id`` counter. The key inputs are:

    * ``type(system).__name__`` — the block class name.
    * ``sorted(parameter.keys())`` — the parameter pytree's name set.
    * For diagrams: a tuple of recursive fingerprints over the
      diagram's children, in the order returned by ``system.nodes``
      (registration order, which is stable across rebuilds of the
      same builder script).

    Two structurally-equivalent systems built in different processes
    (or in two ``DiagramBuilder`` invocations within the same process)
    produce equal fingerprints. Used by :func:`_system_fingerprint`
    to derive the ``hash`` field of the provenance manifest.
    """
    # Lazy import to dodge the framework→simulation cycle.
    from ..framework.diagram import Diagram

    type_name = type(system).__name__
    try:
        param_names = tuple(sorted(str(k) for k in system.parameters.keys()))
    except Exception:
        param_names = ()
    if isinstance(system, Diagram):
        try:
            child_fingerprints = tuple(
                _structural_fingerprint(child) for child in system.nodes
            )
        except Exception:
            child_fingerprints = ()
        return (type_name, param_names, child_fingerprints)
    return (type_name, param_names)


def _system_fingerprint(system: Optional["SystemBase"]) -> dict[str, Any]:
    """Deterministic, JAX-tracer-safe fingerprint of ``system``.

    We avoid touching parameter *values* (which may be tracers or
    arrays whose ``__hash__`` is not stable) and instead key off the
    type name + sorted parameter names — recursively for diagrams via
    :func:`_structural_fingerprint`.

    T-110-followup-stable-fingerprint: ``system_id`` (a per-process
    auto-incrementing counter assigned at ``LeafSystem`` construction
    time) used to be folded into the hash, which made ``config_hash``
    differ across two byte-equivalent runs in different Python
    processes — the opposite of what the "notarized receipt" marketing
    claim implies. The hash now derives from
    :func:`_structural_fingerprint` (recursive type + parameter names),
    which is stable across processes and rebuilds while still
    differentiating genuinely-different diagram structures (different
    leaf set / different parameter names / different leaf order).

    ``system_id`` is preserved in the returned dict as
    informational-only — useful for debugging which leaf instance
    produced a given fingerprint within a single process.
    """
    if system is None:
        return {
            "system_id": None,
            "type": None,
            "parameter_names": [],
            "hash": None,
        }
    try:
        param_names = sorted(str(k) for k in system.parameters.keys())
    except Exception:
        param_names = []
    try:
        system_id = system.system_id
        # ``system_id`` may be an int or a UUID; coerce to string for json.
        system_id_repr: Any = system_id if isinstance(system_id, int) else str(system_id)
    except Exception:
        system_id_repr = None
    type_name = type(system).__name__
    structural = _structural_fingerprint(system)
    digest = hashlib.sha256(repr(structural).encode("utf-8")).hexdigest()
    return {
        # Informational only — NOT folded into ``hash``. Useful for
        # diagnosing which leaf instance produced a given fingerprint
        # within a single process.
        "system_id": system_id_repr,
        "type": type_name,
        "parameter_names": param_names,
        "hash": digest,
    }


def _compute_config_hash(
    *,
    options: dict[str, Any],
    system: dict[str, Any],
    jaxonomy_version: str,
    jax_version: str,
) -> str:
    """Deterministic SHA-256 over the configuration that defines a run.

    The hash is a stable "run identity": two runs that share a
    ``config_hash`` should produce bit-equivalent numerical results
    (modulo non-determinism that ``SimulatorOptions`` explicitly
    accepts, e.g. asynchronous device dispatch order).

    Included inputs:

    * ``options`` — the :class:`SimulatorOptions` snapshot returned by
      :func:`_capture_options`.  Keys are sorted before hashing so dict
      ordering can't leak into the digest.
    * ``system`` — the system fingerprint returned by
      :func:`_system_fingerprint`. As of T-110-followup-stable-fingerprint
      this fingerprint hashes only ``(type_name, sorted parameter names)``
      so the resulting ``config_hash`` is stable across processes — two
      byte-equivalent runs in different Python processes share the same
      hash, which is the headline T-110 reproducibility contract.
    * ``jaxonomy_version`` and ``jax_version`` — library identity.

    Deliberately excluded:

    * ``timestamp`` — varies on every run.
    * Git HEAD / branch / dirty — tests should reproduce across commits.
    * ``numpy_version`` — orthogonal to the simulator/JAX kernel.

    Byte-stability note (T-110-followup-pytree-stable-hash): all
    inputs are routed through :func:`canonicalize_for_hash`, which
    rewrites numpy/JAX arrays, dataclasses, NamedTuples, sets, and
    dicts into a JSON-friendly form whose ``json.dumps(..., sort_keys
    =True)`` output is byte-identical on Python 3.10 / 3.11 / 3.12
    for the same input semantics.  For exotic values that fall
    outside the documented "POD-shaped" set (closures, lambdas,
    live JAX tracers, opaque C extensions), the canonicaliser falls
    back to ``repr(value)`` and the resulting hash is therefore only
    best-effort byte-stable across interpreter releases.
    """
    # Route everything through ``canonicalize_for_hash`` so the
    # serialised bytes don't depend on dict iteration order, dataclass
    # field declaration order, numpy/JAX array repr formatting, or
    # any other Python-version-sensitive ``repr`` output.
    #
    # T-110-followup-stable-fingerprint: strip ``system_id`` from the
    # system dict before hashing. The fingerprint's ``hash`` field
    # already captures the recursive structural identity in a
    # cross-process-stable way; ``system_id`` itself is informational
    # only (per-process auto-increment counter) and must not leak into
    # the digest.
    system_for_hash = {k: v for k, v in (system or {}).items() if k != "system_id"}
    payload = canonicalize_for_hash({
        "jaxonomy_version": jaxonomy_version,
        "jax_version": jax_version,
        "options": options,
        "system": system_for_hash,
    })
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True)
class ProvenanceManifest:
    """Reproducibility snapshot for one ``simulate(...)`` call (T-110).

    Phase 1 captures library versions, the resolved precision policy, a
    deterministic system fingerprint, and the relevant
    :class:`SimulatorOptions` field values.  An ISO-8601 UTC timestamp
    is included so the manifest is self-describing; ``git_head`` is
    populated when ``simulate`` is called from inside a git checkout.

    The ``config_hash`` field (T-110-followup-config-hash) is a
    deterministic SHA-256 of the relevant configuration — same options
    + same system + same jaxonomy/jax versions yield the same hash
    across runs and across git commits (timestamp and git HEAD are
    deliberately excluded).

    The dataclass is frozen so a recorded manifest can't be silently
    mutated downstream.
    """

    jaxonomy_version: str
    jax_version: str
    numpy_version: str
    precision_info: dict[str, Any]
    options: dict[str, Any]
    system: dict[str, Any]
    timestamp: str
    git_head: Optional[str] = None
    # T-110-followup-git-revision: richer git metadata.
    git_head_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_head_commit_time: Optional[str] = None
    # T-110-followup-config-hash: deterministic run-identity hash.
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict representation of the manifest."""
        return {
            "jaxonomy_version": self.jaxonomy_version,
            "jax_version": self.jax_version,
            "numpy_version": self.numpy_version,
            "precision_info": dict(self.precision_info),
            "options": dict(self.options),
            "system": dict(self.system),
            "timestamp": self.timestamp,
            "git_head": self.git_head,
            "git_head_sha": self.git_head_sha,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "git_head_commit_time": self.git_head_commit_time,
            "config_hash": self.config_hash,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialise :meth:`to_dict` via ``json.dumps``."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=repr)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceManifest":
        """Construct a :class:`ProvenanceManifest` from a dict produced by
        :meth:`to_dict` (round-trip helper for serialisation tests)."""
        git_dirty = data.get("git_dirty")
        if git_dirty is not None:
            git_dirty = bool(git_dirty)
        return cls(
            jaxonomy_version=str(data.get("jaxonomy_version", "")),
            jax_version=str(data.get("jax_version", "")),
            numpy_version=str(data.get("numpy_version", "")),
            precision_info=dict(data.get("precision_info", {}) or {}),
            options=dict(data.get("options", {}) or {}),
            system=dict(data.get("system", {}) or {}),
            timestamp=str(data.get("timestamp", "")),
            git_head=data.get("git_head"),
            git_head_sha=data.get("git_head_sha"),
            git_branch=data.get("git_branch"),
            git_dirty=git_dirty,
            git_head_commit_time=data.get("git_head_commit_time"),
            config_hash=str(data.get("config_hash", "") or ""),
        )


def compute_provenance(
    system: Optional["SystemBase"],
    options: Optional["SimulatorOptions"] = None,
    *,
    include_git: bool = True,
    timestamp: Optional[str] = None,
) -> ProvenanceManifest:
    """Build a :class:`ProvenanceManifest` for ``system`` + ``options``.

    All capture happens in plain Python — no JAX tracing — so the
    function is safe to call before or after a JIT'd simulation kernel.

    Args:
        system: the system being simulated (may be ``None`` for tests
            or pre-built recordings).
        options: the active :class:`SimulatorOptions`; ``None`` records
            an empty options dict.
        include_git: when ``False``, skip the git-HEAD lookup (useful
            when the caller knows it isn't in a git checkout or wants a
            faster path).
        timestamp: optional override (ISO-8601 string).  Defaults to the
            current UTC time.  Override is useful for deterministic
            tests.

    Returns:
        A populated :class:`ProvenanceManifest`.
    """
    versions = _capture_versions()
    precision = _capture_precision_info()
    options_dict = _capture_options(options)
    sys_fp = _system_fingerprint(system)
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # T-110-followup-git-revision: gather richer git metadata in a single
    # mockable call.  When ``include_git=False`` the entire git capture
    # is skipped (manifest stays None across all git_* fields).
    if include_git:
        git_info = gather_git_info()
    else:
        git_info = {"sha": None, "branch": None, "dirty": None, "commit_time": None}
    # T-110-followup-config-hash: deterministic run-identity hash —
    # excludes timestamp and git metadata so the same config produces
    # the same hash across commits and re-runs.
    config_hash = _compute_config_hash(
        options=options_dict,
        system=sys_fp,
        jaxonomy_version=versions.get("jaxonomy", ""),
        jax_version=versions.get("jax", ""),
    )
    return ProvenanceManifest(
        jaxonomy_version=versions.get("jaxonomy", ""),
        jax_version=versions.get("jax", ""),
        numpy_version=versions.get("numpy", ""),
        precision_info=precision,
        options=options_dict,
        system=sys_fp,
        timestamp=timestamp,
        git_head=git_info.get("sha"),
        git_head_sha=git_info.get("sha"),
        git_branch=git_info.get("branch"),
        git_dirty=git_info.get("dirty"),
        git_head_commit_time=git_info.get("commit_time"),
        config_hash=config_hash,
    )


# ---------------------------------------------------------------------
# T-110-followup-results-bundle: ``(results, provenance)`` wrapper
# ---------------------------------------------------------------------
#
# Today provenance is exposed as a *field* on the results object
# (``results.provenance``) — consistent across simulate / simulate_batch
# / simulate_jacfwd, but it mixes "what was simulated" with "how it was
# produced".  ``ResultsWithProvenance`` is a thin, opt-in wrapper that
# keeps the two concerns side by side without hiding either one:
#
#   bundled = bundle_results(results)
#   bundled.results.outputs[...]          # underlying results
#   bundled.provenance.config_hash        # paired manifest
#   bundled.outputs[...]                  # forwarded for ergonomics
#
# The legacy ``results.provenance`` attribute is preserved unchanged —
# this is purely an additive ergonomic helper.  ``bundle_results``
# returns the original ``results`` untouched when no provenance is
# attached, so callers can adopt the helper without conditionals.


@dataclasses.dataclass(frozen=True)
class ResultsWithProvenance:
    """Pair a results object with its :class:`ProvenanceManifest`.

    Attribute access is forwarded to the underlying ``results``
    instance, so ``wrapped.outputs[name]`` works exactly like
    ``results.outputs[name]``.  ``wrapped.results`` and
    ``wrapped.provenance`` give explicit access to either side.

    The wrapper is frozen so the pairing can't be silently mutated.
    Construction does not copy or wrap the underlying results — the
    wrapper holds a reference, nothing else.
    """

    results: Any
    provenance: Any

    # ``__getattr__`` is only invoked when normal attribute lookup
    # fails, so ``self.results`` / ``self.provenance`` always resolve
    # against the dataclass fields (no infinite recursion).
    def __getattr__(self, name: str) -> Any:
        # ``object.__getattribute__`` reaches the dataclass slot
        # directly and raises ``AttributeError`` if (somehow) the field
        # isn't yet set — which is the right signal for ``hasattr``.
        results = object.__getattribute__(self, "results")
        try:
            return getattr(results, name)
        except AttributeError as exc:
            # Re-raise with the wrapper's own type in the message so
            # the user sees that the lookup went through us.
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r} "
                f"(neither does the wrapped {type(results).__name__})"
            ) from exc

    def __repr__(self) -> str:
        # Show both sides explicitly so the wrapper is self-describing
        # even when the underlying results' ``repr`` is verbose.
        return (
            f"ResultsWithProvenance(results={self.results!r}, "
            f"provenance={self.provenance!r})"
        )


def bundle_results(results: Any) -> Any:
    """Wrap ``results`` in a :class:`ResultsWithProvenance` if applicable.

    When ``results.provenance`` is populated (a non-None manifest), the
    return value is a :class:`ResultsWithProvenance` carrying both the
    original results object and its provenance.  When ``results`` has
    no ``provenance`` attribute or that attribute is ``None``, the
    original ``results`` object is returned unchanged — so callers can
    sprinkle ``bundle_results(...)`` in front of every simulate call
    without breaking byte-equivalent default-off paths.

    This helper is purely ergonomic.  The legacy
    ``results.provenance`` field is left in place; nothing about the
    underlying results object is mutated.
    """
    provenance = getattr(results, "provenance", None)
    if provenance is None:
        return results
    return ResultsWithProvenance(results=results, provenance=provenance)


# ---------------------------------------------------------------------
# T-110 phase 2: persistence + verification helpers
# ---------------------------------------------------------------------
#
# Phase 1 shipped the manifest dataclass and its to_dict / to_json /
# from_dict round-trip.  Phase 2 adds the two pieces a reproducibility
# workflow actually needs in practice:
#
#   * ``manifest.save(path)`` / ``load_manifest(path)`` — pin a
#     manifest to disk so a release tag, a CI artifact, or a notebook
#     can ship "this is the run we promised."
#   * ``compare_manifests(actual, expected)`` /
#     ``verify_manifest(actual, expected)`` — diff two manifests and
#     surface every field that drifted.  ``verify_*`` raises the
#     ``ManifestMismatch`` exception on any drift so it composes
#     naturally with ``pytest`` and assertion-style CI checks.
#
# Default ``ignore_fields={"timestamp"}``: every recompute populates a
# fresh timestamp and that field is never load-bearing for
# reproducibility.  Callers comparing across-machine or across-git can
# extend the ignore set (``ignore_fields={"timestamp", "git_head",
# "git_head_sha", "git_branch", "git_dirty", "git_head_commit_time"}``);
# callers who *want* timestamp drift to surface can pass
# ``ignore_fields=set()``.
#
# The CI piece of phase 2 (a workflow that bumps a published manifest
# per release tag for a small reference corpus) is deferred as
# ``T-110-followup-ci-manifest-bump`` — it depends on the user's CI
# convention and corpus selection rather than on the code surfaced here.


class ManifestMismatch(AssertionError):
    """Raised by :func:`verify_manifest` when two manifests differ.

    Inherits from :class:`AssertionError` so it composes with
    ``pytest`` and standard assertion-style verification flows
    without callers needing to import the exception explicitly.

    The exception instance carries a ``differences`` attribute holding
    the same ``list[tuple[str, Any, Any]]`` that
    :func:`compare_manifests` returns, so programmatic consumers can
    introspect the drift instead of parsing the message.
    """

    def __init__(self, differences: list[tuple[str, Any, Any]]):
        self.differences: list[tuple[str, Any, Any]] = list(differences)
        lines = [f"{len(self.differences)} manifest field(s) drifted:"]
        for path, actual, expected in self.differences:
            lines.append(f"  {path}: actual={actual!r} expected={expected!r}")
        super().__init__("\n".join(lines))


_DEFAULT_IGNORE_FIELDS = frozenset({"timestamp"})


def _save_manifest_impl(manifest: "ProvenanceManifest", path, indent: int) -> None:
    """Concrete persistence path used by :meth:`ProvenanceManifest.save`."""
    import pathlib

    target = pathlib.Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json(indent=indent))


# Attach `save` to the dataclass after class creation so the helper has
# access to `_save_manifest_impl` defined above.  Equivalent to writing
# `def save(self, path, *, indent=2): ...` inside the class body — kept
# out-of-line so the dataclass declaration stays focused on field
# definitions.
def _ProvenanceManifest_save(
    self: "ProvenanceManifest",
    path,
    *,
    indent: Optional[int] = 2,
) -> None:
    """Persist this manifest as JSON at ``path``.

    Parent directories are created on demand.  Pretty-printed by
    default (``indent=2``) so the file is human-readable; pass
    ``indent=None`` for a compact one-line form when diffing-friendly
    JSON isn't needed.

    Args:
        path: filesystem path (str or pathlib.Path).
        indent: JSON indent level, or None for the compact form.
    """
    _save_manifest_impl(self, path, indent=indent)


ProvenanceManifest.save = _ProvenanceManifest_save  # type: ignore[attr-defined]


def load_manifest(path) -> ProvenanceManifest:
    """Load a :class:`ProvenanceManifest` from a JSON file written by
    :meth:`ProvenanceManifest.save`.

    Args:
        path: filesystem path (str or pathlib.Path) of the saved manifest.

    Returns:
        The reconstructed :class:`ProvenanceManifest`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    import pathlib

    data = json.loads(pathlib.Path(path).read_text())
    return ProvenanceManifest.from_dict(data)


def _diff_value(path: str, actual: Any, expected: Any) -> list[tuple[str, Any, Any]]:
    """Recursive deep-diff for the dict / list / scalar payload shapes
    that show up inside a manifest.  Returns a flat list of
    ``(dotted_path, actual_subtree, expected_subtree)`` triples for
    every leaf that differs.
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        diffs: list[tuple[str, Any, Any]] = []
        all_keys = set(actual) | set(expected)
        for key in sorted(all_keys, key=str):
            sub_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                diffs.append((sub_path, "<missing>", expected[key]))
            elif key not in expected:
                diffs.append((sub_path, actual[key], "<missing>"))
            else:
                diffs.extend(_diff_value(sub_path, actual[key], expected[key]))
        return diffs
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [(f"{path}.length", len(actual), len(expected))]
        diffs = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            diffs.extend(_diff_value(f"{path}[{i}]", a, e))
        return diffs
    if actual != expected:
        return [(path, actual, expected)]
    return []


def compare_manifests(
    actual: ProvenanceManifest,
    expected: ProvenanceManifest,
    *,
    ignore_fields: Optional[set[str]] = None,
) -> list[tuple[str, Any, Any]]:
    """Diff two manifests field-by-field.

    Args:
        actual: the manifest produced by the run being checked.
        expected: the reference manifest (e.g. loaded from a published
            release-tag artifact via :func:`load_manifest`).
        ignore_fields: top-level field names whose drift is acceptable.
            Defaults to ``{"timestamp"}`` since the timestamp is always
            different and never load-bearing for reproducibility.  Pass
            ``ignore_fields=set()`` to compare every field including
            the timestamp.

    Returns:
        A flat list of ``(dotted_path, actual_value, expected_value)``
        triples — one per differing leaf.  An empty list means the two
        manifests agree on every compared field.
    """
    if ignore_fields is None:
        ignore_fields = set(_DEFAULT_IGNORE_FIELDS)

    actual_dict = actual.to_dict()
    expected_dict = expected.to_dict()
    for field in ignore_fields:
        actual_dict.pop(field, None)
        expected_dict.pop(field, None)
    return _diff_value("", actual_dict, expected_dict)


def verify_manifest(
    actual: ProvenanceManifest,
    expected: ProvenanceManifest,
    *,
    ignore_fields: Optional[set[str]] = None,
) -> None:
    """Assert that ``actual`` matches ``expected`` field-by-field.

    Convenience wrapper around :func:`compare_manifests` that raises
    :class:`ManifestMismatch` (an :class:`AssertionError` subclass) if
    any field drifted.  The exception message lists every differing
    field on its own line; the ``.differences`` attribute carries the
    same data structurally for programmatic introspection.

    Composes naturally with ``pytest`` (``ManifestMismatch`` is an
    ``AssertionError``, so test runners will treat it like any other
    assertion failure).

    Args:
        actual: the manifest produced by the run being checked.
        expected: the reference manifest.
        ignore_fields: see :func:`compare_manifests`; default
            ``{"timestamp"}``.

    Raises:
        ManifestMismatch: if any compared field differs.
    """
    differences = compare_manifests(actual, expected, ignore_fields=ignore_fields)
    if differences:
        raise ManifestMismatch(differences)
