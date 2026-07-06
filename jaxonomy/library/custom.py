# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import builtins
from collections import namedtuple
from contextlib import redirect_stderr, redirect_stdout
import functools
from io import StringIO
import keyword
import logging
import traceback
import types
from typing import TYPE_CHECKING, Any, List, Mapping

import jax
import jax.numpy as jnp

from ..framework import LeafSystem, LeafState, parameters as declare_parameters
from ..logging import logdata, logger
from ..framework.error import (
    BlockInitializationError,
    BlockParameterError,
    ErrorCollector,
    PythonScriptError,
    PythonScriptTimeNotSupportedError,
)
from ..backend import io_callback, jit, numpy_api as npa

if TYPE_CHECKING:
    from ..backend.typing import Array, DTypeLike
    from ..framework.context import ContextBase


__all__ = [
    "CustomJaxBlock",
    "CustomPythonBlock",
    "_PerBlockModuleProxy",
    "_save_module_state",
    "_restore_module_state",
]


# ---------------------------------------------------------------------------
# Per-block module isolation helpers
# ---------------------------------------------------------------------------

class _PerBlockModuleProxy:
    """Per-instance module proxy for :class:`CustomPythonBlock` environment isolation.

    Each ``CustomPythonBlock`` instance that imports a module gets its own
    proxy.  Attribute *reads* fall through to the real module; attribute
    *writes* (e.g. ``np.seterr(divide='ignore')``) are captured in a per-block
    override dict and do **not** mutate the shared module in ``sys.modules``.

    For functions like ``numpy.seterr`` that apply side effects internally
    (modifying C-level global state), the proxy also wraps known stateful
    functions with a checkpoint/restore mechanism, ensuring that one block's
    ``seterr`` call does not contaminate another block's numeric error policy.

    Supported isolation guarantees
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **numpy error state** (``numpy.seterr`` / ``numpy.geterr``):
      each block maintains its own error policy, applied atomically around
      each ``exec_init`` / ``exec_step`` call.
    * **numpy random state** (``numpy.random.seed``, ``numpy.random.set_state``):
      block A seeding ``numpy.random`` does not advance block B's random
      stream.
    * **Direct attribute assignment** (``np.MY_CONST = 42``):
      stored in the per-block override dict; invisible to other blocks.

    Limitations
    ~~~~~~~~~~~
    True isolation of every possible side-effecting function (e.g.
    ``sys.stdout``, ``logging`` handlers, C-extension global state other than
    numpy) requires OS-level process isolation, which is not provided here.
    Use :class:`CustomJaxBlock` for production differentiable simulation.
    """

    def __init__(self, module: types.ModuleType) -> None:
        object.__setattr__(self, "_real_module", module)
        object.__setattr__(self, "_overrides", {})

    def __getattr__(self, name: str):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_real_module"), name)

    def __setattr__(self, name: str, value) -> None:
        object.__getattribute__(self, "_overrides")[name] = value

    def __delattr__(self, name: str) -> None:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            del overrides[name]
        else:
            delattr(object.__getattribute__(self, "_real_module"), name)

    def __repr__(self) -> str:
        m = object.__getattribute__(self, "_real_module")
        return f"<_PerBlockModuleProxy wrapping {getattr(m, '__name__', repr(m))}>"

    def __dir__(self):
        m = object.__getattribute__(self, "_real_module")
        overrides = object.__getattribute__(self, "_overrides")
        return sorted(set(dir(m)) | set(overrides))


# ---- known mutable module state snapshots ----

def _save_module_state(env: dict) -> dict:
    """Snapshot mutable global state from modules present in *env*.

    Returns a dict that can be passed to :func:`_restore_module_state`.
    Currently snapshots:

    * ``numpy`` / ``np`` — numeric error policy (``geterr()``) and random
      state (``random.get_state()``)
    * ``random`` — Python built-in random state (``getstate()``)
    """
    snapshot: dict = {}

    def _get_np(env):
        for key in ("np", "numpy"):
            v = env.get(key)
            if v is not None:
                # Unwrap proxy if present
                if isinstance(v, _PerBlockModuleProxy):
                    return object.__getattribute__(v, "_real_module")
                return v
        # Also check sys.modules
        import sys
        return sys.modules.get("numpy")

    np_mod = _get_np(env)
    if np_mod is not None:
        try:
            snapshot["numpy_errstate"] = np_mod.geterr()
        except Exception:
            pass
        try:
            snapshot["numpy_random_state"] = np_mod.random.get_state()
        except Exception:
            pass

    # Python built-in random
    for key in ("random",):
        v = env.get(key)
        if v is not None:
            real = object.__getattribute__(v, "_real_module") if isinstance(v, _PerBlockModuleProxy) else v
            if hasattr(real, "getstate"):
                try:
                    snapshot["random_state"] = real.getstate()
                    snapshot["_random_mod"] = real
                except Exception:
                    pass

    return snapshot


def _restore_module_state(snapshot: dict) -> None:
    """Restore previously-snapshotted mutable module state.

    Applies the values saved by :func:`_save_module_state` back to the real
    module objects.  Silently ignores errors (e.g. if numpy was imported after
    the snapshot was taken).
    """
    import sys

    np_mod = sys.modules.get("numpy")
    if np_mod is not None:
        if "numpy_errstate" in snapshot:
            try:
                np_mod.seterr(**snapshot["numpy_errstate"])
            except Exception:
                pass
        if "numpy_random_state" in snapshot:
            try:
                np_mod.random.set_state(snapshot["numpy_random_state"])
            except Exception:
                pass

    real_random = snapshot.get("_random_mod")
    if real_random is not None and "random_state" in snapshot:
        try:
            real_random.setstate(snapshot["random_state"])
        except Exception:
            pass


def _caused_by_nameerror(e):
    if e is None:
        return None
    if isinstance(e, NameError):
        return e
    return _caused_by_nameerror(e.__cause__)


def _default_exec(
    code: str | types.CodeType,
    env: dict[str, Any],
    logger_: logging.Logger,
    inputs: dict[str, jax.Array] = None,
    return_vars: list[str] = None,
    return_dtypes: list[DTypeLike] = None,
    system: LeafSystem = None,
    code_name: str = "step",
):
    """
    `env` is a mutable state this is required because the python script block
    keeps the state across simulation steps.
    """

    stdout_buffer = StringIO()
    strerr_buffer = StringIO()
    exception = None

    if inputs is not None:
        env.update(inputs)

    with redirect_stderr(strerr_buffer):
        with redirect_stdout(stdout_buffer):
            try:
                exec(code, env, env)
            except BaseException as e:
                exception = e

    stdout = stdout_buffer.getvalue()
    if stdout:
        stdout = stdout[:-1] if stdout[-1] == "\n" else stdout
        logger_.info(stdout, **logdata(block=system))
    stderr = strerr_buffer.getvalue()
    if stderr:
        stderr = stderr[:-1] if stderr[-1] == "\n" else stderr
        logger_.warning(stderr, **logdata(block=system))

    if exception is not None:
        errbuf = StringIO()
        errbuf.write(f"{type(exception).__name__}: {exception}\n")
        if exception.__traceback__:
            lines = traceback.format_tb(exception.__traceback__)
            # skipping first "line" (that looks like: "file custom.py, exec()...")
            errbuf.write("".join(lines[1:]))
        tb_str = errbuf.getvalue().strip()

        # Always log traceback — at error level for UI, debug for headless
        if system is not None and system.ui_id is not None:
            logger_.error(tb_str, **logdata(block=system))
        else:
            logger_.debug(tb_str, **logdata(block=system) if system else {})

        name_error = _caused_by_nameerror(exception)
        if name_error and name_error.name == "time":
            raise PythonScriptTimeNotSupportedError(system=system) from exception
        raise PythonScriptError(system=system) from exception

    if return_vars is None:
        return

    for var in return_vars:
        if var not in env:
            raise PythonScriptError(
                f"Variable '{var}' not defined in {code_name}.", system=system
            )

    return [
        npa.asarray(env[var], dtype=dtype)
        for var, dtype in zip(return_vars, return_dtypes)
    ]


def _filter_non_traceable(globals_dict):
    """
    since we have to use locals as globals, our locals gets really polluted with all
    kinds of stuff. trying to retain this whole thing results in jax errors. so this
    functions job is to split the global env in two: one for "dynamic" arrays that jax
    can trace and another for "static" data that cannot be traced and will be stored
    as a block attribute (e.g. functions, classes, modules, etc).
    """

    KNOWN_NON_TRACEABLE = ["__builtins__", "__main__"]
    dynamic_globals = {}
    static_globals = {}

    for key, value in globals_dict.items():
        if key in KNOWN_NON_TRACEABLE or isinstance(value, types.ModuleType):
            static_globals[key] = value
            continue

        try:
            # Test pytree conversion but don't actually convert.  If the global was
            # declared like `x = [0, 1]` we don't want to convert this to
            # `x = [array(0), array(1)]`.  If the global does need to be converted
            # because its value will be used to initialize an output port, this
            # will be done during output initialization.
            jax.tree_util.tree_map(jnp.asarray, value)

            # Store the original value if the value had a pytree structure.
            dynamic_globals[key] = value
        except TypeError:
            # Feel free to remove below debug log if too noisy
            if not isinstance(value, types.ModuleType) and key not in ["__builtins__"]:
                logger.debug(
                    'Filtering non-traceable global "%s" (%s).', key, type(value)
                )
            # The value is not traceable, so store it as a static block attribute.
            static_globals[key] = value

    return dynamic_globals, static_globals


def _validate_custom_block_io_names(block, inputs, outputs):
    """T-036e: eager validation of CustomJaxBlock / CustomPythonBlock
    `inputs` / `outputs` port-name lists.

    Catches six concrete opaque-error sources before they become cryptic
    downstream failures:

      1. Non-list / non-tuple `inputs` / `outputs`.
      2. Non-string entries.
      3. Non-identifier strings (would shadow a builtin or break `exec`).
      4. Reserved Python keywords (`if`, `def`, ...).
      5. Duplicate entries within the same list.
      6. Name collision between `inputs` and `outputs` (silent shadowing
         in the exec'd `user_statements` namespace).

    Raises ``BlockParameterError`` naming the block, the parameter
    (``inputs`` or ``outputs``), and the offending entry.
    """
    for param_name, names in (("inputs", inputs), ("outputs", outputs)):
        if not isinstance(names, (list, tuple)):
            raise BlockParameterError(
                message=(
                    f"CustomBlock {block.name!r} parameter {param_name!r} must "
                    f"be a list or tuple of names; got {type(names).__name__}."
                ),
                system=block,
                parameter_name=param_name,
            )
        seen: set[str] = set()
        for entry in names:
            if not isinstance(entry, str):
                raise BlockParameterError(
                    message=(
                        f"CustomBlock {block.name!r} {param_name!r} entries "
                        f"must be strings; got {type(entry).__name__}: {entry!r}."
                    ),
                    system=block,
                    parameter_name=param_name,
                )
            if not entry.isidentifier():
                raise BlockParameterError(
                    message=(
                        f"CustomBlock {block.name!r} {param_name!r} entry "
                        f"{entry!r} is not a valid Python identifier; "
                        "rename to letters/underscores not starting with a digit."
                    ),
                    system=block,
                    parameter_name=param_name,
                )
            if keyword.iskeyword(entry):
                raise BlockParameterError(
                    message=(
                        f"CustomBlock {block.name!r} {param_name!r} entry "
                        f"{entry!r} is a reserved Python keyword; "
                        "rename to avoid shadowing language syntax in user_statements."
                    ),
                    system=block,
                    parameter_name=param_name,
                )
            if entry in seen:
                raise BlockParameterError(
                    message=(
                        f"CustomBlock {block.name!r} has duplicate "
                        f"{param_name!r} entry {entry!r}; each "
                        f"{param_name[:-1]} name must be unique."
                    ),
                    system=block,
                    parameter_name=param_name,
                )
            seen.add(entry)
    overlap = set(inputs) & set(outputs)
    if overlap:
        offender = sorted(overlap)[0]
        raise BlockParameterError(
            message=(
                f"CustomBlock {block.name!r} has the name {offender!r} in "
                "both `inputs` and `outputs`; user_statements would silently "
                "shadow the input. Rename one of them."
            ),
            system=block,
            parameter_name="outputs",
        )


# ---------------------------------------------------------------------------
# T-036e: AST-based signature diagnostics for Custom*Block scripts.
# ---------------------------------------------------------------------------
#
# These checks fire at __init__ time when ``strict=True`` is passed to the
# block constructor.  They walk the AST of ``init_script``, ``user_statements``
# and ``finalize_script`` and surface concrete typo / signature errors that
# would otherwise manifest as a cryptic ``NameError`` at first eval.
#
# Concrete cases caught:
#   1. init_script / user_statements reference names that aren't in
#      ``inputs ∪ outputs ∪ {"time", "t"} ∪ recognised builtins / imports``.
#   2. ``t`` vs ``time`` confusion — surfaces a fix-up hint.
#   3. Silent dead-store: init_script writes to a name that isn't in
#      ``outputs``, isn't a private ``_*`` scratch var, isn't read elsewhere,
#      and looks like a typo of a declared output (e.g. ``output_a`` when
#      ``out_a`` is the declared output).  Pyflakes-style two-pass walk.
#   4. Empty ``inputs=[]`` paired with a script that reads from a name that
#      "looks like" an input (e.g. ``in_0``) — gentle hint.
#   5. Generalised case 4: when ``inputs=[]`` and the user references *any*
#      undefined symbol, the case-1 error message now suggests declaring it
#      in ``inputs=[...]`` (the most likely user intent).
#
# Conservative-by-design: the check is gated behind ``strict=True`` (default
# off → backwards-compatible).  False positives are worse than false negatives
# here; when in doubt, the walker treats a name as legitimate.

# Names that are always available in the exec'd environment (framework-bound
# or python-builtin).  ``time`` is bound by the framework; ``t`` is *not* —
# we keep ``t`` in the suggestion lookup table so we can hint about it.
_FRAMEWORK_BOUND_NAMES = frozenset({
    "time",            # framework binds this (CustomPythonBlock does not, see WC-98)
    "__main__",
    "true", "false",   # CustomPythonBlock.local_env_base
})

# Common math / array libraries the user is likely to reference without an
# explicit ``import`` line (CustomJaxBlock auto-injects some of these via
# the static_env on subsequent calls; legacy scripts assume they're present).
_COMMON_SCRIPT_GLOBALS = frozenset({
    "np", "numpy", "jnp", "jax", "scipy", "math",
    # Common math fns that legacy scripts call bare (e.g. ``y = sin(x)``).
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "log2",
    "sqrt", "pi", "e", "inf", "nan",
    "abs", "min", "max", "sum",
    # numpy / jax array constructors most often used bare.
    "array", "zeros", "ones", "empty", "arange", "linspace",
})

_PYTHON_BUILTINS = frozenset(builtins.__dict__.keys())


def _collect_assigned_and_referenced_names(tree: ast.AST):
    """Walk *tree* and return (assigned, referenced, imported) sets of bare
    Name identifiers.

    Conservative semantics:
      * ``assigned`` includes targets of ``Assign``, ``AnnAssign``,
        ``AugAssign``, ``NamedExpr`` (walrus), ``For`` loop vars, comprehension
        loop vars, ``with ... as x``, function/class definitions, exception
        ``as x`` handlers, and parameters of any nested function/lambda.
      * ``referenced`` includes any ``Name`` node in ``Load`` context that is
        not also a local in some enclosing comprehension / lambda — we only
        care about top-level free variables of the script.
      * ``imported`` includes names introduced by ``Import`` and ``ImportFrom``
        (using the alias if present).

    Anything we can't classify (e.g. ``__class__`` references in nested
    classes, fancy ``del`` statements) is conservatively dropped from
    ``referenced`` so we don't raise spurious errors.
    """
    assigned: set[str] = set()
    referenced: set[str] = set()
    imported: set[str] = set()

    # Walk function/class/comprehension scopes recursively, collecting their
    # local-only names so we can subtract them from the top-level "free
    # variable" set.
    def _collect_local_targets(node):
        """Names *bound* locally to this node (function args, comp targets)."""
        local: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for a in node.args.args:
                local.add(a.arg)
            for a in node.args.posonlyargs:
                local.add(a.arg)
            for a in node.args.kwonlyargs:
                local.add(a.arg)
            if node.args.vararg:
                local.add(node.args.vararg.arg)
            if node.args.kwarg:
                local.add(node.args.kwarg.arg)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                for n in ast.walk(gen.target):
                    if isinstance(n, ast.Name):
                        local.add(n.id)
        return local

    def _walk_assign_target(target):
        for n in ast.walk(target):
            if isinstance(n, ast.Name):
                assigned.add(n.id)

    # Explicit pre-pass for module-level statements only — we don't want to
    # confuse e.g. a function parameter named "x" with a module-level
    # reference to "x".
    for node in ast.walk(tree):
        # Top-level imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    # Star import: we can't statically know what names came
                    # in.  Bail out by adding a sentinel so downstream checks
                    # accept anything.
                    imported.add("*")
                else:
                    imported.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assigned.add(node.name)
        elif isinstance(node, ast.ClassDef):
            assigned.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                _walk_assign_target(tgt)
        elif isinstance(node, ast.AugAssign):
            _walk_assign_target(node.target)
        elif isinstance(node, ast.AnnAssign):
            _walk_assign_target(node.target)
        elif isinstance(node, ast.NamedExpr):
            _walk_assign_target(node.target)
        elif isinstance(node, ast.For):
            _walk_assign_target(node.target)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    _walk_assign_target(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                assigned.add(node.name)
        elif isinstance(node, ast.Global):
            for name in node.names:
                assigned.add(name)
        elif isinstance(node, ast.Nonlocal):
            for name in node.names:
                assigned.add(name)

    # Now collect referenced (Load-context) Name nodes, *but* skip those that
    # appear only inside a nested function / lambda / comprehension scope
    # where they're bound locally.  Conservative approach: collect every
    # Load-context Name and let the caller ignore it if it's also assigned
    # at module scope or imported.
    #
    # We additionally skip references that are bound as comprehension or
    # function parameters in their *immediate* enclosing scope, by walking
    # function/lambda/comp bodies separately.
    class _RefCollector(ast.NodeVisitor):
        def __init__(self):
            self.scope_stack: list[set[str]] = [set()]

        def _push_scope(self, locals_: set[str]):
            self.scope_stack.append(locals_)

        def _pop_scope(self):
            self.scope_stack.pop()

        def _is_local(self, name: str) -> bool:
            # A Name is "local" only if shadowed in an inner scope (excluding
            # the outermost module scope, which is what we're checking
            # against).
            return any(name in s for s in self.scope_stack[1:])

        def visit_FunctionDef(self, node):
            locals_ = _collect_local_targets(node)
            # Visit decorators / defaults in *enclosing* scope.
            for dec in node.decorator_list:
                self.visit(dec)
            for d in node.args.defaults:
                self.visit(d)
            for d in node.args.kw_defaults:
                if d is not None:
                    self.visit(d)
            # Body in inner scope.
            self._push_scope(locals_)
            for stmt in node.body:
                self.visit(stmt)
            self._pop_scope()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            locals_ = _collect_local_targets(node)
            for d in node.args.defaults:
                self.visit(d)
            for d in node.args.kw_defaults:
                if d is not None:
                    self.visit(d)
            self._push_scope(locals_)
            self.visit(node.body)
            self._pop_scope()

        def _visit_comp(self, node):
            locals_ = _collect_local_targets(node)
            self._push_scope(locals_)
            # The leftmost iterator runs in the *enclosing* scope per Python
            # semantics, but for our purposes treating the whole comp as
            # local is conservative-correct (we'd just miss a reference,
            # not surface a false positive).
            self.generic_visit(node)
            self._pop_scope()

        visit_ListComp = _visit_comp
        visit_SetComp = _visit_comp
        visit_DictComp = _visit_comp
        visit_GeneratorExp = _visit_comp

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                if not self._is_local(node.id):
                    referenced.add(node.id)

        def visit_ClassDef(self, node):
            # Class body's local namespace is opaque from outside — treat
            # method bodies as separate scopes.
            for dec in node.decorator_list:
                self.visit(dec)
            for base in node.bases:
                self.visit(base)
            self._push_scope({node.name})
            for stmt in node.body:
                self.visit(stmt)
            self._pop_scope()

    _RefCollector().visit(tree)

    return assigned, referenced, imported


def _collect_top_level_assignment_targets(tree: ast.AST) -> set[str]:
    """Collect *only* Name targets of ``Assign`` / ``AugAssign`` / ``AnnAssign``
    at module level (i.e. not inside a function/lambda/class/comprehension).

    This is the right granularity for the silent-dead-store check: a function
    definition or import alias is not a "store" in the dead-store sense, and
    a parameter / comprehension loop var is locally scoped — none of these
    can shadow an output port at exec time.
    """
    assigned: set[str] = set()

    # Walk only top-level statements; recurse into ``If`` / ``For`` / ``With``
    # / ``Try`` because Python flattens those into the same module namespace,
    # but stop at function / class / lambda bodies.
    def _walk_stmt(stmt):
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                for n in ast.walk(tgt):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        assigned.add(n.id)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                assigned.add(stmt.target.id)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                assigned.add(stmt.target.id)
        elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.stmt):
                    _walk_stmt(child)
        # FunctionDef / ClassDef / Lambda etc. are intentionally NOT recursed
        # into: a write inside their body is not a module-level write.

    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            _walk_stmt(stmt)
    return assigned


# Heuristics for "this name looks like the user meant an output" — used by
# the silent-dead-store check to decide whether the name is suspect enough
# to surface (vs. just a one-off scratch var).
def _looks_like_output_typo(name: str, outputs: list[str]) -> str | None:
    """If *name* looks like a typo of one of *outputs*, return the closest
    output; otherwise return None.

    Strategy: an explicit prefix match (``output_<basename>`` -> ``out_<basename>``,
    ``out<basename>`` -> ``out_<basename>``, etc.) plus a `difflib`
    similarity fallback gated to a high cutoff so we don't fire on unrelated
    scratch-var names.
    """
    import difflib

    if not outputs:
        return None
    # Direct prefix substitutions for the most common typos.
    for out in outputs:
        # ``output_a`` vs ``out_a``: shared suffix after stripping ``output_``
        # vs ``out_`` prefix.
        for src_prefix, tgt_prefix in (
            ("output_", "out_"),
            ("output", "out"),
            ("out_", "output_"),
        ):
            if name.startswith(src_prefix) and out.startswith(tgt_prefix):
                if name[len(src_prefix):] == out[len(tgt_prefix):]:
                    return out
    # Fall back to difflib similarity for other near-misses (single-char
    # transposition, etc.).  High cutoff (0.75) to keep false-positives down.
    matches = difflib.get_close_matches(name, outputs, n=1, cutoff=0.75)
    return matches[0] if matches else None


def _looks_like_input_name(name: str) -> bool:
    """Heuristic: does *name* look like an unbound input port reference?

    Used by case (5) when ``inputs=[]`` to decide whether to suggest the
    user declare the symbol in ``inputs=[...]``.  Conservative: only
    fires for names matching a small set of common conventions.
    """
    if name.startswith(("in_", "input_", "signal_", "u_")):
        return True
    if name in {"u", "x", "y_in"}:
        return True
    # ``in_0`` / ``in_1`` / ``input0`` etc.
    if name.startswith("input") and name[5:].isdigit():
        return True
    return False


def _validate_custom_block_signature(
    block,
    *,
    init_script: str,
    user_statements: str,
    finalize_script: str,
    inputs: list[str],
    outputs: list[str],
    parameter_names: list[str],
    has_time_binding: bool,
):
    """T-036e: AST-driven signature diagnostics for Custom*Block scripts.

    Raises ``BlockParameterError`` when a script references a name that
    cannot possibly resolve at eval time (typo / missing input / wrong
    time variable name).  Best-effort & conservative: when a check would
    require runtime context we have no static visibility into, the
    validator yields and lets the original eval-time error fire.

    Only enabled when the block is constructed with ``strict=True``.
    """

    # Build the union of names that are unconditionally available across
    # all three scripts (framework-bound + python builtins + common script
    # globals like ``np`` that the JAX block injects at exec-time).
    base_allowed: set[str] = set()
    base_allowed |= _FRAMEWORK_BOUND_NAMES
    base_allowed |= _COMMON_SCRIPT_GLOBALS
    base_allowed |= _PYTHON_BUILTINS
    base_allowed |= set(inputs)
    base_allowed |= set(outputs)
    base_allowed |= set(parameter_names)
    if has_time_binding:
        base_allowed.add("time")

    # init_script may *introduce* names that are then visible to
    # user_statements / finalize_script — we collect them on the first pass
    # and forward into the second.
    init_introduced: set[str] = set()

    def _check_one(script: str, label: str, extra_allowed: set[str]):
        if not script or not script.strip():
            return set()  # nothing introduced
        try:
            tree = ast.parse(script)
        except SyntaxError:
            # The compile() call later in __init__ will surface the syntax
            # error with proper diagnostics — skip our check here.
            return set()

        assigned, referenced, imported = _collect_assigned_and_referenced_names(tree)

        if "*" in imported:
            # Star-import in the script — bail out, we can't statically
            # decide what's defined.
            return assigned | imported

        allowed = (
            base_allowed
            | extra_allowed
            | assigned
            | imported
        )

        for name in sorted(referenced):
            if name in allowed:
                continue
            # Tolerate dunders and private names — common in user code.
            if name.startswith("_") and name.endswith("_"):
                continue

            # Helpful suggestion for the t-vs-time confusion.
            hint = ""
            if name == "t" and has_time_binding:
                hint = (
                    " (did you mean 'time'? the framework binds the "
                    "current simulation time as 'time', not 't'.)"
                )
            elif not inputs and (
                name in ("in_0", "in_1", "in_2")
                or _looks_like_input_name(name)
            ):
                # Case (5): empty inputs=[] paired with what looks like an
                # input read.  Suggest declaring the symbol as an input.
                hint = (
                    f" (the block has inputs=[]; declare {name!r} in "
                    "`inputs=[...]` to bind it.)"
                )
            elif not inputs:
                # Case (5) generalised: any unrecognised read with empty
                # inputs=[] is most plausibly a missing input declaration.
                hint = (
                    f" (the block has inputs=[]; if {name!r} is meant to "
                    "come from an upstream block, declare it in "
                    "`inputs=[...]`.)"
                )

            raise BlockParameterError(
                message=(
                    f"CustomBlock {block.name!r} {label} references "
                    f"undefined symbol {name!r}"
                    f"{hint}. "
                    f"Allowed symbols include: declared inputs={inputs}, "
                    f"outputs={outputs}, time"
                    + (", parameters=" + str(parameter_names) if parameter_names else "")
                    + "."
                ),
                system=block,
                parameter_name=label,
            )

        return assigned | imported

    init_introduced = _check_one(
        init_script, "init_script", extra_allowed=set()
    )

    # ``user_statements`` and ``finalize_script`` see everything init_script
    # introduced (function defs, imports, assigned globals).
    _check_one(
        user_statements, "user_statements", extra_allowed=init_introduced
    )
    _check_one(
        finalize_script, "finalize_script", extra_allowed=init_introduced
    )

    # ----- Case (3): silent dead-store detection --------------------------
    # Two-pass walk over init_script: collect module-level Name *writes*
    # (Assign / AugAssign / AnnAssign), then compute the union of *reads*
    # across init_script + user_statements + finalize_script.  Any name
    # that's written-and-never-read is suspect; if it also has a close
    # match in ``outputs`` we surface a typo hint.  Conservative: scratch
    # variables (private ``_*``, lowercase one-off names that don't look
    # like outputs) are never flagged.
    _check_silent_dead_store(
        block,
        init_script=init_script,
        user_statements=user_statements,
        finalize_script=finalize_script,
        outputs=outputs,
        inputs=inputs,
        parameter_names=parameter_names,
    )


def _check_silent_dead_store(
    block,
    *,
    init_script: str,
    user_statements: str,
    finalize_script: str,
    outputs: list[str],
    inputs: list[str],
    parameter_names: list[str],
) -> None:
    """T-036e case (3): pyflakes-style silent-dead-store guard for
    ``init_script``.

    The check fires only when an init-script *write* target satisfies ALL
    of the following:

    1. Is not in ``outputs`` (declared output port — legitimate).
    2. Does not start with ``_`` (Python private-scratch convention).
    3. Is not read in any of ``init_script`` / ``user_statements`` /
       ``finalize_script``.
    4. Has a close-match candidate in ``outputs`` (i.e. it *looks like*
       a typo of an output, e.g. ``output_a`` for ``out_a``).

    All four together — the check is intentionally narrow.  A pure scratch
    var that's later read (``tmp = 1.0; out_0 = tmp``) is fine; a private
    ``_tmp`` is fine; an unused name with no close match in outputs is
    fine (might be a documented constant).

    Raises ``BlockParameterError`` with a "did you mean ``<output>``?"
    hint when the heuristic flags a typo.
    """
    if not init_script or not init_script.strip():
        return
    try:
        init_tree = ast.parse(init_script)
    except SyntaxError:
        return  # syntax error caught later by compile()

    init_writes = _collect_top_level_assignment_targets(init_tree)
    if not init_writes:
        return

    # Union of all reads across the three scripts.
    all_reads: set[str] = set()
    for script in (init_script, user_statements, finalize_script):
        if not script or not script.strip():
            continue
        try:
            tree = ast.parse(script)
        except SyntaxError:
            continue
        _, referenced, _ = _collect_assigned_and_referenced_names(tree)
        all_reads |= referenced

    # Names that the user could legitimately have meant — not flagged.
    legit = (
        set(outputs)
        | set(inputs)
        | set(parameter_names)
    )

    suspect_unread = set()
    for name in init_writes:
        if name in legit:
            continue
        if name.startswith("_"):
            # Private scratch convention — never flagged.
            continue
        if name in all_reads:
            # Read elsewhere — not a dead-store.
            continue
        suspect_unread.add(name)

    if not suspect_unread:
        return

    # Of the suspects, only fire when there's a close-match output (the
    # smell that distinguishes "typo of output" from "documented unused
    # constant").  Conservative-by-design — false positives are bad.
    for name in sorted(suspect_unread):
        match = _looks_like_output_typo(name, outputs)
        if match is None:
            continue
        raise BlockParameterError(
            message=(
                f"CustomBlock {block.name!r} init_script assigns to "
                f"{name!r} but the value is never read and the name is "
                f"not in outputs={outputs}. Did you mean {match!r}? "
                "Rename the assignment target, or prefix the name with "
                "'_' to mark it as an intentional scratch variable."
            ),
            system=block,
            parameter_name="init_script",
        )


class CustomJaxBlock(LeafSystem):
    """JAX implementation of the PythonScript block.

    A few important notes and changes/limitations to this JAX implementation:
    - For this block all code must be written using the JAX-supported subset of Python:
        * Numerical operations should use `jax.numpy = jnp` instead of `numpy = np`
        * Standard control flow is not supported (if/else, for, while, etc.). Instead
            use `lax.cond`, `lax.fori_loop`, `lax.while_loop`, etc.
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#structured-control-flow-primitives
            Where possible, NumPy-style operations like `jnp.where` or `jnp.select` should
            be preferred to lax control flow primitives.
        * Functions must be pure and arrays treated as immutable.
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates
        Provided these assumptions hold, the code can be JIT compiled, differentiated,
        run on GPU, etc.
    - Variable scoping: the `init_code` and `step_code` are executed in the same scope,
        so variables declared in the `init_code` will be available in the `step_code`
        and can be modified in that scope. Internally, everything declared in
        `init_code` is treated as a single state-like cache entry.
        However, variables declared in the `step_code` will NOT persist between
        evaluations. Users should think of `step_code` as a normal Python function
        where locally declared variables will disappear on leaving the scope.
    - Persistent variables (outputs and anything declared in `init_code`) must have
        static shapes and dtypes. This means that you cannot declare `x = 0.0` in
        `init_code` and then later assign `x = jnp.zeros(4)` in `step_code`.

    These changes mean that many older PythonScript blocks may not be backwards compatible.

    Input ports:
        Variable number of input ports, one for each input variable declared in `inputs`.
        The order of the input ports is the same as the order of the input variables.

    Output ports:
        Variable number of output ports, one for each output variable declared in `outputs`.
        The order of the output ports is the same as the order of the output variables.

    Parameters:
        dt (float): The discrete time step of the block, or None if the block is
            in agnostic time mode.
        init_script (str): A string containing Python code that will be executed
            once when the block is initialized. This code can be used to declare
            persistent variables that will be available in the `step_code`.
        user_statements (str): A string containing Python code that will be executed
            once per time step (or per output port evaluation, in agnostic mode).
            This code can use the persistent variables declared in `init_script` and
            the block inputs.
        finalize_script (str): A string containing Python code that will be executed
            once when the simulation completes (or is otherwise torn down). This code
            can use the persistent variables declared in `init_script`. Supported for
            :class:`CustomPythonBlock` only; raises :class:`PythonScriptError` for
            :class:`CustomJaxBlock`.
        accelerate_with_jax (bool): If True, the block will be JIT compiled. If False,
            the block will be executed in pure Python.  This parameter exists for
            compatibility with UI options; when creating pure Python blocks from code
            (e.g. for testing), explicitly create the CustomPythonBlock class.
        time_mode (str): One of "discrete" or "agnostic". If "discrete", the block
            step code will be evaluated at peridodic intervals specified by "dt".
            If "agnostic", the block step code will be evaluated once per output
            port evaluation, and the block will not have a discrete time step.
        inputs (List[str]): A list of input variable names. The order of the input
            ports is the same as the order of the input variables.
        outputs (Mapping[str, Tuple[DTypeLike, ShapeLike]]): A dictionary mapping
            output variable names to a tuple of dtype and shape. The order of the
            output ports is the same as the order of the output variables.
        static_parameters (Mapping[str, Array]): A dictionary mapping parameter names to
            values. Parameters are treated as immutable and cannot be modified in
            the step code. Static parameters can't be used in ensemble simulations or
            optimization workflows.
        dynamic_parameters (Mapping[str, Array]): A dictionary mapping parameter names to
            values. Parameters are treated as immutable and cannot be modified in
            the step code. Dynamic parameters can be arrays or scalars, but must have static
            shapes and dtypes in order to support JIT compilation.
    """

    @declare_parameters(
        static=[
            "dt",
            "init_script",
            "user_statements",
            "finalize_script",
            "accelerate_with_jax",
            "time_mode",
        ]
    )
    def __init__(
        self,
        dt: float = None,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",
        accelerate_with_jax: bool = True,
        time_mode: str = "discrete",  # [discrete, agnostic]
        inputs: List[str] = None,  # [name]
        outputs: List[str] = None,
        dynamic_parameters: Mapping[str, Array] = None,
        static_parameters: Mapping[str, Array] = None,
        strict: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        dynamic_parameters = dynamic_parameters if dynamic_parameters else {}
        static_parameters = static_parameters if static_parameters else {}

        if time_mode not in ["discrete", "agnostic"]:
            raise BlockInitializationError(
                f"Invalid time mode '{time_mode}' for PythonScript block", system=self
            )

        if time_mode == "discrete" and dt is None:
            raise BlockInitializationError(
                "When in discrete time mode, dt is required for block", system=self
            )

        self.time_mode = time_mode

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []

        # T-036e: validate inputs / outputs port-name lists eagerly so opaque
        # downstream errors (duplicate-port AssertionError, AttributeError on
        # exec'd statement, etc.) are replaced by a clear named-block error.
        _validate_custom_block_io_names(self, inputs, outputs)

        # T-036e (deeper): when strict=True, AST-walk the init / step /
        # finalize scripts to catch typo'd symbol references at construction
        # time rather than at first eval.  Default-off for backwards-compat.
        if strict:
            param_names = list(
                (dynamic_parameters or {}).keys()
            ) + list((static_parameters or {}).keys())
            # CustomJaxBlock binds ``time`` in the exec env; CustomPythonBlock
            # historically did not (see WC-98 comment at exec_step) — but
            # both subclasses funnel through here, so we allow ``time`` and
            # let the t-vs-time hint catch the legacy mistake.
            _validate_custom_block_signature(
                self,
                init_script=init_script,
                user_statements=user_statements,
                finalize_script=finalize_script,
                inputs=inputs,
                outputs=outputs,
                parameter_names=param_names,
                has_time_binding=True,
            )

        self.dt = dt

        # Note: 'optimize' level could be lowered in debug mode
        try:
            self.init_code = compile(
                init_script, filename="<init>", mode="exec", optimize=2
            )
        except BaseException as e:
            raise PythonScriptError(
                f"Syntax error in init_script for PythonScript block '{self.name_path_str}': {e}",
                system=self,
            ) from e

        try:
            self.step_code = compile(
                user_statements, filename="<step>", mode="exec", optimize=2
            )
        except BaseException as e:
            raise PythonScriptError(
                f"Syntax error in user_statements for PythonScript block '{self.name_path_str}': {e}",
                system=self,
            ) from e

        self._has_finalize_script = bool(finalize_script.strip())
        try:
            self.finalize_code = compile(
                finalize_script, filename="<finalize>", mode="exec", optimize=2
            )
        except BaseException as e:
            raise PythonScriptError(
                f"Syntax error in finalize_script for PythonScript block '{self.name_path_str}': {e}",
                system=self,
            ) from e

        if finalize_script != "" and not isinstance(self, CustomPythonBlock):
            raise PythonScriptError(
                f"PythonScript block '{self.name_path_str}' has a finalize_script "
                "but this is only supported for CustomPythonBlock (non-JAX) blocks.",
                system=self,
                parameter_name="finalize_script",
            )

        # Declare parameters
        for param_name, value in dynamic_parameters.items():
            if isinstance(value, list):
                value = npa.asarray(value)
            as_array = isinstance(value, npa.ndarray) or npa.isscalar(value)
            self.declare_dynamic_parameter(param_name, value, as_array=as_array)

        for param_name, value in static_parameters.items():
            self.declare_static_parameter(param_name, value)

        # Run the init_script
        persistent_env = self.exec_init()

        # Declare an input port for each of the input variables
        self.input_names = inputs
        for name in inputs:
            self.declare_input_port(name)

        # Declare a cache component for each of the output variables
        self._create_cache_type(outputs)

        if time_mode == "discrete":
            self._configure_discrete(dt, outputs, persistent_env)
        else:
            self._configure_agnostic(outputs, persistent_env)

    def initialize(
        self,
        dt: float = None,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",
        accelerate_with_jax: bool = True,
        time_mode: str = "discrete",  # [discrete, agnostic]
        **parameters,
    ):
        pass

    def _initialize_outputs(self, outputs, persistent_env):
        default_outputs = {name: None for name in outputs}

        for name in outputs:
            # If the initial value is set explicitly in the init script,
            # override the default value.  We don't need to do this for
            # agnostic configuration since the outputs will be calculated
            # every evaluation anyway.
            if name in persistent_env:
                value = npa.asarray(persistent_env[name])
                default_outputs[name] = value

                # Also update the persistent environment so that the data types
                # are consistent with the state.
                persistent_env[name] = value

            # Otherwise throw an error, since we don't know what the initial values
            # should be, or even what shape/dtype they should have.
            else:
                msg = (
                    f"Output variable '{name}' not explicitly initialized in "
                    "init_script for PythonScript block in 'Discrete' time mode. "
                    "Either initialize the variable as an array with the correct "
                    "shape and dtype, or make the block time mode 'Agnostic'."
                )
                raise PythonScriptError(message=msg, system=self)

        return self.CacheType(
            persistent_env=persistent_env,
            **default_outputs,
        )

    def _configure_discrete(self, dt, outputs, persistent_env):
        default_values = self._initialize_outputs(outputs, persistent_env)

        # The step function acts as a periodic update that will update all components
        # of the discrete state.
        self.step_callback_index = self.declare_cache(
            self.exec_step,
            period=dt,
            offset=dt,
            requires_inputs=True,
            default_value=default_values,
        )

        cache = self.callbacks[self.step_callback_index]

        # Get the index into the state cache (different in general from the index
        # into the callback list, since not all callbacks are cached).
        self.step_cache_index = cache.cache_index

        def _make_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                return getattr(state.cache[self.step_cache_index], o_port_name)

            return _output

        # Declare output ports for each state variable
        for o_port_name in outputs:
            self.declare_output_port(
                _make_callback(o_port_name),
                name=o_port_name,
                prerequisites_of_calc=[cache.ticket],
                requires_inputs=False,
                period=dt,
                offset=0.0,
            )

    def _configure_agnostic(self, outputs, persistent_env):
        # Create a callback to evaluate the step code and extract the
        # output. Note that this is inefficient since the step code will
        # be evaluated once _for each output port_, but it's the only way
        # to do this unless (until) we implement some variety of block
        # or function pre-ordering.
        def _make_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                xd = self.exec_step(time, state, *inputs, **parameters)
                return getattr(xd, o_port_name)

            return _output

        # Declare output ports for each state variable
        for o_port_name in outputs:
            self.declare_output_port(
                jit(_make_callback(o_port_name)),
                name=o_port_name,
                requires_inputs=True,
            )

        # This callback doesn't need to do anything since it's never
        # actually called - the cache here just stores the initial environment
        # and the output ports are evaluated directly.  This should be changed
        # to avoid re-evaluation with multiple output ports once we can do full
        # function ordering.
        def _cache_callback(time, state, *inputs, **parameters):
            return state.cache[self.step_cache_index]

        # Since this is the return type for `exec_step` we have to declare all
        # the output ports as entries in the namedtuple, even though those values
        # won't actually be cached in "agnostic" time mode.  This is just so that
        # both "discrete" and "agnostic" modes can share the same code.
        default_values = self.CacheType(
            persistent_env=persistent_env,
            **{o_port_name: None for o_port_name in outputs},
        )
        self.step_callback_index = self.declare_cache(
            _cache_callback,
            default_value=default_values,
            requires_inputs=False,
            prerequisites_of_calc=[inport.ticket for inport in self.input_ports],
        )

        cache = self.callbacks[self.step_callback_index]
        self.step_cache_index = cache.cache_index

    def _create_cache_type(self, outputs):
        # Store the output ports as a name for type inference and casting
        self.output_names = outputs

        # Also store the dictionary of local environment variables as a cache entry
        # This is the only persistent state of the system (besides outputs) - anything
        # declared in the "step" function will be forgotten at the end of the step

        self.CacheType = namedtuple("CacheType", self.output_names + ["persistent_env"])

    @property
    def local_env_base(self):
        # Define a starting point for the local code execution environment.
        # we have to inclide __main__ so that the code behaves like a module.
        # this allows for code like this:
        #   imports ...
        #   a = 1
        #   def f(b):
        #       return a+b
        #   out_0 = f(2)
        #
        # without getting a 'a not defined' error.
        return {
            "__main__": {},
        }

    def exec_init(self) -> dict[str, Array]:
        # Before executing the step code, we have to build up the local environment.
        # This includes specified modules, python block user defined parameters.

        default_parameters = {
            name: param.get() for name, param in self.dynamic_parameters.items()
        }

        local_env = {
            **self.local_env_base,
            **default_parameters,
        }

        # similar to above where we included __main__ so the code behaves as a module,
        # here we have to pass the local_env with __main__ as 1] globals, since that
        # is what allow the code to be executed as a module. 2] local since that is where
        # the new bindings will be written, that we need to retain since the code in step_code
        # may depend on these bindings.
        try:
            _default_exec(
                self.init_code,
                local_env,
                logger_=logger,
                system=self,
                code_name="init",
            )

        except BaseException as e:
            logger.error(
                "PythonScript block '%s' init script failed",
                self.name_path_str,
                **logdata(block=self),
            )
            raise PythonScriptError(system=self) from e

        # persistent_env contains bindings for parameters and for values from init_script
        persistent_env, static_env = _filter_non_traceable(local_env)

        # Since this is called during block initialization and not any JIT-compiled code,
        # we can safely store any untraceable variables as block attributes.  For example,
        # this may contain custom functions, classes, etc.
        self.static_env = static_env

        return persistent_env

    def exec_step(self, time: float, state: LeafState, *inputs, **parameters):
        # Before executing the step code, we have to build up the local environment.
        # This includes the persistent variables (anything declared in `init_code`),
        # time, block inputs, user-defined parameters, and specified modules.

        # Retrieve the variables declared in `init_code` from the discrete state
        full_env = state.cache[self.step_cache_index]
        persistent_env = full_env.persistent_env

        # Inputs are in order of port declaration, so they match `self.input_names`
        input_env = dict(zip(self.input_names, inputs))

        # Create a dictionary of all the information that the step function will need
        base_copy = self.local_env_base.copy()
        local_env = {
            **self.static_env,
            **base_copy,
            **persistent_env,
            **input_env,
            **parameters,
        }

        # Execute the step code in the local environment
        try:
            _default_exec(
                self.step_code,
                local_env,
                logger_=logger,
                inputs=input_env,
                system=self,
                code_name="step",
            )

        except PythonScriptError:
            raise
        except BaseException as e:
            logger.error(
                "PythonScript block '%s' step failed.",
                self.name_path_str,
                **logdata(block=self),
            )
            raise PythonScriptError(system=self) from e

        # Updated state variables are stored in the local environment
        xd = {name: local_env[name] for name in self.output_names}

        # Store the persistent variables in the corresponding discrete state
        xd["persistent_env"] = {key: local_env[key] for key in persistent_env}

        # Make sure the results have a consistent data type
        for name in self.output_names:
            xd[name] = npa.asarray(local_env[name])

            # Also make sure the value stored in the persistent environment
            # has the same data type
            if name in persistent_env:
                xd["persistent_env"][name] = xd[name]

        return self.CacheType(**xd)

    def check_types(
        self,
        context: ContextBase,
        error_collector: ErrorCollector = None,
    ):
        """Test-compile the init and step code to check for errors."""
        try:
            # Note that exec_step doesn't use parameters or time
            inputs = self.collect_inputs(context)
            jit(self.exec_step)(None, context[self.system_id].state, *inputs)
        except BaseException as exc:
            with ErrorCollector.context(error_collector):
                name_error = _caused_by_nameerror(exc)
                if name_error and name_error.name == "time":
                    raise PythonScriptTimeNotSupportedError(system=self) from exc
                if isinstance(exc, PythonScriptError):
                    raise
                raise PythonScriptError(system=self) from exc


class CustomPythonBlock(CustomJaxBlock):
    """Container for arbitrary user-defined Python code.

    Implemented to support legacy PythonScript blocks.

    Not traceable (no JIT compilation or autodiff). The internal implementation
    and behavior of this block differs vastly from the JAX-compatible block as
    this block stores state directly within the Python instance. Objects
    and modules can be kept as discrete state.

    Note that in "agnostic" mode, the step code will be evaluated _once per
    output port evaluation_. Because locally defined environment variables
    (in the init script) are preserved between evaluations, any mutation of
    these variables will be preserved. This can lead to unexpected behavior
    and should be avoided. Stateful behavior should be implemented using
    discrete state variables instead.

    Warning: The finalize_script parameter is currently accepted but not executed. 
    This is a known limitation. Do not rely on finalize_script for cleanup operations.
    """

    __exec_fn = _default_exec

    def __init__(
        self,
        dt: float = None,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",
        inputs: List[str] = None,  # [name]
        outputs: List[str] = None,
        accelerate_with_jax: bool = False,
        time_mode: str = "discrete",
        static_parameters: Mapping[str, Array] = None,
        strict: bool = False,
        **kwargs,
    ):
        self._static_data_initialized = False
        self._parameters = static_parameters or {}
        self._persistent_env = {}

        # Per-block mutable module state (e.g. numpy error policy, random seed).
        # Populated after exec_init; applied + restored around every exec_step.
        # This provides isolation between multiple CustomPythonBlock instances
        # that share the same module objects from sys.modules.
        self._block_module_state: dict = {}

        # Will populate return type information during static initialization
        self.result_shape_dtypes = None
        self.return_dtypes = None

        super().__init__(
            dt=dt,
            init_script=init_script,
            user_statements=user_statements,
            finalize_script=finalize_script,
            inputs=inputs,
            outputs=outputs,
            accelerate_with_jax=accelerate_with_jax,
            time_mode=time_mode,
            static_parameters=self._parameters,
            strict=strict,
            **kwargs,
        )

        if time_mode == "agnostic" and npa.active_backend == "jax":
            logger.warning(
                "System %s is in agnostic time mode but is not traced with JAX. Be "
                "advised that the step code will be evaluated once per output port "
                "evaluation. Any mutation of the local environment should be strictly "
                "avoided as it will likely lead to unexpected behavior.",
                self.name_path_str,
            )

    def initialize(self, **kwargs):
        pass

    @property
    def has_feedthrough_side_effects(self) -> bool:
        # See explanation in `SystemBase.has_ode_side_effects`.
        return self.time_mode == "agnostic"

    @staticmethod
    def set_exec_fn(exec_fn: callable):
        CustomPythonBlock.__exec_fn = exec_fn

    @property
    def local_env_base(self):
        # Define a starting point for the local code execution environment.
        return {
            "__main__": {},
            "true": True,
            "false": False,
        }

    def exec_init(self) -> None:
        default_parameters = {
            name: param.get() for name, param in self.dynamic_parameters.items()
        }

        local_env = {
            **self.local_env_base,
            **self._parameters,
            **default_parameters,
        }

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            code=self.init_code,
            env=local_env,
            logger_=logger,
            system=self,
            code_name="init",
        )

        # Snapshot global module state BEFORE running init_script so we can
        # detect which changes the init_script introduces (e.g. numpy.seterr).
        pre_init_global = _save_module_state(local_env)

        try:
            io_callback(exec_fn, None)
        except KeyboardInterrupt as e:
            logger.error(
                "Python block '%s' init script execution was interrupted.",
                self.name,
                **logdata(block=self),
            )
            raise PythonScriptError(
                message="Python block init script execution was interrupted.",
                system=self,
            ) from e
        except PythonScriptError as e:
            logger.error("%s: init script failed.", self.name, **logdata(block=self))
            raise e
        except BaseException as e:
            logger.error("%s: init script failed.", self.name, **logdata(block=self))
            raise PythonScriptError(system=self) from e

        # Capture the module state after init_script ran.  This becomes the
        # block's "initial" isolated state for subsequent exec_step calls.
        self._block_module_state = _save_module_state(local_env)

        # Restore the global module state so that this block's init_script
        # does not contaminate other blocks' initialisation.
        _restore_module_state(pre_init_global)

        self._persistent_env = local_env

        return None

    def exec_step(self, time, state, *inputs, **parameters):
        if not self._static_data_initialized:
            # return_dtypes is inferred in initialize_static_data()
            raise PythonScriptError(
                "Trying to execute step code before static data has been initialized",
                system=self,
            )
        logger.debug(
            "Executing step for %s with state=%s, inputs=%s",
            self.name,
            state,
            inputs,
        )

        # Inputs are in order of port declaration, so they match `self.input_names`
        input_env = dict(zip(self.input_names, inputs))

        base_copy = self.local_env_base.copy()
        local_env = {
            **base_copy,
            **self._persistent_env,
            **parameters,
        }

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            code=self.step_code,
            env=local_env,
            logger_=logger,
            return_vars=self.output_names,
            return_dtypes=self.return_dtypes,
            system=self,
            code_name="step",
        )

        def wrapped_exec_fn(inputs):
            # --- module isolation: checkpoint / apply / restore ---
            # 1. Save the current global state (may have been mutated by another block)
            global_state_before = _save_module_state(local_env)
            # 2. Apply this block's remembered module state
            _restore_module_state(self._block_module_state)
            try:
                result = exec_fn(inputs=inputs)
            except KeyboardInterrupt:
                logger.error(
                    "Python block '%s' step script execution was interrupted.",
                    self.name,
                    **logdata(block=self),
                )
                raise
            except NameError as e:
                err_msg = (
                    f"Python block '{self.name}' step script execution failed with a NameError on"
                    + f" missing variable '{e.name}'."
                    + " All names used in this script should be declared in the init script."
                    + f" The execution environment contains the following names: {', '.join(list(local_env.keys()))}"
                )
                logger.error(err_msg)
                logger.error("NameError: %s", e, **logdata(block=self))
                raise PythonScriptError(system=self) from e
            except PythonScriptError as e:
                logger.error("%s: exec_step failed.", self.name, **logdata(block=self))
                raise e
            except BaseException as e:
                logger.error("%s: exec_step failed.", self.name, **logdata(block=self))
                raise PythonScriptError(system=self) from e
            else:
                # 3. Capture any module state changes made by this block's step
                self._block_module_state = _save_module_state(local_env)
                return result
            finally:
                # 4. Always restore the global state that was in place before
                #    this block ran so that other blocks are unaffected.
                _restore_module_state(global_state_before)

        return_vars = io_callback(
            wrapped_exec_fn,
            self.result_shape_dtypes,
            inputs=input_env,
        )

        # Keep local env for next step but only if defined in init_script
        # NOTE: If this restriction turns out to be counterproductive, we can
        # remove it and remove the NameError handling above as well. The thinking
        # here is that this could help avoiding stuff like `if time == 0: x = 0`
        # See https://jaxonomy.atlassian.net/browse/WC-98
        self._persistent_env = {
            key: local_env[key] for key in self._persistent_env if key in local_env
        }

        # Updated state variables are stored in the local environment
        xd = {name: return_vars[i] for i, name in enumerate(self.output_names)}

        return self.CacheType(persistent_env=None, **xd)

    def _initialize_outputs(self, outputs, _persistent_env):
        # Override the base implemenetation since `persistent_env` will be None
        # in this case. Instead, pass the class attribute where the environment
        # is actually maintained.
        default_outputs = {name: None for name in outputs}
        default_values = self.CacheType(
            persistent_env=self._persistent_env,
            **default_outputs,
        )
        default_values = super()._initialize_outputs(outputs, self._persistent_env)
        default_outputs = default_values._asdict()
        self._persistent_env = default_outputs.pop("persistent_env")

        # Determine return data types
        self._initialize_result_shape_dtypes(
            [default_outputs[output] for output in outputs]
        )

        return self.CacheType(
            persistent_env=None,
            **default_outputs,
        )

    def _initialize_result_shape_dtypes(self, outputs):
        self.result_shape_dtypes = []
        self.return_dtypes = []
        for value in outputs:
            self.result_shape_dtypes.append(
                jax.ShapeDtypeStruct(value.shape, value.dtype)
            )
            self.return_dtypes.append(value.dtype)

    def initialize_static_data(self, context):
        # If in agnostic mode, call the step function once to determine the
        # data types and then store those in result_shape_dtype and return_dtypes.
        context = LeafSystem.initialize_static_data(self, context)

        if self.result_shape_dtypes is not None:
            # These data types are already known (block is in discrete mode)
            self._static_data_initialized = True
            return context

        inputs = self.collect_inputs(context)
        input_env = dict(zip(self.input_names, inputs))

        base_copy = self.local_env_base.copy()
        local_env = {
            **base_copy,
            **self._persistent_env,
        }

        # Will not do any type conversion
        return_dtypes = [None for _ in self.output_names]

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            self.step_code,
            local_env,
            logger_=logger,
            return_vars=self.output_names,
            return_dtypes=return_dtypes,
            system=self,
            code_name="step",
        )

        return_vars = exec_fn(inputs=input_env)

        self._initialize_result_shape_dtypes(return_vars)

        self._static_data_initialized = True

        return context

    def exec_finalize(self) -> None:
        """Execute the finalize_script using the current persistent environment.

        Called once at the end of the simulation via :meth:`post_simulation_finalize`.
        The script runs in the same environment that was maintained throughout the
        simulation, so all variables declared in ``init_script`` (and updated by
        ``user_statements``) are available.

        Has no effect if ``finalize_script`` is empty.
        """
        if not self._has_finalize_script:
            return

        local_env = {
            **self.local_env_base,
            **self._persistent_env,
        }

        exec_fn = functools.partial(
            CustomPythonBlock.__exec_fn,
            code=self.finalize_code,
            env=local_env,
            logger_=logger,
            system=self,
            code_name="finalize",
        )

        try:
            io_callback(exec_fn, None)
        except KeyboardInterrupt as e:
            logger.error(
                "Python block '%s' finalize script execution was interrupted.",
                self.name,
                **logdata(block=self),
            )
            raise PythonScriptError(
                message="Python block finalize script execution was interrupted.",
                system=self,
            ) from e
        except PythonScriptError as e:
            logger.error("%s: finalize script failed.", self.name, **logdata(block=self))
            raise e
        except BaseException as e:
            logger.error("%s: finalize script failed.", self.name, **logdata(block=self))
            raise PythonScriptError(system=self) from e

    def post_simulation_finalize(self) -> None:
        """Run ``finalize_script`` and then call the base-class hook."""
        self.exec_finalize()
        return super().post_simulation_finalize()

    def check_types(
        self,
        context: ContextBase,
        error_collector=None,
    ):
        pass
