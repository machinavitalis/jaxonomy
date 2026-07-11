# SPDX-License-Identifier: MIT

"""
This module defines a system for managing and manipulating parameters in jaxonomy.

Key Components:

1. **Parameter Class**: Represents a parameter with a value that can be an expression,
another parameter, or a variety of data types such as arrays, numbers, or strings.
It supports simple operations like addition, subtraction, multiplication, etc.
More complex expressions can be represented as python expressions, which are evaluated.
For example, a parameter value can be `np.eye(p)` where `p` is a parameter.

2. **ParameterCache Class**: Manages the cache and dependencies of `Parameter` objects.
It ensures that when a parameter's value changes, all dependent parameters are
invalidated and recalculated as needed.

SystemBase objects use Parameter objects to store dynamic and static parameters.
In particular, jaxonomy model parameters are represented in the Diagram system
as dynamic parameters.

SystemBase objects do not need to manage Parameter objects beyond their
declaration, as these parameters are resolved before invoking the methods of
the SystemBase object. This is achieved by the `parameters` decorator
in system_base.py, `InitializeParameterResolver` in leaf_system.py and the
context creation logic that resolves all parameters before a simulation run.

Example usage:

```
c = Parameter(value=1.0)
builder = jaxonomy.DiagramBuilder()
constant1 = builder.add(library.Constant(c))
constant2 = builder.add(library.Constant(c + 1))
diagram = builder.build()

context = diagram.create_context()
constant1.output_ports[0].eval(context) # 1.0
constant2.output_ports[0].eval(context) # 2.0

c.set(2.0)
context = diagram.create_context()
constant1.output_ports[0].eval(context) # out = 2.0
constant2.output_ports[0].eval(context) # out = 3.0

c.get() # 2.0
```

"""

import ast
import copy
from collections import defaultdict
import dataclasses
import enum
from functools import wraps
import threading
from typing import Union, TYPE_CHECKING

import jax
from jax import Array
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from . import build_recorder
from .error import ParameterError
from ..backend import utils
from ..backend.backend import IS_JAXLITE
from ..backend.typing import ArrayLike, DTypeLike, ShapeLike


if TYPE_CHECKING:
    from ..system.system_base import SystemBase


class ParameterExpr(list):
    pass


class Ops(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    FLOORDIV = enum.auto()
    MOD = enum.auto()
    POW = enum.auto()
    NEG = enum.auto()
    POS = enum.auto()
    ABS = enum.auto()
    EQ = enum.auto()
    NE = enum.auto()
    LT = enum.auto()
    LE = enum.auto()
    GT = enum.auto()
    GE = enum.auto()
    MATMUL = enum.auto()


__OPS_FN__ = {
    Ops.ADD: lambda x, y: x + y,
    Ops.SUB: lambda x, y: x - y,
    Ops.MUL: lambda x, y: x * y,
    Ops.DIV: lambda x, y: x / y,
    Ops.FLOORDIV: lambda x, y: x // y,
    Ops.MOD: lambda x, y: x % y,
    Ops.POW: lambda x, y: x**y,
    Ops.NEG: lambda x: -x,
    Ops.POS: lambda x: +x,
    Ops.ABS: abs,
    Ops.EQ: lambda x, y: x == y,
    Ops.NE: lambda x, y: x != y,
    Ops.LT: lambda x, y: x < y,
    Ops.LE: lambda x, y: x <= y,
    Ops.GT: lambda x, y: x > y,
    Ops.GE: lambda x, y: x >= y,
    Ops.MATMUL: lambda x, y: x @ y,
}

__OPS_STR__ = {
    Ops.ADD: "+",
    Ops.SUB: "-",
    Ops.MUL: "*",
    Ops.DIV: "/",
    Ops.FLOORDIV: "//",
    Ops.MOD: "%",
    Ops.POW: "**",
    Ops.NEG: "-",
    Ops.POS: "+",
    Ops.ABS: "abs",
    Ops.EQ: "==",
    Ops.NE: "!=",
    Ops.LT: "<",
    Ops.LE: "<=",
    Ops.GT: ">",
    Ops.GE: ">=",
    Ops.MATMUL: "@",
}


class _VarRecorder(ast.NodeVisitor):
    """Used to record all variables in a Python expression."""

    def __init__(self):
        super().__init__()
        self.vars = set()

    def visit_Name(self, node):
        self.vars.add(node.id)
        return node.id


ArrayLikeTypes = (
    Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_,
    np.number,  # NumPy scalar types
    bool,
    int,
    float,
    complex,  # Python scalar types
)


def _expr_parameter_refs(python_expr: str, env: dict) -> set["Parameter"]:
    """Parameters referenced by name inside a Python-expression value.

    Unlike :func:`resolve_parameters` this does not call ``get()`` on the
    referenced parameters, so it is safe to run at construction/copy time
    before the parameters are necessarily computable.
    """
    try:
        tree = ast.parse(python_expr, mode="eval")
    except SyntaxError:
        # A malformed expression surfaces with a useful error at compute
        # time (ParameterCache.__compute__); don't fail construction here.
        return set()
    var_recorder = _VarRecorder()
    var_recorder.visit(tree.body)
    return {
        env[var]
        for var in var_recorder.vars
        if isinstance(env.get(var), Parameter)
    }


def resolve_parameters(python_expr: str, env: dict, mode="eval"):
    # look for variables used in the expression
    tree = ast.parse(python_expr, mode=mode)
    var_recorder = _VarRecorder()
    var_recorder.visit(tree.body)

    # record the parameters and resolve them
    parameters = set()
    resolved_params = {}
    for var in var_recorder.vars:
        if var in env:
            if isinstance(env[var], Parameter):
                parameters.add(env[var])
                resolved_params[var] = env[var].get()

    return parameters, resolved_params


def _resolve_array_like(value: ArrayLike) -> ArrayLike:
    if value.ndim == 0:
        return value

    # If jax array, there can not be dependents because all elements
    # are numerical.
    if isinstance(value, Array):
        return value

    # If ndarray, only resolve if dtype is 'object'. Other dtypes
    # can not contain Parameter elements.
    if isinstance(value, np.ndarray) and value.dtype != np.object_:
        return value

    # FIXME: this changes the array from a well-formed ndarray
    # like: array([[1,2], [3,4]]) to an array of arrays
    # like: array([array([1, 2]), array([3, 4])])
    # The latter form becomes extremely inefficient in JAX.
    # In the vast majority of cases though, we have numerical arrays
    # that have already been resolved to their final values.
    vals = []
    for val in value:
        if isinstance(val, Parameter):
            vals.append(ParameterCache.__compute__(val))
        else:
            vals.append(_resolve_array_like(val))

    numeric_types = (int, float, complex, np.number, bool)
    is_numeric = all(isinstance(v, numeric_types) for v in vals)
    if not is_numeric:
        return vals

    return np.array(vals)


def _resolve_array_param_value(param: "Parameter") -> ArrayLike:
    if not isinstance(param.value, (Array, np.ndarray)):
        raise ValueError("param.value must be an Array or ndarray")

    if not ParameterCache.get_dependents(param):
        return param.value

    return _resolve_array_like(param.value)


def _list_to_str(lst: list):
    str_repr = []
    for val in lst:
        if isinstance(val, list):
            str_repr.append(_list_to_str(val))
        else:
            str_repr.append(str(val))
    return f"[{', '.join(str_repr)}]"


def _tuple_to_str(tpl):
    str_repr = []
    for val in tpl:
        if isinstance(val, tuple):
            str_repr.append(_tuple_to_str(val))
        else:
            str_repr.append(str(val))
    if len(str_repr) == 1:
        return f"({str_repr[0]},)"
    return f"({', '.join(str_repr)})"


def _compute_list(tpl, is_tuple):
    new_lst = []
    for val in tpl:
        if isinstance(val, Parameter):
            new_lst.append(val.get())
        elif isinstance(val, list):
            new_lst.append(_compute_list(val, is_tuple=False))
        elif isinstance(val, tuple):
            new_lst.append(_compute_list(val, is_tuple=True))
        else:
            new_lst.append(val)
    if is_tuple:
        return tuple(new_lst)
    return new_lst


def _add_dependents(lst: list | tuple, param):
    for val in lst:
        if isinstance(val, Parameter):
            ParameterCache.add_dependent(val, param)
        elif isinstance(val, (list, tuple)):
            _add_dependents(val, param)


def _str_to_expression(s: str) -> str:
    # repr will add quotes and escape the string so that it can be passed again to
    # eval().
    return repr(s)


def _array_to_str(arr: Array | np.ndarray):
    if isinstance(arr, jax.core.Tracer):
        # in case this function is called inside JAX jit
        return f"np.array(<unknown>, dshape={arr.shape} dtype=np.{arr.dtype})"

    if arr.ndim == 0:
        return str(arr.item())

    if isinstance(arr, Array):
        # Should we serialize as jnp.array?
        if arr.weak_type:
            return f"np.array({arr.tolist()})"
        if arr.dtype in (np.int64, np.float64):
            return f"np.array({arr.tolist()})"
        return f"np.array({arr.tolist()}, dtype=np.{arr.dtype})"

    if arr.dtype != np.object_:
        if arr.dtype in (np.int64, np.float64):
            return f"np.array({arr.tolist()})"
        if np.issubdtype(arr.dtype, np.str_):
            return f"np.array({arr.tolist()})"
        return f"np.array({arr.tolist()}, dtype=np.{arr.dtype})"

    return f"np.array({_list_to_str(arr.tolist())})"


def _value_as_str(value) -> str:
    # Return '' for None because we parse '' coming from the UI as None
    if value is None:
        return ""

    if isinstance(value, ArrayLikeTypes):
        if isinstance(value, (Array, np.ndarray)):
            return _array_to_str(value)
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, np.number):
            dtype = value.dtype
            return f"np.{dtype}({value.item()})"
        return str(value)

    if isinstance(value, ParameterExpr):
        i = 0
        str_repr = []
        while i < len(value):
            val = value[i]
            if isinstance(val, Parameter):
                val_str = val.name if val.name is not None else str(val)
                if isinstance(val.value, ParameterExpr):
                    val_str = f"({val_str})"
                str_repr.append(val_str)
            elif isinstance(val, Ops):
                if val in (Ops.NEG, Ops.POS, Ops.ABS):
                    if i + 1 >= len(value):
                        raise ValueError()
                    next_val = value[i + 1]
                    if val is Ops.ABS:
                        str_repr.append(f"abs({next_val})")
                    elif val is Ops.NEG:
                        str_repr.append(f"-{next_val}")
                    elif val is Ops.POS:
                        str_repr.append(f"+{next_val}")
                    i += 1
                else:
                    str_repr.append(__OPS_STR__[val])
            elif isinstance(val, (Array, np.ndarray)):
                str_repr.append(f"np.array({val.tolist()})")
            else:
                str_repr.append(str(val))
            i += 1

        t = " ".join(str_repr)
        return t

    if isinstance(value, list):
        return _list_to_str(value)

    if isinstance(value, tuple):
        return _tuple_to_str(value)

    if isinstance(value, str):
        return _str_to_expression(value)

    if isinstance(value, Parameter):
        return str(value)

    if IS_JAXLITE:
        # ravel_pytree is unlikely to fail with the mocked up version of JAX because it
        # does not have proper support for actual jax types and pytrees.
        return str(value)

    try:
        # Test if is a Pytree
        value, _ = ravel_pytree(value)
        return _value_as_str(value)
    except BaseException:
        # Not a pytree
        pass

    return str(value)


class ParameterCache:
    """Global parameter value cache used by all :class:`Parameter` instances.

    Thread safety:
        All public methods are protected by a class-level reentrant lock
        (``threading.RLock``).  Using an ``RLock`` rather than a plain ``Lock``
        is necessary because ``__compute__`` may call ``param.get()`` recursively
        (for compound parameter expressions), which would deadlock under a
        non-reentrant lock held by the outer ``get()`` call.

        Concurrent simulations in separate threads sharing the same ``Parameter``
        objects are serialised correctly.  However, mutating a parameter from one
        thread while another thread is actively simulating with it is not
        recommended — the lock ensures the state remains consistent, but the
        simulation semantics of mid-run mutation are undefined.
    """

    __dependents__: dict["Parameter", set["Parameter"]] = {}
    __cache__: dict["Parameter", ArrayLike] = {}
    __is_dirty__ = defaultdict(lambda: True)
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def _register(cls, param: "Parameter") -> None:
        """Register a newly created Parameter in the cache (called from __post_init__)."""
        with cls._lock:
            if param not in cls.__dependents__:
                cls.__dependents__[param] = set()

    @classmethod
    def get(cls, param: "Parameter") -> ArrayLike:
        with cls._lock:
            if cls.__is_dirty__[param]:
                cls.__cache__[param] = cls.__compute__(param)
                cls.__is_dirty__[param] = False
            return cls.__cache__[param]

    @classmethod
    def replace(cls, param: "Parameter", value: ArrayLike):
        with cls._lock:
            param.value = value
            # Invalidate this parameter and propagate dirty flag recursively to all dependents.
            cls.__invalidate__(param)

    @classmethod
    def remove(cls, param: "Parameter"):
        with cls._lock:
            # Remove param from every set it appears in as a dependent.
            # Use list() to snapshot values so dict iteration is safe if anything
            # changes (e.g. via __del__ called on another thread simultaneously).
            for dependents in list(cls.__dependents__.values()):
                dependents.discard(param)  # discard is safe if param not present

            cls.__dependents__.pop(param, None)
            cls.__cache__.pop(param, None)
            if param in cls.__is_dirty__:
                del cls.__is_dirty__[param]

    @classmethod
    def add_dependent(cls, param: "Parameter", dependent: "Parameter"):
        # Mark 'dependent' as having a dependency on 'param', that is,
        # 'param' is built as an expression that involves 'dependent'.
        with cls._lock:
            cls._register(param)
            cls.__dependents__[param].add(dependent)

    @classmethod
    def get_dependents(cls, param: "Parameter"):
        with cls._lock:
            cls._register(param)
            return cls.__dependents__[param]

    @classmethod
    def print_dependents(cls, param: "Parameter", indent=0):
        """Prints the dependents tree of a parameter"""
        with cls._lock:
            cls._register(param)
            indent_str = "|" + "--" * indent if indent > 0 else ""
            print(indent_str + repr(param))
            for dependent in list(cls.__dependents__[param]):
                cls.print_dependents(dependent, indent + 1)

    @classmethod
    def static_dependents(cls, param: "Parameter"):
        with cls._lock:
            cls._register(param)
            dependents = set()
            for dependent in list(cls.__dependents__[param]):
                if dependent.is_static:
                    dependents.add(dependent)
                dependents |= cls.static_dependents(dependent)
            return dependents

    @classmethod
    def __invalidate__(cls, param: "Parameter"):
        # Caller MUST already hold cls._lock (called from replace() or recursively).
        cls.__cache__[param] = None
        cls.__is_dirty__[param] = True
        # Snapshot the dependent set to avoid mutation-during-iteration if another
        # thread somehow inserts a new dependent while we traverse (belt-and-suspenders).
        for dependent in list(cls.__dependents__.get(param, ())):
            cls.__invalidate__(dependent)

    @classmethod
    def __compute__(cls, param: "Parameter"):
        if isinstance(param.value, ParameterExpr):
            acc = None
            right_value = None
            op = None
            i = 0

            while i < len(param.value):
                val = param.value[i]

                if isinstance(val, Parameter):
                    right_value = val.get()
                elif isinstance(val, ArrayLikeTypes):
                    right_value = val
                elif isinstance(val, Ops):
                    if val in (Ops.NEG, Ops.POS, Ops.ABS):
                        if i + 1 >= len(param.value):
                            raise ParameterError(
                                param, message="Invalid parameter value"
                            )
                        if isinstance(param.value[i + 1], Parameter):
                            right_value = __OPS_FN__[val](param.value[i + 1].get())
                        elif isinstance(param.value[i + 1], ArrayLikeTypes):
                            right_value = __OPS_FN__[val](param.value[i + 1])
                        else:
                            raise ParameterError(
                                param,
                                message=f"Invalid value in parameter list: {param.value[i + 1]} of type {type(param.value[i + 1])}",
                            )
                        i += 1
                    else:
                        op = val
                else:
                    raise ParameterError(
                        param,
                        message=f"Invalid value in parameter list: {val} of type {type(val)}",
                    )

                if acc is not None and right_value is not None and op is not None:
                    acc = __OPS_FN__[op](acc, right_value)
                    op = None
                    right_value = None
                elif right_value is not None:
                    acc = right_value
                    right_value = None
                i += 1

            if acc is not None:
                return acc
            if right_value is not None:
                return right_value
            raise ParameterError(param, message="Invalid parameter value")

        if isinstance(param.value, Parameter):
            return cls.__compute__(param.value)

        if isinstance(param.value, tuple):
            t = _compute_list(param.value, is_tuple=True)
            return t

        if isinstance(param.value, list):
            t = _compute_list(param.value, is_tuple=False)
            return t

        if isinstance(param.value, dict):
            return {key: Parameter.unwrap(val) for key, val in param.value.items()}

        if isinstance(param.value, np.ndarray):
            vals = _resolve_array_param_value(param)
            return np.array(vals, dtype=param.value.dtype)

        if isinstance(param.value, Array):
            vals = _resolve_array_param_value(param)
            if param.value.weak_type:
                return jnp.array(vals)
            return jnp.array(vals, dtype=param.value.dtype)

        if isinstance(param.value, np.number):
            if isinstance(param.value.item(), Parameter):
                return type(param.value)(cls.__compute__(param.value.item()))
            return param.value

        if isinstance(param.value, str) and param.is_python_expr:
            # T-002: produce a useful diagnosis when the expression references
            # an undefined symbol or when py_namespace was never populated.
            scope = param.py_namespace
            try:
                _, resolved_parameters = resolve_parameters(param.value, scope)
                return eval(
                    param.value,
                    scope,
                    {**scope, **resolved_parameters},
                )
            except (TypeError, NameError, KeyError) as err:
                available = sorted(scope.keys()) if scope else "none"
                raise ValueError(
                    f"Parameter {param.name!r} expression {param.value!r} "
                    f"references undefined symbol(s). Available symbols: "
                    f"{available}"
                ) from err

        return param.value


def _op(op: Ops, left, right):
    param = Parameter(
        value=ParameterExpr([left, op, right]),
    )
    if isinstance(left, Parameter):
        ParameterCache.add_dependent(left, param)
    if isinstance(right, Parameter):
        ParameterCache.add_dependent(right, param)
    return param


def _record_parameter_creation(parameter):
    if not build_recorder.is_recording():
        return

    args = {}
    for field_info in dataclasses.fields(parameter):
        field_name = field_info.name
        field_value = getattr(parameter, field_name)
        default_value = field_info.default
        if field_name != "value" and field_value != default_value:
            args[field_name] = field_value
    args["value"] = _value_as_str(parameter.value)
    build_recorder.create_parameter(args)


@dataclasses.dataclass
class Parameter:
    value: Union[ParameterExpr, "Parameter", ArrayLike, str, tuple]

    # shape & dtype are set at init time when constructing the parameter,
    # they are not necessarily the actual value's shape and dtype
    dtype: DTypeLike = None
    shape: ShapeLike = None
    as_array: bool = False

    # name is used by reference submodels, model parameters and init script
    # variables so that they can be referred to in other fields
    # (we need this for serialization).
    name: str = None

    # For complex parameter values, we can specify a Python expression as string
    # This is useful for expressions like "np.eye(p)" where p is a parameter.
    is_python_expr: bool = False
    py_namespace: dict = None

    is_static: bool = False  # TODO: staticness should be propagated to dependents
    system: "SystemBase" = None

    # T-038a — opt-in per-parameter dtype hint.  When non-None and the value
    # is array-like (`np.ndarray` / `jax.Array`), the value is cast to this
    # dtype at construction time.  This is the metadata foundation for the
    # per-block dtype override mechanism; downstream block-side code may also
    # read ``_dtype_hint`` to allocate compatible buffers.  Default ``None``
    # is byte-equivalent to the pre-T-038a behavior.
    _dtype_hint: DTypeLike = None

    def get(self):
        value = ParameterCache.get(self)
        if self.as_array and not isinstance(value, Array):
            value = utils.make_array(value, self.dtype, self.shape)
        return value

    def set(self, value: Union["Parameter", ArrayLike, str, tuple]):
        ParameterCache.replace(self, value)

    @property
    def static_dependents(self):
        return ParameterCache.static_dependents(self)

    @property
    def is_dirty(self):
        return ParameterCache.__is_dirty__[self]

    @classmethod
    def unwrap(cls, value):
        """Get the underlying value of raw arrays and Parameter objects alike."""
        if value is None:
            return None
        if isinstance(value, (Array, bool, int, float, complex)):
            return value
        if isinstance(value, (np.ndarray, np.number)):
            if np.issubdtype(value.dtype, np.number):
                return value
            if value.shape == ():
                return Parameter.unwrap(value.item())
            return Parameter(value).get()
        if isinstance(value, Parameter):
            return value.get()
        if isinstance(value, list):
            return [cls.unwrap(val) for val in value]
        if isinstance(value, tuple):
            return tuple(cls.unwrap(val) for val in value)
        if isinstance(value, dict):
            return {key: cls.unwrap(val) for key, val in value.items()}
        # Fallback for unhandled types: forward to __compute__
        return Parameter(value).get()

    def __post_init__(self):
        ParameterCache._register(self)  # thread-safe registration

        # T-038a — apply optional dtype hint to array-like values.  Skipped
        # when the hint is None (the common path) so this remains a no-op
        # for every existing caller.  Only triggers on concrete array values
        # — ParameterExpr / strings / nested Parameters are left untouched
        # so the resolved value picks up its dtype downstream.
        if self._dtype_hint is not None and isinstance(
            self.value, (Array, np.ndarray)
        ):
            self.value = jnp.asarray(self.value, dtype=self._dtype_hint)

        if isinstance(self.value, Parameter):
            ParameterCache.add_dependent(self.value, self)
        if isinstance(self.value, ParameterExpr):
            for val in self.value:
                if isinstance(val, Parameter):
                    ParameterCache.add_dependent(val, self)
        if isinstance(self.value, (list, tuple)):
            _add_dependents(self.value, self)
        if self.is_python_expr and isinstance(self.value, str) and self.py_namespace:
            # A string expression like "k" or "np.eye(p)" depends on the
            # Parameter objects it names in its evaluation scope. Registering
            # them here (not only at deserialization time) means the links
            # survive deepcopy — __deepcopy__ re-runs __post_init__ — so
            # set() on a copied alias still invalidates copied referencing
            # blocks (T-141).
            for dep in _expr_parameter_refs(self.value, self.py_namespace):
                ParameterCache.add_dependent(dep, self)

        _record_parameter_creation(self)

    def __setstate__(self, state):
        self.__dict__.update(state)
        ParameterCache._register(self)
        if isinstance(self.value, Parameter):
            ParameterCache.add_dependent(self.value, self)
        if isinstance(self.value, ParameterExpr):
            for val in self.value:
                if isinstance(val, Parameter):
                    ParameterCache.add_dependent(val, self)
        if isinstance(self.value, (list, tuple)):
            _add_dependents(self.value, self)
        if self.is_python_expr and isinstance(self.value, str) and self.py_namespace:
            for dep in _expr_parameter_refs(self.value, self.py_namespace):
                ParameterCache.add_dependent(dep, self)

    def __deepcopy__(self, memo):
        """Copy fields and re-run post-init so :class:`ParameterCache` bookkeeping matches."""
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for field in dataclasses.fields(cls):
            value = getattr(self, field.name)
            if field.name == "py_namespace" and value is not None:
                # py_namespace is the evaluation scope for a string-valued
                # parameter expression: {**globals, **locals}, which includes
                # imported modules. Modules are un-deep-copyable singletons
                # (deepcopy raises "cannot pickle 'module' object"), so the
                # scope dict itself is rebuilt shallow — sharing module/global
                # references. Parameter entries are the exception: they must
                # go through the memo so a copied expression evaluates against
                # the *copied* aliases (the ones with_parameters mutates), not
                # the originals (T-141).
                setattr(
                    result,
                    field.name,
                    {
                        k: copy.deepcopy(v, memo) if isinstance(v, Parameter) else v
                        for k, v in value.items()
                    },
                )
            else:
                setattr(result, field.name, copy.deepcopy(value, memo))
        result.__post_init__()
        return result

    def __add__(self, other):
        return _op(Ops.ADD, self, other)

    def __radd__(self, other):
        return _op(Ops.ADD, other, self)

    def __sub__(self, other):
        return _op(Ops.SUB, self, other)

    def __rsub__(self, other):
        return _op(Ops.SUB, other, self)

    def __mul__(self, other):
        return _op(Ops.MUL, self, other)

    def __rmul__(self, other):
        return _op(Ops.MUL, other, self)

    def __truediv__(self, other):
        return _op(Ops.DIV, self, other)

    def __rtruediv__(self, other):
        return _op(Ops.DIV, other, self)

    def __floordiv__(self, other):
        return _op(Ops.FLOORDIV, self, other)

    def __rfloordiv__(self, other):
        return _op(Ops.FLOORDIV, other, self)

    def __mod__(self, other):
        return _op(Ops.MOD, self, other)

    def __rmod__(self, other):
        return _op(Ops.MOD, other, self)

    def __pow__(self, other):
        return _op(Ops.POW, self, other)

    def __rpow__(self, other):
        return _op(Ops.POW, other, self)

    def __neg__(self):
        p = Parameter(value=ParameterExpr([Ops.NEG, self]))
        ParameterCache.add_dependent(self, p)
        return p

    def __pos__(self):
        p = Parameter(value=ParameterExpr([Ops.POS, self]))
        ParameterCache.add_dependent(self, p)
        return p

    def __abs__(self):
        p = Parameter(value=ParameterExpr([Ops.ABS, self]))
        ParameterCache.add_dependent(self, p)
        return p

    def __eq__(self, other):
        return _op(Ops.EQ, self, other)

    def __ne__(self, other):
        return _op(Ops.NE, self, other)

    def __lt__(self, other):
        return _op(Ops.LT, self, other)

    def __le__(self, other):
        return _op(Ops.LE, self, other)

    def __gt__(self, other):
        return _op(Ops.GT, self, other)

    def __ge__(self, other):
        return _op(Ops.GE, self, other)

    def __del__(self):
        ParameterCache.remove(self)

    def __hash__(self):
        return id(self)

    def __str__(self):
        # Calling str() on a Parameter object is confusing. What's the intent?
        # 1. Serializing to a valid Python expression?
        # 2. Is it for logs? For debugging?
        # 3. Is it part of building a wider expression (like a list of parameters)?
        # 4. Evaluating the actual value of a string parameter?
        # Here, we support 2 & 4. We'll likely have to change this when we want support
        # for non-literal string parameters in the UI.

        expr, _ = self.value_as_api_param(
            allow_param_name=True,
            allow_string_literal=True,
        )
        return expr

    def __matmul__(self, other):
        return _op(Ops.MATMUL, self, other)

    def __int__(self):
        if self.dtype is not None:
            return self.dtype(self.get())
        return int(self.get())

    def __float__(self):
        if self.dtype is not None:
            return self.dtype(self.get())
        return float(self.get())

    # NOTE: __bool__ is intentionally not defined. Adding bool(Parameter) ->
    # bool(self.get()) broke some tests (it forces concretization of the
    # wrapped value); keep numeric coercions only.

    def __complex__(self):
        return complex(self.get())

    def value_as_api_param(
        self, allow_param_name=True, allow_string_literal=True
    ) -> tuple[str, bool]:
        """Returns an API-compatible expression[1] that defines this parameter

        What we return depends on the caller's context, since it depends on
        whether we are serializing for a model, submodel or block parameter.

        The boolean is the value of 'is_string' (means "string literal" or
        "do not call eval").

        [1] The returned string can be serialized to JSON, but it is not an
            already escaped JSON string!

        Args:
            allow_param_name: Set to false for (sub)model parameters. Optional.
                If true, and the value is defined by a name, just the name will
                be returned.
            allow_string_literal: Set to false for (sub)model parameters. Optional.
                If true, and the value is a string, then the string will be
                returned and 'is_string' will be returned as True.
        """
        if self.name is not None and allow_param_name:
            return self.name, False

        if self.is_python_expr and isinstance(self.value, str):
            return self.value, False

        if allow_string_literal and isinstance(self.value, str):
            return self.value, True

        return _value_as_str(self.value), False

    def __repr__(self):
        # This must return a valid python expression since it is used for
        # serialization to Python.
        ex, _ = self.value_as_api_param(allow_string_literal=False)
        if len(ex) > 100:
            ex = ex[:50] + "..." + ex[-50:]

        return (
            "Parameter("
            f"name={self.name}, value={ex}, "
            f"is_python_expr={self.is_python_expr}, "
            f"system={self.system.name if self.system is not None else None}"
            ")"
        )


def with_resolved_parameters(func):
    """Function wrapper to resolve Parameters from all arguments"""

    @wraps(func)
    def func_with_resolved_parameters(*args, **kwargs):
        args = [Parameter.unwrap(arg) for arg in args]
        kwargs = {k: Parameter.unwrap(v) for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        return result

    return func_with_resolved_parameters
