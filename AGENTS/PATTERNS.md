# Jaxonomy De Facto Patterns

Extracted from the live codebase. These are the conventions already in use — follow them when adding or modifying code.

---

## LeafSystem Subclass Skeleton

```python
from jaxonomy.framework import LeafSystem, parameters, DependencyTicket
from jaxonomy.backend import numpy_api as npa

class MyBlock(LeafSystem):
    @parameters(dynamic=["gain"], static=["n_out"])
    def __init__(self, gain, n_out=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)   # always pass name and **kwargs up

        # Declare ports/state — returns integer indices
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(None, name="out_0")
        self._ode_cb_idx = self.declare_continuous_state()
        self.n_out = n_out  # store statics needed by initialize()

    def initialize(self, gain, n_out=1, **kwargs):
        """Called by framework after __init__ with resolved parameter values."""
        def _output(time, state, *inputs, **params):
            return params["gain"] * inputs[0]

        self.configure_output_port(self._output_port_idx, _output,
                                   prerequisites_of_calc=[self.input_ports[0].ticket])
        self.configure_continuous_state(self._ode_cb_idx, ode=self._ode,
                                        default_value=npa.zeros(n_out))

    def _ode(self, time, state, u, **params):
        x = state.continuous_state
        return -params["gain"] * x + u
```

Rules:
- `**kwargs` must always be forwarded to `super().__init__()`.
- `name` is always a keyword arg defaulting to `None`.
- **Declare** (shape/index only) in `__init__`; **configure** (callbacks, values) in `initialize()`.
- Callbacks are private methods prefixed `_`.
- Store index return values as `self._*_idx` attributes.

---

## `@parameters` Decorator

Applied to `__init__` to auto-call `declare_dynamic_parameter` / `declare_static_parameter`:

```python
@parameters(dynamic=["kp", "ki", "kd"], static=["initial_state"])
def __init__(self, kp, ki, kd, initial_state=0.0, **kwargs):
    ...
```

Dynamic parameters are JAX-traceable (differentiable, vmappable). Static are not. Can also declare manually:

```python
self.declare_dynamic_parameter("alpha", alpha)
self.declare_static_parameter("edge_detection", edge_detection)
```

---

## Port Declaration

### Input ports

```python
self.declare_input_port()              # auto-named "in_0", "in_1", ...
self.u_in_index = self.declare_input_port("u")  # named
self.y_in_index = self.declare_input_port("y")
```

### Output ports

```python
# Feedthrough (depends on inputs):
self._port_idx = self.declare_output_port(
    self._eval_output,
    prerequisites_of_calc=[self.input_ports[0].ticket],
    requires_inputs=True,
)

# State-based (no feedthrough — important to set correctly):
self._port_idx = self.declare_output_port(
    self._output,
    prerequisites_of_calc=[DependencyTicket.xd],
    requires_inputs=False,
)

# Sample-and-hold at period dt:
self.declare_output_port(
    self._eval_output,
    period=dt,
    offset=0.0,
    default_value=npa.zeros(p),
    requires_inputs=self.is_feedthrough,
)

# Continuous state shortcut:
self.declare_continuous_state_output(name="x")

# Placeholder, configured in initialize():
self._port_idx = self.declare_output_port(None, name="out_0")
# then:
self.configure_output_port(self._port_idx, _func, prerequisites_of_calc=[...])
```

`requires_inputs=False` prevents false algebraic-loop detection and reduces compile time. Set it whenever the output only reads state.

### Continuous state

```python
# Declare only, configure in initialize():
self._ode_cb_idx = self.declare_continuous_state()
# Configure:
self.configure_continuous_state(self._ode_cb_idx, ode=self.ode, default_value=x0)

# Declare + configure in one shot:
self.declare_continuous_state(shape=(2,), ode=self.ode, dtype=npa.float64)
```

### Discrete state

One discrete state per `LeafSystem` (a second call overwrites):

```python
self.declare_discrete_state(default_value=x0)                    # array
self.declare_discrete_state(default_value=x0, dtype=npa.bool_)
self.declare_discrete_state(                                      # NamedTuple
    default_value=self.DiscreteStateType(prev_input=x0, output=False),
    as_array=False,
)
```

### Periodic update event

```python
# Two-step pattern:
self._update_idx = self.declare_periodic_update()
# In initialize():
self.configure_periodic_update(self._update_idx, callback=self._update,
                                period=self.dt, offset=0.0)

# One-shot:
self.declare_periodic_update(self._update, period=dt, offset=0.0)
```

---

## Callback Signatures

All user callbacks share this signature:

```python
def callback(time, state, *inputs, **params) -> result
```

- `time` — scalar float, simulation time
- `state` — `LeafState`; access via `state.continuous_state`, `state.discrete_state`, `state.mode`
- `*inputs` — positional unpacking of connected input port values (arrays)
- `**params` — dynamic parameter values keyed by declared name

### ODE (continuous dynamics)

```python
def _ode(self, time, state, u, **params):
    x = state.continuous_state
    A, B = params["A"], params["B"]
    return A @ x + B @ u

def _ode(self, time, state, **params):    # no inputs
    T = state.continuous_state[0]
    return -params["alpha"] * (T - params["T_amb"])
```

### Periodic update (returns new discrete state value)

```python
def _update(self, time, state, u, **params):
    x = state.discrete_state
    return params["A"] @ x + params["B"] @ u

def _update(self, time, state, *inputs):   # no dynamic params
    return state.discrete_state ** 3
```

### Output callback (returns port value)

```python
def _eval_output(self, time, state, *inputs, **params):
    return state.discrete_state

def _output(time, state, *inputs, **params):   # closure style
    return params["value"]
```

### Guard function (zero-crossing)

Returns a scalar float; transition fires when it crosses zero:

```python
def _guard_turn_on(self, time, state, **params):
    return state.continuous_state[0] - params["T_low"]
```

### Reset map

Returns an updated `LeafState`:

```python
def _reset_map(self, time, state, *inputs, **params):
    xc_new = state.continuous_state.at[1].set(-state.continuous_state[1])
    return state.with_continuous_state(xc_new)
```

Unused arguments are prefixed with `_` to suppress linting:

```python
def _output(self, _time, _state, *_inputs, **_params):
    ...
```

---

## NamedTuple Discrete State Conventions

Named-tuple states are defined as **inner classes** of the block:

```python
class EdgeDetection(LeafSystem):
    class DiscreteStateType(NamedTuple):
        prev_input: Array
        output: bool

class KalmanFilterBase(LeafSystem):
    class DiscreteStateType(NamedTuple):
        x_hat_minus: npa.ndarray
        P_hat_minus: npa.ndarray
        x_hat_plus: npa.ndarray
        P_hat_plus: npa.ndarray

class RandomNumber(LeafSystem):
    class RNGState(NamedTuple):
        key: Array
        val: Array
```

Declare with `as_array=False`; access by field name:

```python
self.declare_discrete_state(
    default_value=self.DiscreteStateType(prev_input=x0, output=False),
    as_array=False,
)
# In callbacks:
x = state.discrete_state.x_hat_plus
```

When the type must be created at runtime (dynamic field names):

```python
from collections import namedtuple
self.DiscreteStateType = namedtuple("DiscreteStateType", attribs)
```

---

## DiagramBuilder Patterns

```python
builder = jaxonomy.DiagramBuilder()

# Add systems — returns the system
sine  = builder.add(Sine(name="Sin_0"))
integ = builder.add(Integrator(x0, name="Integrator_0"))

# Connect output → input
builder.connect(sine.output_ports[0], integ.input_ports[0])

# Export diagram-level ports (for sub-diagrams)
builder.export_input(integ.input_ports[0], name="plant:input")
builder.export_output(integ.output_ports[0], name="x")

diagram = builder.build(name="root")
```

Port access: always `.input_ports[i]` / `.output_ports[i]`. Named lookup: `system.get_output_port("name")`. Sub-system access: `diagram["subsystem_name"]`.

### Nesting diagrams

```python
inner = inner_builder.build(name="inner")
outer_b = jaxonomy.DiagramBuilder()
outer_b.add(inner)
outer_b.connect(src.output_ports[0], inner.input_ports[0])
outer = outer_b.build(name="outer")
```

### Immutable parameter updates (for grad / vmap)

```python
updated = diagram.with_parameters({"gain.gain": jnp.array(2.0)})
updated = outer.with_parameters({"inner.gain.gain": jnp.array(5.0)})  # nested path
```

---

## `simulate()` API

```python
results = jaxonomy.simulate(
    system,
    context,
    t_span=(0.0, tf),
    options=jaxonomy.SimulatorOptions(
        rtol=1e-6,
        atol=1e-8,
        enable_tracing=True,
        enable_autodiff=False,
        max_major_steps=None,      # required when enable_autodiff=True
        ode_solver_method="auto",
        math_backend="jax",
    ),
    recorded_signals={
        "x": integ.output_ports[0],
        "v": vel.output_ports[0],
    },
)
```

### Results access

```python
results.time                             # Array of sample times
results.outputs["x"]                     # recorded signal by key
results.context.continuous_state         # final root state
results.context[block.system_id].continuous_state  # sub-system state
```

### Context creation

```python
ctx = system.create_context()
ctx = ctx.with_continuous_state(x0)
```

### Autodiff

```python
options = jaxonomy.SimulatorOptions(
    enable_autodiff=True,
    max_major_steps=100,   # must be explicit
    math_backend="jax",
)
n = jaxonomy.estimate_max_major_steps(system, (0.0, tf), safety_factor=2)
```

---

## JAX Patterns

### Backend abstraction

```python
from jaxonomy.backend import numpy_api as npa   # inside jaxonomy/ source
import jax.numpy as jnp                          # in tests, examples, callbacks
import numpy as np                                # static/structural code in __init__
```

Use `npa` inside library blocks so the backend stays switchable. Use `jnp` directly in traced callbacks. Use `np` for shapes, dtypes, and anything that must not be traced.

Switch the active backend with `set_backend("jax" | "numpy" | "torch")`, re-exported at `jaxonomy.set_backend` and also available as `from jaxonomy.backend import set_backend`. The setter mutates a context-variable-scoped dispatcher (DEC-030) — `conftest.py` resets it after each test so tests don't leak state.

### Traced code rules

```python
# Conditionals: jnp.where, never Python if/else inside traced functions
heat_rate = jnp.where(mode == HEAT, Q, 0.0)

# Immutable array updates:
xc_new = xc.at[1].set(-xc[1])

# JIT on utilities:
@partial(jax.jit, static_argnums=(0,))
def helper(static_arg, x): ...

# Grad / VJP:
grad = jax.grad(loss)(params)
out, vjp_fn = jax.vjp(f, x)

# Custom VJP for differentiating through events:
f = jax.custom_vjp(reset_map)
f.defvjp(fwd, bwd)

# Pytree registration for custom containers:
jax.tree_util.register_pytree_node(MyState, flatten_fn, unflatten_fn)
```

---

## Stochastic Blocks and `jax.vmap`

### The problem

`RandomNumber` and `WhiteNoise` store their PRNG key in discrete state. When you vmap over simulations all instances share the same key and produce identical noise.

### Solution 1: Different integer seeds (simplest)

```python
diagrams = [build_diagram(noise_seed=i) for i in range(n_batch)]
results  = [simulate(d, ...) for d in diagrams]
```

### Solution 2: `with_parameters` (JAX-native)

```python
base = build_diagram()
keys = jax.random.split(jax.random.PRNGKey(42), n_batch)
diagrams = [base.with_parameters({"noise.key": keys[i]}) for i in range(n_batch)]
```

### Solution 3: vmap with explicit key parameter

```python
def run_one(key):
    d = base.with_parameters({"noise.key": key})
    return simulate(d, ...)["y"][-1]

outputs = jax.vmap(run_one)(jax.random.split(jax.random.PRNGKey(0), n_batch))
# outputs.shape == (n_batch,)
```

### Canonical stochastic block implementation

```python
class MyStochasticBlock(LeafSystem):
    class RNGState(NamedTuple):
        key: jax.Array
        value: jax.Array

    def __init__(self, seed: int = 0, dt: float = 0.01, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_discrete_state(
            default_value=self.RNGState(
                key=jax.random.PRNGKey(seed),
                value=jnp.zeros(()),
            ),
            as_array=False,
        )
        self.declare_output_port(self._output, requires_inputs=False,
                                 prerequisites_of_calc=[DependencyTicket.xd])
        self.declare_periodic_update(self._update, period=dt, offset=0.0)

    def _update(self, time, state, *inputs, **params):
        key, subkey = jax.random.split(state.discrete_state.key)
        return self.RNGState(key=key, value=jax.random.normal(subkey))

    def _output(self, time, state, *inputs, **params):
        return state.discrete_state.value
```

Rules: store key in discrete state, always split before use, use subkey for sampling, never sample from key directly.

---

## Naming Conventions

| Thing | Convention | Examples |
|---|---|---|
| Block class | `PascalCase` | `LTISystem`, `EdgeDetection`, `KalmanFilterBase` |
| Block instance name | `ClassName_N` or descriptive | `"Integrator_0"`, `"gain"`, `"plant"` |
| Auto port names | `in_N`, `out_N` | `"in_0"`, `"out_1"` |
| Named port | `snake_case` | `"u"`, `"y"`, `"T"`, `"plant:input"` |
| Callback methods | `_verb` or `_verb_noun` | `_ode`, `_eval_output`, `_update`, `_guard_turn_on` |
| Index attributes | `_*_idx` | `_output_port_idx`, `_ode_cb_idx` |
| Inner state type | `DiscreteStateType`, or descriptive | `RNGState`, `RigidBodyState` |
| Parameter names | match math convention | `kp`, `ki`, `A`, `B`, `x_hat_0` |

---

## Import Conventions

### User / test code

```python
import jaxonomy
from jaxonomy.library import Integrator, Sine, Gain, LTISystem, PID
from jaxonomy.simulation import SimulatorOptions
import jax.numpy as jnp
import numpy as np
```

### Inside `jaxonomy/` source

```python
from ..framework import LeafSystem, parameters, DependencyTicket
from ..backend import numpy_api as npa, cond
from ..logging import logger
```

---

## Docstring and comment conventions

Default to writing no comments. When you do write one, prefer linking to a stable reference over a moving one:

- **Don't cite TODO / CHANGELOG / `notes/` / planning files.** They decay: entries get collapsed, renamed, or deleted, and the pointer rots silently. The information you wanted to preserve belongs inline, not behind a redirect.
- **Cite `AGENTS/DECISIONS.md` DEC-NNN** when the code embodies a non-obvious trade-off an ADR settled. ADRs are append-only and supersession-tracked — designed to be referenced.
- **Cite external standards / papers** (FMI 2.0 §4.2.4, Cao/Li/Petzold/Serban 2003, RFC numbers, etc.). They don't change inside this repo.
- **A bare `T-NNN` tag in a docstring is fine** for grep-coupling a test or block to its CHANGELOG line. Don't append a file path — the tag alone is enough; the path is the part that rots.
- **Code-path gotchas live at the site, not in `RULES.md`.** If the `why` is load-bearing, inline one to three lines. If not, drop it.

---

## Test Patterns

```python
pytestmark = pytest.mark.minimal   # near-universal module-level marker

class TestContinuousTime:
    def test_step_response(self):
        model = ScalarLinear(a=-1.0)
        model.input_ports[0].fix_value(1.0)   # fix disconnected input
        ctx = model.create_context()
        ctx = ctx.with_continuous_state(jnp.array(0.0))
        results = jaxonomy.simulate(model, ctx, (0.0, 5.0))
        xf = results.context.continuous_state
        assert jnp.allclose(xf, expected, rtol=1e-4, atol=1e-6)
```

- Minimal purpose-built `LeafSystem` subclasses defined at module level and shared across tests.
- `@pytest.mark.parametrize("enable_tracing", [True, False])` for tracing coverage.
- `conftest.py` resets the backend after each test; tests are otherwise self-contained.
- `from jaxonomy.testing.markers import skip_if_not_jax, requires_jax` for JAX-only tests.
