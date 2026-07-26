# SPDX-License-Identifier: MIT

"""Lookup-table parameter validation must accept JAX tracers.

``ComponentBase._val_lut_param_type`` originally checked
``isinstance(p, (ArrayLike, List))``, which rejected ``DynamicJaxprTracer``
— so any lookup-table component (``BatteryCellECM``, ``IntegratedMotor``,
tabulated resistors) constructed inside ``jax.jit`` / ``jax.vmap`` raised
``AcausalModelError``. The check is now duck-typed (sequences, or anything
exposing ``.shape`` — numpy arrays, JAX arrays, and tracers alike) while
still rejecting scalars and strings.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.acausal import EqnEnv
from jaxonomy.acausal import battery as bat
from jaxonomy.acausal import electrical as elec
from jaxonomy.acausal.error import AcausalModelError
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _some_component():
    return elec.Resistor(EqnEnv(), name="r")


@pytest.mark.parametrize(
    "good",
    [
        [0.0, 1.0],
        (0.0, 1.0),
        np.linspace(0.0, 1.0, 5),
        jnp.linspace(0.0, 1.0, 5),
    ],
)
def test_validator_accepts_concrete_array_likes(good):
    _some_component()._val_lut_param_type("lut", good, "xp")


@pytest.mark.parametrize("bad", [1.0, 3, "0,1", None])
def test_validator_rejects_non_arrays(bad):
    with pytest.raises(AcausalModelError, match="must be an array or list"):
        _some_component()._val_lut_param_type("lut", bad, "xp")


def test_validator_accepts_tracers():
    comp = _some_component()

    @jax.jit
    def f(x):
        comp._val_lut_param_type("lut", x, "xp")  # must not raise on a tracer
        return jnp.sum(x)

    f(jnp.linspace(0.0, 1.0, 5))


def test_lut_component_constructs_under_trace():
    # The reported repro shape: a lookup-table component built inside a JAX
    # transformation, with the tabulated values as traced parameters.
    def build(ocv_volts):
        ev = EqnEnv()
        bat.BatteryCellECM(
            ev,
            name="cell",
            ocv_soc=jnp.linspace(0.0, 1.0, 5),
            ocv_volts=ocv_volts,
        )
        return jnp.sum(ocv_volts)

    # eval_shape traces `build` with abstract tracers exactly like jit/vmap.
    jax.eval_shape(build, jnp.linspace(3.0, 4.2, 5))
