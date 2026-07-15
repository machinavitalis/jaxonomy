# SPDX-License-Identifier: MIT

"""T-130: DX API papercuts surfaced by tutorial authors."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import uq
from jaxonomy.library.lookup_table import interp_1d
from jaxonomy.uq.aleatoric_epistemic import decompose_variance_sobol

pytestmark = pytest.mark.minimal


class TestInterp1dModeAlias:
    XP = jnp.array([0.0, 1.0, 2.0, 3.0])
    FP = jnp.array([0.0, 1.0, 4.0, 9.0])

    def test_mode_is_alias_for_method(self):
        a = interp_1d(1.5, self.XP, self.FP, method="pchip")
        b = interp_1d(1.5, self.XP, self.FP, mode="pchip")
        assert float(a) == float(b)

    def test_both_method_and_mode_raises(self):
        with pytest.raises(TypeError, match="not both"):
            interp_1d(1.5, self.XP, self.FP, method="pchip", mode="akima")

    def test_default_still_linear(self):
        assert float(interp_1d(0.5, self.XP, self.FP)) == pytest.approx(0.5)


class TestUniformKindKeywordOnly:
    def test_positional_kind_rejected(self):
        with pytest.raises(TypeError):
            uq.Uniform(0.0, 1.0, "epistemic")

    def test_keyword_kind_accepted(self):
        assert uq.Uniform(0.0, 1.0, kind="epistemic").kind == "epistemic"
        assert uq.Uniform(0.0, 1.0).kind == "aleatoric"

    def test_normal_kind_keyword_only_too(self):
        with pytest.raises(TypeError):
            uq.Normal(0.0, 1.0, "epistemic")


class TestSobolQoiContract:
    def test_qoi_fn_error_attributed_at_boundary(self):
        """A qoi_fn that only knows one group's keys fails with the
        boundary message naming the full key set, not a bare KeyError."""

        def qoi_only_a(params):
            return params["a"] * params["missing_key"]

        with pytest.raises(ValueError, match="union of aleatoric and epistemic"):
            decompose_variance_sobol(
                qoi_only_a,
                aleatoric_dists={"a": uq.Uniform(0.0, 1.0)},
                epistemic_dists={"e": uq.Uniform(0.0, 1.0, kind="epistemic")},
                n_samples=8,
            )

    def test_wrong_shape_message_names_keys(self):
        def qoi_scalar(params):
            return jnp.sum(params["a"] + params["e"])  # scalar, not (4N,)

        with pytest.raises(ValueError, match="consuming every parameter"):
            decompose_variance_sobol(
                qoi_scalar,
                aleatoric_dists={"a": uq.Uniform(0.0, 1.0)},
                epistemic_dists={"e": uq.Uniform(0.0, 1.0, kind="epistemic")},
                n_samples=8,
            )


class TestDeclareDiscreteStateName:
    def test_name_kwarg_accepted_and_stored(self):
        class Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="counter")
                self.declare_discrete_state(
                    default_value=jnp.array(0.0), name="count"
                )

        blk = Counter()
        assert blk.discrete_state_name == "count"
        assert float(np.asarray(blk._default_discrete_state)) == 0.0

    def test_name_defaults_to_none(self):
        class Plain(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="plain")
                self.declare_discrete_state(default_value=jnp.array(1.0))

        assert Plain().discrete_state_name is None
