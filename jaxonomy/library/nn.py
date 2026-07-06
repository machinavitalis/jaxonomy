# SPDX-License-Identifier: MIT

"""Minimal example of a neural network block.

This uses equinox to start, but we should generalize to support other
JAX libraries as well as PyTorch, pending switchable backends.
"""

from typing import TYPE_CHECKING
import warnings
import numpy as np
import jax
from jax import random
import jax.numpy as jnp

from ..framework import parameters
from ..library import FeedthroughBlock
from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    import equinox as eqx
else:
    eqx = LazyLoader("eqx", globals(), "equinox")


class MLP(FeedthroughBlock):
    """
    A feedforward neural network block representing an Equinox multi-layer
    perceptron (MLP). The output `y` of the MLP is computed as

    ```
        y = MLP(x, theta)
    ```

    where `theta` are the parameters of the MLP, and `x` is the input to the MLP.
    This block is differentialble w.r.t. the MLP parameters `theta`. Note that `theta`,
    does not include the hyperparameters representing the architecture of the MLP.

    Input ports:
        (0) The input to the MLP.

    Output ports:
        (0) The output of the MLP.

    Parameters:
        in_size (int):
            The dimension of the input to the MLP.
        out_size (int):
            The dimension of the output of the MLP.
        width_size (int):
            The width of every hidden layers of the MLP.
        depth (int):
            The depth of the MLP. This represents the number of hidden layers,
            including the output layer.
        seed (int):
            The seed for the random number generator for initialization of the
            MLP parameters (weights and biases of every layer).
            If None, a random 32-bit seed will be generated.
        activation_str (str):
            The activation function to use after each internal layer of the MLP.
            Possible values are ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"elu"``,
            ``"swish"``, ``"gelu"``, ``"leaky_relu"``, ``"rbf"``, and
            ``"identity"``. Default is ``"relu"``.
        final_activation_str (str):
            The activation function to use for the output layer of the MLP.
            Same choices as ``activation_str``. Default is ``"identity"``.
        use_bias (bool):
            Whether to add a bias to the internal layers of the MLP.
            Default is True.
        use_final_bias (bool):
            Wheter to add a bias to the output layer of the MLP.
            Default is True.
        file_name (str):
            Optional file name containing the serialized parameters of the MLP.
            If provided, the parameters are loaded from the file, and set as the
            parameters of the MLP. Default is None.
    """

    @parameters(
        static=[
            "in_size",
            "out_size",
            "width_size",
            "depth",
            "seed",
            "activation_str",
            "final_activation_str",
            "use_bias",
            "use_final_bias",
            "file_name",
        ],
    )
    def __init__(
        self,
        in_size=None,
        out_size=None,
        width_size=None,
        depth=None,
        seed=None,
        activation_str="relu",
        final_activation_str="identity",
        use_bias=True,
        use_final_bias=True,
        file_name=None,
        **kwargs,
    ):
        """
        see https://docs.kidger.site/equinox/examples/serialisation/ for rationale
        of implementation here. We can't serialize the activation function, so we
        serialize a string representing a selection for activation function amongst
        a finite set of options.
        """
        super().__init__(None, **kwargs)
        # The Equinox MLP object is built in ``initialize()``, which runs when
        # ``create_context()`` is first called. Seed a sentinel so ``self.mlp``
        # access before then raises a clear error rather than AttributeError
        # (T-B4-followup-mlp-pre-context).
        self._mlp = None

    @property
    def mlp(self):
        """The underlying Equinox ``eqx.nn.MLP`` object.

        Built lazily in :meth:`initialize`, which runs the first time
        ``create_context()`` is called on a diagram containing this block.
        Accessing it before then raises a clear error pointing at
        ``create_context()`` (T-B4-followup-mlp-pre-context).
        """
        if self._mlp is None:
            raise AttributeError(
                f"MLP block {self.name!r}: the underlying Equinox network is "
                f"not built until the block is initialized. Call "
                f"`diagram.create_context()` (or `block.create_context()` for a "
                f"standalone block) first, then access `.mlp`. The architecture "
                f"hyperparameters (in_size / out_size / width_size / depth) are "
                f"available immediately; only the parameterised network object "
                f"is deferred."
            )
        return self._mlp

    def initialize(
        self,
        in_size=None,
        out_size=None,
        width_size=None,
        depth=None,
        seed=None,
        activation_str="relu",
        final_activation_str="identity",
        use_bias=True,
        use_final_bias=True,
        file_name=None,
        mlp_params=None,
    ):
        # mlp_params is stored as a dynamic parameter.  The guard below
        # (`if "mlp_params" in self.dynamic_parameters`) ensures it is updated
        # rather than overwritten on subsequent calls (e.g. after optimization),
        # so optimized weights survive re-initialization of the block.

        if in_size is None or out_size is None or width_size is None or depth is None:
            raise ValueError("Must specify in_size, out_size, width_size, and depth.")
        else:
            # Cast to int for safety
            in_size = int(in_size)
            out_size = int(out_size)
            width_size = int(width_size)
            depth = int(depth)

        # file_name may come as an empty string through json parsing
        if file_name == "":
            file_name = None

        # Mapping from activation string to callable.
        # Extend here to add new activations; see https://jax.readthedocs.io/en/latest/jax.nn.html
        def _match_activation(activation_str):
            activation_mapping = {
                "relu": jax.nn.relu,
                "sigmoid": jax.nn.sigmoid,
                "tanh": jnp.tanh,
                "elu": jax.nn.elu,
                "swish": jax.nn.silu,
                "gelu": jax.nn.gelu,
                "leaky_relu": jax.nn.leaky_relu,
                "rbf": lambda x: jnp.exp(-(x**2)),
                "identity": lambda x: x,
            }
            if activation_str not in activation_mapping:
                warnings.warn(
                    f"Provided activation function {activation_str} not recognized. "
                    "Using Identity function as activation."
                )
            return activation_mapping.get(activation_str, lambda x: x)

        seed = (
            np.random.randint(0, 2**32, dtype=np.int64) if seed is None else int(seed)
        )
        self.key = random.PRNGKey(seed)

        self._mlp = eqx.nn.MLP(
            in_size,
            out_size,
            width_size,
            depth,
            key=self.key,
            activation=_match_activation(activation_str),
            final_activation=_match_activation(final_activation_str),
            use_bias=use_bias,
            use_final_bias=use_final_bias,
        )

        if file_name is not None:
            with open(file_name, "rb") as fp:
                self._mlp = eqx.tree_deserialise_leaves(fp, self._mlp)

        # partition into a pytree of params and static components
        mlp_params, self.mlp_static = eqx.partition(self._mlp, eqx.is_array)

        if "mlp_params" in self.dynamic_parameters:
            self.dynamic_parameters["mlp_params"].set(mlp_params)
        else:
            self.declare_dynamic_parameter("mlp_params", mlp_params, as_array=False)

        def _eval_MLP(inputs, **parameters):
            mlp_params = parameters["mlp_params"]
            mlp = eqx.combine(mlp_params, self.mlp_static)
            return mlp(inputs)

        self.replace_op(_eval_MLP)

    def serialize(self, file_name, mlp_params=None):
        """
        Serialize only the parameters of the MLP. Note that the hyperparameters
        representing the architecture of the MLP are not serialized. This is because
        of the following use-cases imagined:
        (i) The user may train the Equinox MLP outside of Jaxonomy. In this case,
        it seems unnecessary to force the user to serialize the hyperparameters of the
        MLP in the strict form chosen by Jaxonomy. It would seem much easier
        for the user to just input these hyperparameters when creating the MLP block
        in Jaxonomy UI, and upload the naturally produced serialized parameters file
        by Equinox.
        (ii) The user may want to train the Equinox MLP within Jaxonomy in a notebook,
        and then use the block within Colimator UI. In this case, while serialization of
        the hyperparameters of the MLP would be a litte more convenient compared
        to manually inputting the hyperparameters in the UI, it seems like a small
        convenience relative to disadvantages of (i). Ideally the user should be
        able to use the API to push the learnt parameters.
        (iii) When we support training in the UI, the hyperparameters are naturally
        serialzed with `declare_configuraton_parameters`, and thus, in this case too,
        only serializatio of the MLP parameters is necessary.

        The choice of an optional `mlp_params` is to enable training of the
        models in a notebook and easily seralizing them for use in the UI.
        """

        if mlp_params is None:
            mlp = self._mlp
        else:
            mlp = eqx.combine(mlp_params, self.mlp_static)
        with open(file_name, "wb") as f:
            eqx.tree_serialise_leaves(f, mlp)
