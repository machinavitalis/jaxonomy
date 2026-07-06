# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import uuid4

from jaxonomy.logging import logger
from jaxonomy.framework import Parameter


if TYPE_CHECKING:
    from jaxonomy.framework import Diagram


class ReferenceSubdiagramProtocol(Protocol):
    def __call__(
        self, *args: Any, instance_name: str, parameters: dict[str, Any], **kwargs: Any
    ) -> "Diagram": ...


class ReferenceSubdiagram:
    """Registry for reusable diagram templates ("reference subdiagrams").

    A reference subdiagram is a parameterized diagram factory.  It is registered
    once via :meth:`register` and can then be instantiated multiple times with
    different parameter values via :meth:`create_diagram`.

    Example::

        def my_submodel(instance_name, parameters):
            builder = DiagramBuilder()
            gain = parameters["gain"].get()
            ...
            return builder.build(instance_name)

        ref_id = ReferenceSubdiagram.register(
            my_submodel,
            default_parameters=[Parameter("gain", 1.0)],
        )
        diagram = ReferenceSubdiagram.create_diagram(ref_id, "my_instance")
    """

    _registry: dict[str, Callable[[Any], "Diagram"]] = {}
    _default_parameters: dict[str, list[Parameter]] = {}  # noqa: F821

    @classmethod
    def create_diagram(
        cls,
        ref_id: str,
        instance_name: str,
        *args,
        instance_parameters: dict[str, Any] = None,
        **kwargs,
    ) -> "Diagram":
        """Create a diagram based on the given reference ID and parameters.

        Note that for submodels we evaluate all parameters, there is no
        "pure" string parameters.

        Args:
            ref_id (str): The reference ID of the diagram.
            instance_name (str): Name for this specific instance.
            *args: Variable length arguments passed to the constructor.
            instance_parameters (dict[str, Any], optional): Per-instance parameter
                overrides.  Keys must match names declared at registration time.
                Example: ``{"gain": 3.0}``
            **kwargs: Keyword arguments passed to the constructor.

        Returns:
            Diagram: The created diagram.

        Raises:
            ValueError: If the reference subdiagram with the given ref_id is not found.
            ValueError: If an instance_parameter key does not match any registered
                parameter.
        """
        if ref_id not in ReferenceSubdiagram._registry:
            raise ValueError(f"ReferenceSubdiagram with ref_id {ref_id} not found.")

        params_def = ReferenceSubdiagram.get_default_parameters(ref_id)

        default_params = {p.name: p for p in params_def}

        # override the default values with any 'modified' values.
        new_instance_parameters = {}
        if instance_parameters:
            for param_name, param in instance_parameters.items():
                if param_name not in default_params:
                    raise ValueError(
                        f"Parameter {param_name} not found in parameter definitions."
                    )
                new_instance_parameters[param_name] = Parameter(
                    name=param_name, value=param
                )

        all_params = {**default_params, **new_instance_parameters}

        diagram = ReferenceSubdiagram._registry[ref_id](
            *args,
            instance_name=instance_name,
            parameters=all_params,
            **kwargs,
        )

        diagram.ref_id = ref_id
        diagram.instance_parameters = set(new_instance_parameters.keys())

        for param in params_def:
            if param.name in new_instance_parameters:
                diagram.declare_dynamic_parameter(
                    param.name, new_instance_parameters[param.name]
                )
            else:
                diagram.declare_dynamic_parameter(param.name, param)

        return diagram

    @staticmethod
    def register(
        constructor: ReferenceSubdiagramProtocol,
        default_parameters: list[Parameter] = None,  # noqa: F821
        ref_id: str | None = None,
        # Deprecated alias – use default_parameters instead
        parameter_definitions: list[Parameter] = None,  # noqa: F821
    ) -> str:
        """Register a diagram constructor as a reusable reference subdiagram.

        Args:
            constructor: A callable that builds a :class:`Diagram` given
                ``instance_name`` and ``parameters``.
            default_parameters: Default :class:`Parameter` values for this
                subdiagram.  Instances can override individual parameters at
                creation time via :meth:`create_diagram`.
            ref_id: Optional stable identifier.  A UUID is generated if omitted.
            parameter_definitions: **Deprecated** – use ``default_parameters``.

        Returns:
            str: The ``ref_id`` that can be passed to :meth:`create_diagram`.
        """
        import warnings

        if parameter_definitions is not None:
            warnings.warn(
                "The 'parameter_definitions' argument is deprecated; "
                "use 'default_parameters' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if default_parameters is None:
                default_parameters = parameter_definitions

        if ref_id is None:
            ref_id = str(uuid4())
        if default_parameters is None:
            default_parameters = []

        logger.debug("Registering ReferenceSubdiagram with ref_id %s", ref_id)
        if ref_id in ReferenceSubdiagram._registry:
            logger.debug(
                "ReferenceSubdiagram with ref_id %s already registered.",
                ref_id,
            )

        ReferenceSubdiagram._registry[ref_id] = constructor
        ReferenceSubdiagram._default_parameters[ref_id] = default_parameters

        return ref_id

    @staticmethod
    def get_default_parameters(
        ref_id: str,
    ) -> list[Parameter]:  # noqa: F821
        """Return the default parameters for the given reference subdiagram."""
        if ref_id not in ReferenceSubdiagram._default_parameters:
            return []
        return ReferenceSubdiagram._default_parameters[ref_id]

    @staticmethod
    def get_parameter_definitions(
        ref_id: str,
    ) -> list[Parameter]:  # noqa: F821
        """Return the default parameters for the given reference subdiagram.

        .. deprecated::
            Use :meth:`get_default_parameters` instead.
        """
        return ReferenceSubdiagram.get_default_parameters(ref_id)
