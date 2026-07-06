"""Cloud-offloaded batch simulation — placeholder.

No remote execution backend is bundled with this build, so
:func:`simulate_cloud` is a stub that raises :class:`NotImplementedError`.
The full implementation will return once the backend is available.
"""
from __future__ import annotations

from typing import Any


def simulate_cloud(*args: Any, **kwargs: Any):
    """Run a batch of simulations on a remote execution backend.

    Not available in this build: no cloud execution backend is bundled.
    Use the local :func:`jaxonomy.simulate` / batch / distributed runners
    instead. This entry point is reserved and will be implemented, and
    documented, once the backend ships.

    Raises:
        NotImplementedError: always, in this build.
    """
    raise NotImplementedError(
        "simulate_cloud() is not available in this build: no cloud execution "
        "backend is bundled. Use the local simulate()/batch runners instead."
    )
