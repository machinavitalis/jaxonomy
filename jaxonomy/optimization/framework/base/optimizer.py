# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationResult:
    """
    Unified result returned by all jaxonomy optimizers.

    Supports dict-like access (``result["param"]``) for backward compatibility
    with code that treated the old return value as a plain parameter dict.

    Attributes:
        params: dict mapping parameter name → optimized value (same as the
            dict that optimizers used to return directly).
        success: ``True`` if the optimizer reported convergence.
        nit: Number of iterations (or epochs / generations).
        nfev: Number of objective-function evaluations.
        message: Human-readable status message from the optimizer.
        final_loss: Objective value at the optimum.  ``None`` when not
            available (e.g. population-based methods that track fitness
            separately).
        loss_history: Sequence of objective values recorded during
            optimization (one per epoch / generation).
    """

    params: dict[str, Any]
    success: bool = True
    nit: int = 0
    nfev: int = 0
    message: str = ""
    final_loss: float | None = None
    loss_history: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Backward-compatible dict-like interface
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.params[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.params

    def __iter__(self):
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def items(self):
        return self.params.items()

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def __repr__(self) -> str:
        return (
            f"OptimizationResult("
            f"params={self.params}, "
            f"success={self.success}, "
            f"nit={self.nit}, "
            f"nfev={self.nfev}, "
            f"final_loss={self.final_loss}, "
            f"message={self.message!r}"
            f")"
        )


class Optimizer(ABC):
    """Base class that all optimizers should inherit from."""

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        pass

    @property
    def metrics(self):
        return {}
