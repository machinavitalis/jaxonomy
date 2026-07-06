# SPDX-License-Identifier: MIT

"""Persistent JIT compilation cache helper.

Wraps JAX's persistent compilation cache configuration so users can opt in
with one call (``jaxonomy.enable_persistent_jit_cache()``).  The first run
still pays the JIT compile cost; subsequent runs with the same JAXPR +
device + JAX version read pre-compiled artefacts from the cache directory.
See ``docs/jit_cache.md`` for warmup behaviour and tunable details.
"""

from __future__ import annotations

from pathlib import Path

import jax

from .logging import logger


_DEFAULT_CACHE_DIR = "~/.cache/jaxonomy/jit/"


def enable_persistent_jit_cache(cache_dir: str | None = None) -> None:
    """Opt in to JAX's persistent compilation cache.

    Parameters
    ----------
    cache_dir:
        Directory used for the on-disk cache.  Defaults to
        ``~/.cache/jaxonomy/jit/``.  Created (with parents) if missing.
    """
    target = Path(cache_dir) if cache_dir else Path(_DEFAULT_CACHE_DIR)
    target = target.expanduser()
    target.mkdir(parents=True, exist_ok=True)

    resolved = str(target)
    jax.config.update("jax_compilation_cache_dir", resolved)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    logger.info("jaxonomy: persistent JIT cache enabled at %s", resolved)


__all__ = ["enable_persistent_jit_cache"]
