# SPDX-License-Identifier: MIT

"""Internal module to properly initialize logging and JAX/x64"""

import os

# Enable x64 by default, see also backend.py
# Note: this enables floats to default to 64-bit but not integers, so there are
# still issues on Windows where int defaults to 32bit but some other calculations
# will yield int64.
# Setting np.int_ = np.int64 globally is a big hack and does not fix it.
os.environ.setdefault("JAX_ENABLE_X64", "true")

# pylint: disable=wrong-import-position
from . import logging  # noqa: E402

_log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.set_log_level(_log_level)
logging.set_stream_handler()

_per_package_log_levels = os.environ.get("LOG_LEVELS", None)
if _per_package_log_levels is not None:
    _per_package_log_levels = _per_package_log_levels.split(",")
    _per_package_log_levels = [level.split(":") for level in _per_package_log_levels]
    for pkg, level in _per_package_log_levels:
        logging.set_log_level(level, pkg=pkg)

# Register custom VJP pickler/unpickler with copyreg to support cloudpickle serialization/deserialization across processes
try:
    import copyreg
    import jax
    import jax._src.custom_derivatives

    def _reconstruct_custom_vjp(fun, fwd, bwd, nondiff_argnums):
        if nondiff_argnums:
            f = jax.custom_vjp(fun, nondiff_argnums=nondiff_argnums)
        else:
            f = jax.custom_vjp(fun)
        if fwd is not None and bwd is not None:
            f.defvjp(fwd, bwd)
        return f

    def _reduce_custom_vjp(f):
        return (_reconstruct_custom_vjp, (f.fun, f.fwd, f.bwd, f.nondiff_argnums))

    copyreg.pickle(jax._src.custom_derivatives.custom_vjp, _reduce_custom_vjp)
except (ImportError, AttributeError):
    pass

