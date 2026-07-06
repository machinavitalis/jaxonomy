# SPDX-License-Identifier: MIT

import json
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from jaxonomy.mcp._helpers import parse_csv_string, resolve_output_port
from jaxonomy.mcp.server import mcp


def _numpy_to_json_list(x):
    try:
        return np.asarray(jax.device_get(x)).tolist()
    except Exception:
        return np.asarray(x).tolist()


@mcp.tool()
def run_simulation(
    model_json: str,
    t_start: float,
    t_stop: float,
    recorded_signals: list,
    backend: str = "jax",
    max_major_steps: int = 500,
) -> str:
    """
    Run a Jaxonomy simulation.

    Args:
        model_json: JSON string of the model
        t_start: simulation start time (seconds)
        t_stop: simulation end time (seconds)
        recorded_signals: list of signal names to record.
            Use the block name and port, e.g.
            ["integrator.out_0", "gain.out_0"]
        backend: "jax" (fast, GPU) or "numpy" (compatible)
        max_major_steps: maximum simulation steps
                         (increase for longer simulations)

    Returns JSON with:
        time: list of time points
        signals: {name: list of values}
        final_time: actual simulation end time
    """
    try:
        from jaxonomy import simulate
        from jaxonomy.dashboard.serialization.from_model_json import load_model
        from jaxonomy.simulation import SimulatorOptions

        model_dict = json.loads(model_json)
        sim_context = load_model(model_dict)
        diagram = sim_context.diagram

        rec: dict = {}
        for spec in recorded_signals:
            if not isinstance(spec, str):
                spec = str(spec)
            port = resolve_output_port(diagram, spec)
            rec[spec] = port

        ctx = diagram.create_context()
        options = SimulatorOptions(
            math_backend=backend,
            max_major_steps=max_major_steps,
        )
        results = simulate(
            diagram,
            ctx,
            (float(t_start), float(t_stop)),
            options=options,
            recorded_signals=rec,
        )
        if results.outputs is None:
            return json.dumps({"error": "simulate returned no outputs"})

        signals_out = {}
        for name in recorded_signals:
            k = str(name)
            if k in results.outputs:
                signals_out[k] = _numpy_to_json_list(results.outputs[k])

        return json.dumps(
            {
                "time": _numpy_to_json_list(results.time),
                "signals": signals_out,
                "final_time": float(np.asarray(jax.device_get(results.time[-1]))),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def fit_parameters(
    model_json: str,
    data_csv: str,
    signal_map: str,
    params_to_fit: list,
    n_steps: int = 300,
    learning_rate: float = 1e-3,
    method: str = "adam",
    bounds: list | None = None,
) -> str:
    """
    Fit model parameters to measured data using finite-difference gradients.

    Args:
        model_json: JSON string of the base model
        data_csv: CSV string with columns for time and
                  measured signals. First column must be time.
                  Example: "t,position,velocity\\n0.0,0.0,1.0\\n..."
        signal_map: JSON string mapping CSV column names to
                    diagram signal names.
                    Example: '{"position": "integ.out_0"}'
        params_to_fit: list of dot-notation parameter paths
                       to optimize.
                       Example: ["motor.R", "motor.L"]
        n_steps: number of optimizer iterations (for scipy methods,
                 used as maxiter)
        learning_rate: optimizer learning rate (optax methods only)
        method: optimizer to use. One of:
                  "adam" (default), "sgd", "rmsprop" — Optax gradient
                  descent with central finite-difference gradients;
                  "l_bfgs_b" — scipy L-BFGS-B with FD gradients;
                  "nelder_mead" — scipy Nelder-Mead, gradient-free.
        bounds: list of [lo, hi] per parameter (same order as
                params_to_fit), or null for unbounded.
                Use null on either side for a one-sided bound.
                Example: [[0.0, 10.0], [null, 5.0], [0.0, null]]

    Returns JSON with:
        fitted_model_json: updated model with fitted params
        final_loss: final MSE loss value
        loss_history: loss values over iterations (sampled)
        fitted_params: {param_name: fitted_value}
        converged: bool (loss decreased by > 10x)
        n_iter: actual number of optimizer iterations completed
        n_fev: total number of loss function evaluations (approximate)
        convergence_message: human-readable convergence status
        relative_decrease: (init_loss - final_loss) / init_loss

    Note:
        Gradients are computed with central finite differences on the
        simulation loss so the full simulator need not be JAX-traced.
        For high-dimensional fits, prefer the dedicated optimization APIs
        in ``jaxonomy.optimization``.
    """
    try:
        from jaxonomy import simulate
        from jaxonomy.dashboard.serialization.from_model_json import load_model
        from jaxonomy.dashboard.serialization.to_model_json import convert
        from jaxonomy.framework.parameter import Parameter
        from jaxonomy.simulation import SimulatorOptions, estimate_max_major_steps

        # --- validate method ---
        _OPTAX_METHODS = {"adam", "sgd", "rmsprop"}
        _SCIPY_METHODS = {"l_bfgs_b", "nelder_mead"}
        method_lower = method.lower()
        if method_lower not in _OPTAX_METHODS | _SCIPY_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Supported: {sorted(_OPTAX_METHODS | _SCIPY_METHODS)}"
            )

        model_dict = json.loads(model_json)
        sim_context = load_model(model_dict)
        diagram = sim_context.diagram

        smap = json.loads(signal_map)
        if not smap:
            raise ValueError("signal_map must map at least one CSV column to a signal")

        header, data = parse_csv_string(data_csv)
        t_data = data[:, 0]
        col_index = {h: i for i, h in enumerate(header)}

        rec: dict = {}
        for csv_col, sig_spec in smap.items():
            if csv_col not in col_index:
                raise ValueError(f"CSV column {csv_col!r} not in header {header}")
            rec[str(sig_spec)] = resolve_output_port(diagram, str(sig_spec))

        plist = list(params_to_fit)
        flat_params = []
        for path in plist:
            pdict = diagram.list_parameters()
            if path not in pdict:
                raise KeyError(f"Parameter path {path!r} not found on diagram")
            v = pdict[path]
            flat_params.append(float(Parameter.unwrap(v)))

        # --- parse bounds ---
        # parsed_bounds: list of (lo_or_None, hi_or_None), length == len(plist)
        if bounds is not None:
            if len(bounds) != len(plist):
                raise ValueError(
                    f"bounds has {len(bounds)} entries but params_to_fit has "
                    f"{len(plist)} entries; they must match."
                )
            parsed_bounds = []
            for b in bounds:
                if b is None or b == [None, None]:
                    parsed_bounds.append((None, None))
                else:
                    lo = b[0] if b[0] is not None else None
                    hi = b[1] if b[1] is not None else None
                    parsed_bounds.append((lo, hi))
        else:
            parsed_bounds = None

        t0 = float(t_data[0])
        t1 = float(t_data[-1])
        # estimate_max_major_steps inspects the diagram's periodic events, which
        # requires the (recursively-built) dependency graphs of every subsystem.
        # Those are populated as a side effect of context creation, so build a
        # context first — every other estimate_max_major_steps caller receives an
        # already-created base_context. Without this the callback trackers deref a
        # None dependency_graph ("'NoneType' object is not subscriptable").
        diagram.create_context()
        max_major_steps = estimate_max_major_steps(diagram, (t0, t1))

        def build_updates(vec_np: np.ndarray) -> dict:
            return {path: float(vec_np[i]) for i, path in enumerate(plist)}

        y_meas = {
            csv_col: data[:, col_index[csv_col]] for csv_col in smap.keys()
        }

        options = SimulatorOptions(math_backend="jax", max_major_steps=max_major_steps)

        def loss_fn_np(vec_np: np.ndarray) -> float:
            d = diagram.with_parameters(build_updates(vec_np))
            ctx = d.create_context()
            # with_parameters deep-copies the diagram, and SystemBase.__deepcopy__
            # assigns fresh system_ids to the copy. The ports in `rec` were
            # resolved against the original `diagram`, so they carry stale
            # system_ids that don't exist in `d`'s context. Re-resolve each
            # recorded signal against the copy actually being simulated.
            rec_d = {name: resolve_output_port(d, name) for name in rec}
            res = simulate(
                d,
                ctx,
                (t0, t1),
                options=options,
                recorded_signals=rec_d,
            )
            if res.outputs is None:
                return float("inf")
            total = 0.0
            for csv_col, sig_spec in smap.items():
                y_t = y_meas[csv_col]
                y_sim = res.outputs[str(sig_spec)]
                y_s = jnp.interp(t_data, res.time, y_sim)
                total += float(jnp.mean((y_s - y_t) ** 2))
            return total / len(smap)

        def grad_fd(vec_np: np.ndarray, eps: float = 1e-5) -> np.ndarray:
            g = np.zeros_like(vec_np, dtype=np.float64)
            for j in range(len(vec_np)):
                v = vec_np.copy()
                v[j] += eps
                lp = loss_fn_np(v)
                v[j] -= 2 * eps
                lm = loss_fn_np(v)
                g[j] = (lp - lm) / (2 * eps)
            return g

        if len(flat_params) > 50:
            raise ValueError(
                f"fit_parameters received {len(flat_params)} parameters to fit, "
                "which exceeds the hard cap of 50. Fitting too many parameters "
                "simultaneously is likely to produce poor results and will be very slow. "
                "Consider fitting fewer parameters at a time or using a dedicated "
                "optimization library."
            )

        n_sims = 2 * len(flat_params) * n_steps
        if n_sims > 500:
            import warnings
            warnings.warn(
                f"fit_parameters will run approximately {n_sims} simulations "
                f"({2 * len(flat_params)} per step × {n_steps} steps). "
                f"This may be slow for large parameter counts. "
                f"Consider reducing n_steps or the number of parameters.",
                UserWarning,
                stacklevel=2,
            )

        vec_np = np.array(flat_params, dtype=np.float64)
        loss_hist = []
        init_loss = None
        final_loss = None
        n_iter = 0
        n_fev = 0
        convergence_message = ""

        if method_lower in _OPTAX_METHODS:
            import optax

            theta = jnp.array(vec_np)
            if method_lower == "adam":
                optimizer = optax.adam(learning_rate)
            elif method_lower == "sgd":
                optimizer = optax.sgd(learning_rate)
            else:  # rmsprop
                optimizer = optax.rmsprop(learning_rate)
            opt_state = optimizer.init(theta)

            sample_every = max(1, n_steps // 20)

            for i in range(n_steps):
                g_np = grad_fd(vec_np)
                n_fev += 2 * len(vec_np)
                g = jnp.array(g_np)
                theta_j = jnp.array(vec_np)
                updates, opt_state = optimizer.update(g, opt_state, theta_j)
                theta_j = optax.apply_updates(theta_j, updates)
                vec_np = np.array(jax.device_get(theta_j), dtype=np.float64)

                # project to bounds after each step
                if parsed_bounds is not None:
                    for j, (lo, hi) in enumerate(parsed_bounds):
                        if lo is not None:
                            vec_np[j] = max(vec_np[j], lo)
                        if hi is not None:
                            vec_np[j] = min(vec_np[j], hi)

                lv = loss_fn_np(vec_np)
                n_fev += 1
                final_loss = lv
                if init_loss is None:
                    init_loss = lv
                if i % sample_every == 0:
                    loss_hist.append(lv)

            n_iter = n_steps

            if init_loss is None or init_loss == 0.0:
                convergence_message = "Initial loss is zero"
            elif final_loss is not None and final_loss < init_loss / 10.0:
                convergence_message = "Converged (loss decreased >10x)"
            else:
                convergence_message = (
                    f"Loss did not decrease >10x after {n_steps} steps"
                )

        else:
            # scipy path
            import scipy.optimize as sciopt

            scipy_method_name = "L-BFGS-B" if method_lower == "l_bfgs_b" else "Nelder-Mead"

            scipy_bounds = [
                (lo, hi)
                for (lo, hi) in (parsed_bounds or [(None, None)] * len(plist))
            ]

            sample_every = max(1, n_steps // 20)
            _call_count = [0]

            def loss_fn_tracked(v):
                _call_count[0] += 1
                lv = loss_fn_np(v)
                # record loss history at roughly sample_every intervals
                if _call_count[0] % sample_every == 0:
                    loss_hist.append(lv)
                return lv

            # evaluate initial loss before handing off to scipy
            init_loss = loss_fn_np(vec_np)
            _call_count[0] += 1

            opt_res = sciopt.minimize(
                loss_fn_tracked,
                vec_np,
                method=scipy_method_name,
                bounds=scipy_bounds if method_lower == "l_bfgs_b" else None,
                options={"maxiter": n_steps},
            )

            vec_np = np.array(opt_res.x, dtype=np.float64)
            final_loss = float(opt_res.fun)
            n_iter = int(opt_res.nit)
            # nfev from scipy already counts calls to loss_fn_tracked;
            # for L-BFGS-B, scipy does its own internal FD for the jacobian,
            # so we add our tracked calls and the initial evaluation.
            n_fev = int(opt_res.nfev) + 1  # +1 for init_loss evaluation
            convergence_message = str(opt_res.message)

            if not loss_hist:
                loss_hist.append(final_loss)

        fitted = build_updates(vec_np)
        d_fit = diagram.with_parameters(fitted)
        model_out, _ = convert(d_fit)
        fitted_json = json.dumps(model_out.to_dict(), indent=2)

        converged = bool(
            init_loss is not None
            and final_loss is not None
            and init_loss > 0
            and final_loss < init_loss / 10.0
        )

        relative_decrease = (
            float((init_loss - final_loss) / init_loss)
            if (init_loss is not None and init_loss > 0 and final_loss is not None)
            else 0.0
        )

        return json.dumps(
            {
                "fitted_model_json": fitted_json,
                "final_loss": final_loss,
                "loss_history": loss_hist,
                "fitted_params": fitted,
                "converged": converged,
                "n_iter": n_iter,
                "n_fev": n_fev,
                "convergence_message": convergence_message,
                "relative_decrease": relative_decrease,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})