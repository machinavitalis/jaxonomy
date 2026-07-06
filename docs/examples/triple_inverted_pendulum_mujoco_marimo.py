# Triple inverted pendulum on a cart: LQR with Jaxonomy + MuJoCo visualization
# Run: marimo edit docs/examples/triple_inverted_pendulum_mujoco_marimo.py
# Deps: marimo, jaxonomy, jax, mujoco, matplotlib, pillow, numpy (no ffmpeg, no python-control)
#
# Editor note: uses `from mujoco import MjModel, MjData, ...` for type checkers; `__paths` avoids
# bare `__file__` (undefined in some marimo runs); plot/video/sim cells use `_`-prefixed locals
# so marimo does not report duplicate variable names across cells.

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    from __future__ import annotations

    import pathlib
    from typing import Tuple

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import mujoco
    from mujoco import MjData, MjModel, MjvOption, Renderer, mjtVisFlag
    import numpy as np
    from jaxonomy import library, simulate, SimulatorOptions
    from jaxonomy.framework import DiagramBuilder

    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image


@app.cell
def __mo():
    import marimo as mo

    return (mo,)


@app.cell
def __intro(mo):
    mo.md(r"""
    # Triple inverted pendulum: Jaxonomy LQR + MuJoCo kick-up and stabilization

    This tutorial designs an **LQR** gain $K$ with **Jaxonomy's
    `library.LinearQuadraticRegulator`** (a continuous-time algebraic-Riccati solve on the
    MuJoCo-linearized plant), then validates it against **MuJoCo** physics on a triple pendulum
    on a cart. Jaxonomy is the control-design engine; MuJoCo `mj_step` is the high-fidelity
    reference. We simulate in two phases:

    1. **Kick-up (open loop)**: from the **upright** equilibrium, apply a short **cosine burst** on the cart
       force, $u(t)=A\cos(2\pi f t)$, to excite the chain (impulse-like start because $u(0)=A$).
    2. **Stabilization (closed loop)**: after a fixed time, switch to **$u=-\mathrm{clip}(Kx)$** using the
       same $K$ as in the Jaxonomy diagram. $Q$ and $R$ are chosen so the gains are **not** immediately
       saturated (large $R$), which is critical for `mj_step` to track the continuous-time LQR design.

    Visualization uses **MuJoCo forward kinematics** (`mj_forward` + body positions) and, when OpenGL is
    available, an optional **MuJoCo `Renderer`** pass. Animations are **GIF via Pillow** — **no ffmpeg**.

    Reference vibe: [cart–pendulum style demos on YouTube](https://www.youtube.com/watch?v=Rh7JuL3PRSY).
    """)
    return


@app.cell
def __paths():
    # Marimo often runs cells without __file__; fall back to cwd (run from repo / docs/examples).
    _here = globals().get("__file__")
    HERE = pathlib.Path(_here).resolve().parent if _here else pathlib.Path.cwd()
    xml_path = HERE / "mujoco" / "assets" / "triple_pendulum_cart.xml"
    assert xml_path.is_file(), f"Missing MJCF: {xml_path}"
    xml = str(xml_path)
    return (xml,)


@app.cell
def __linearization(xml):
    def linearize_at_key(
        xml_file: str, key_name: str = "upright", eps: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = MjModel.from_xml_path(xml_file)
        d = MjData(m)
        kid = m.key(key_name).id
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        nq, nv, nu = m.nq, m.nv, m.nu
        q0 = d.qpos.copy()
        v0 = d.qvel.copy()
        u0 = d.ctrl.copy()

        def rhs(q: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
            d.qpos[:] = q
            d.qvel[:] = v
            d.ctrl[:] = u
            mujoco.mj_forward(m, d)
            return np.concatenate([d.qvel, d.qacc])

        nx = nq + nv
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        for j in range(nx):
            dx = np.zeros(nx)
            dx[j] = eps
            dq, dv = dx[:nq], dx[nq:]
            A[:, j] = (rhs(q0 + dq, v0 + dv, u0) - rhs(q0 - dq, v0 - dv, u0)) / (2 * eps)
        for j in range(nu):
            du = np.zeros(nu)
            du[j] = eps
            B[:, j] = (rhs(q0, v0, u0 + du) - rhs(q0, v0, u0 - du)) / (2 * eps)
        return A, B, q0, v0, u0

    A_np, B_np, q_eq, v_eq, u_eq = linearize_at_key(xml, "upright")
    # Large R → small gains so LQR is not instantly saturated; Q weights balance cart vs links.
    # Tuned with MuJoCo `mj_step` for: brief open-loop kick (see rollout) then this LQR.
    Q_np = np.diag([80.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, 15.0])
    R_np = np.array([[5000.0]])
    # Synthesize the stabilizing gain with Jaxonomy's LQR (continuous-time
    # algebraic-Riccati solve). This is the SAME K the LinearQuadraticRegulator
    # block builds in the diagram below; reading it off the block and driving
    # the MuJoCo rollout with it makes Jaxonomy the control-design engine and
    # MuJoCo the high-fidelity physics reference it is validated against.
    _lqr_gain = library.LinearQuadraticRegulator(
        jnp.asarray(A_np),
        jnp.asarray(B_np),
        jnp.asarray(Q_np),
        jnp.asarray(R_np),
        name="lqr_gain",
    )
    K_np = np.asarray(_lqr_gain.K).reshape(1, -1)

    A_j = jnp.asarray(A_np)
    B_j = jnp.asarray(B_np)
    Q_j = jnp.asarray(Q_np)
    R_j = jnp.asarray(R_np)
    q_eq_j = jnp.asarray(q_eq)
    return A_j, B_j, K_np, Q_j, R_j, q_eq_j


@app.cell
def __diagram(A_j, B_j, Q_j, R_j, mo, xml):
    builder = DiagramBuilder()
    plant = builder.add(
        library.MJX(
            file_name=xml,
            dt=None,
            key_frame_0="upright",
            name="plant",
        )
    )
    mux = builder.add(library.Multiplexer(2, name="state_mux"))
    lqr = builder.add(
        library.LinearQuadraticRegulator(A_j, B_j, Q_j, R_j, name="lqr"),
    )
    builder.connect(plant.output_ports[0], mux.input_ports[0])
    builder.connect(plant.output_ports[1], mux.input_ports[1])
    builder.connect(mux.output_ports[0], lqr.input_ports[0])
    builder.connect(lqr.output_ports[0], plant.input_ports[0])
    diagram = builder.build()

    mo.md(
        r"""
        ## Jaxonomy diagram

        `Multiplexer` stacks `qpos` and `qvel` into the $8$-vector $x$ for LQR.
        The regulator implements $u = -K x$ (equilibrium at the upright keyframe).
        """
    )
    diagram.pprint()
    return diagram, plant


@app.cell
def __rollout_mjstep(K_np, xml):
    """MuJoCo `mj_step`: cosine **kick-up** impulse from upright, then Jaxonomy-tuned LQR."""
    m = MjModel.from_xml_path(xml)
    _mjd = MjData(m)
    mujoco.mj_resetDataKeyframe(m, _mjd, m.key("upright").id)

    _dt = float(m.opt.timestep)
    _T = 8.0
    n_steps = int(_T / _dt)
    q_hist = np.zeros((n_steps, m.nq))
    v_hist = np.zeros((n_steps, m.nv))
    u_hist = np.zeros((n_steps, m.nu))
    kick_phase = np.zeros(n_steps, dtype=bool)

    # Open-loop cart force u(t)=A cos(2π f t) for t∈[0,T_kick); then u=-clip(Kx).
    _T_kick = 0.35
    _A_kick, _f_kick = 40.0, 2.5
    _u_lim = 300.0

    for _sk in range(n_steps):
        _t = _sk * _dt
        q_hist[_sk] = _mjd.qpos.copy()
        v_hist[_sk] = _mjd.qvel.copy()
        _x = np.concatenate([_mjd.qpos, _mjd.qvel])
        if _t < _T_kick:
            _u = float(
                np.clip(_A_kick * np.cos(2.0 * np.pi * _f_kick * _t), -_u_lim, _u_lim)
            )
            kick_phase[_sk] = True
        else:
            _u = float(np.clip(-(K_np @ _x.reshape(-1, 1)).item(), -_u_lim, _u_lim))
            kick_phase[_sk] = False
        u_hist[_sk] = _u
        _mjd.ctrl[:] = _u
        mujoco.mj_step(m, _mjd)

    t_series = np.arange(n_steps) * _dt
    return m, q_hist, t_series, u_hist, v_hist, kick_phase


@app.cell
def __plots(mo, q_hist, t_series, u_hist, v_hist, kick_phase):
    mo.md(r"## Time series (MuJoCo `mj_step`: kick-up then LQR)")

    _fig, _axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    _kick = kick_phase.astype(bool)
    if np.any(_kick):
        _tk = t_series[_kick]
        _axes[0].axvspan(float(_tk.min()), float(_tk.max()), color="orange", alpha=0.12, label="open-loop kick")
    labels = [r"$x_{\mathrm{cart}}$", r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    for _qi in range(4):
        _axes[0].plot(t_series, q_hist[:, _qi], label=labels[_qi])
    _axes[0].set_ylabel("qpos [rad or m]")
    _axes[0].legend(loc="upper right", fontsize=8)
    _axes[0].grid(True, alpha=0.3)
    _axes[0].set_title("Generalized positions")

    for _vi in range(4):
        _axes[1].plot(t_series, v_hist[:, _vi], label=f"v{_vi}")
    _axes[1].set_ylabel("qvel")
    _axes[1].legend(loc="upper right", fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    _axes[1].set_title("Generalized velocities")

    _axes[2].plot(t_series, u_hist[:, 0], color="C3")
    _axes[2].set_ylabel("u [N]")
    _axes[2].set_xlabel("time [s]")
    _axes[2].grid(True, alpha=0.3)
    _axes[2].set_title("Cart force (saturated)")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def __visualize_animation(m, mo, q_hist, t_series):
    mo.md(
        r"""
        ## Animation (MuJoCo + Pillow GIF, no ffmpeg)

        Frames use **MuJoCo `mj_forward`** body positions (side view in $xz$). If OpenGL works, a **3D
        `Renderer`** clip is appended below the schematic.
        """
    )

    _dt = float(t_series[1] - t_series[0]) if len(t_series) > 1 else 0.01
    # Cap the frame count so the GIFs stay well under marimo's 10 MB output limit.
    _TARGET_FRAMES = 100
    _stride = max(1, len(q_hist) // _TARGET_FRAMES)

    def _quantize(_rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(_rgb).quantize(colors=128, method=Image.Quantize.MEDIANCUT)

    _body_names = ("cart", "link1", "link2", "link3")
    _bids = tuple(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _n) for _n in _body_names
    )

    def _stick_rgb(_q: np.ndarray) -> np.ndarray:
        _d = MjData(m)
        _d.qpos[:] = _q
        _d.qvel[:] = 0.0
        mujoco.mj_forward(m, _d)
        _pts = np.stack([_d.xpos[_bi].copy() for _bi in _bids], axis=0)
        _w_in, _h_in, _dpi = 6.4, 4.0, 100
        _fig, _ax = plt.subplots(figsize=(_w_in, _h_in), dpi=_dpi)
        _ax.set_aspect("equal")
        _ax.plot(_pts[:, 0], _pts[:, 2], "o-", color="0.2", lw=3, markersize=9)
        _ax.fill_between([-2.6, 2.6], -0.05, 0.0, color="0.85", zorder=0)
        _ax.set_xlim(-2.5, 2.5)
        _ax.set_ylim(-0.1, 0.55)
        _ax.set_xlabel("x [m]")
        _ax.set_ylabel("z [m]")
        _ax.set_title("MuJoCo FK — triple pendulum on cart")
        _ax.grid(True, alpha=0.25)
        _canvas = FigureCanvasAgg(_fig)
        _canvas.draw()
        _rgba = np.asarray(_canvas.buffer_rgba())
        plt.close(_fig)
        return _rgba[:, :, :3].copy()

    _pil_frames = [_quantize(_stick_rgb(q_hist[_fi]))
                   for _fi in range(0, len(q_hist), _stride)]

    _duration_ms = max(30, int(1000.0 * _stride * _dt))
    _buf = io.BytesIO()
    _pil_frames[0].save(
        _buf,
        format="GIF",
        save_all=True,
        append_images=_pil_frames[1:],
        duration=_duration_ms,
        loop=0,
        optimize=True,
    )
    _viz_rows = [
        mo.md(f"**Side view (matplotlib)** — {len(_pil_frames)} frames, "
              f"{len(_buf.getvalue()) / 1024:.0f} KB"),
        mo.image(_buf.getvalue()),
    ]
    _gl_frames = []
    try:
        _rw, _rh = 480, 270
        _gl_stride = _stride * 2
        _renderer = Renderer(m, width=_rw, height=_rh)
        _scene_option = MjvOption()
        _scene_option.flags[mjtVisFlag.mjVIS_JOINT] = True
        _mjd_gl = MjData(m)
        for _fi in range(0, len(q_hist), _gl_stride):
            _mjd_gl.qpos[:] = q_hist[_fi]
            _mjd_gl.qvel[:] = 0.0
            mujoco.mj_forward(m, _mjd_gl)
            _renderer.update_scene(_mjd_gl, scene_option=_scene_option)
            _gl_frames.append(_quantize(_renderer.render()))
        if _gl_frames:
            _buf2 = io.BytesIO()
            _gl_frames[0].save(
                _buf2,
                format="GIF",
                save_all=True,
                append_images=_gl_frames[1:],
                duration=_duration_ms * 2,
                loop=0,
                optimize=True,
            )
            _viz_rows.append(mo.md(f"**MuJoCo Renderer (3D)** — {len(_gl_frames)} frames, "
                                   f"{len(_buf2.getvalue()) / 1024:.0f} KB"))
            _viz_rows.append(mo.image(_buf2.getvalue()))
    except Exception as _e:
        _viz_rows.append(
            mo.md(
                f"*MuJoCo Renderer skipped ({type(_e).__name__}: {_e}). Schematic GIF above is enough for the tutorial.*"
            )
        )
    mo.vstack(_viz_rows)
    return


@app.cell
def __optional_jaxonomy_sim(diagram, mo, plant, q_eq_j):
    mo.md(
        r"""
        ## Optional: `jaxonomy.simulate` with MJX (same block diagram)

        This runs the **adaptive** ODE solver on the **MJX** continuous-time dynamics. For large tilts it can
        disagree with `mj_step` or drift from the upright setpoint; here we use a **small** initial angle on
        link 1 so the trajectory stays in the linear regulator’s useful neighborhood.
        """
    )

    _ctx = diagram.create_context()
    _x0 = jnp.concatenate([q_eq_j.at[1].set(0.01), jnp.zeros(4)])
    _ctx = _ctx.with_continuous_state([_x0])
    _sim_options = SimulatorOptions(
        max_major_steps=4000,
        atol=1e-8,
        rtol=1e-7,
        max_minor_step_size=0.002,
    )
    _sim_results = simulate(
        diagram,
        _ctx,
        (0.0, 1.5),
        options=_sim_options,
        recorded_signals={"qpos": plant.output_ports[0]},
    )
    _t_j = np.asarray(_sim_results.time)
    _q_j = np.asarray(_sim_results.outputs["qpos"])
    _opt_fig, _opt_ax = plt.subplots(1, 1, figsize=(9, 3))
    _opt_ax.plot(_t_j, _q_j[:, 0], label=r"$x_{\mathrm{cart}}$")
    _opt_ax.plot(_t_j, _q_j[:, 1], label=r"$\theta_1$")
    _opt_ax.set_xlabel("time [s]")
    _opt_ax.set_ylabel("qpos")
    _opt_ax.grid(True, alpha=0.3)
    _opt_ax.legend()
    _opt_ax.set_title("MJX + jaxonomy.simulate (optional; small IC)")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def __summary(mo):
    mo.md(r"""
    ## Summary

    - **Model**: four generalized positions (cart slide + three hinges) and matching velocities.
    - **Linearization**: central differences on $[ \dot{q}; \dot{v} ] = [ v; a(q,v,u) ]$ at the upright keyframe.
    - **LQR**: Jaxonomy's `LinearQuadraticRegulator` yields $K$ (continuous-time Riccati solve) and applies $u = -K x$ in the diagram; the MuJoCo `mj_step` rollout is driven by that same $K$.
    - **Kick + LQR**: short open-loop **cosine** cart pulse, then saturated linear feedback $K$ from LQR.
    - **Visualization**: schematic GIF (Pillow) from MuJoCo FK; optional 3D GIF from `Renderer` when GL works.

    Full **swing-up from hanging** needs a dedicated nonlinear / MPC policy; this notebook focuses on a reliable
    **disturbance + recovery** demo with the **same** $K$ wired in Jaxonomy.
    """)
    return


if __name__ == "__main__":
    app.run()
