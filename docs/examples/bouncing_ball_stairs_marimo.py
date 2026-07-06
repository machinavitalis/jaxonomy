# Three balls cascading down stairs: Jaxonomy hybrid dynamics + MuJoCo visualization
# Run: marimo edit docs/examples/bouncing_ball_stairs_marimo.py
# Deps: marimo, jaxonomy, jax, mujoco, matplotlib, pillow, numpy (no ffmpeg)
#
# A bocce, a golf ball, and a ping-pong ball drop onto the same staircase.
# The *motion* is computed by Jaxonomy as a hybrid dynamical system — each ball
# is a point mass in free fall with a zero-crossing event at every tread and a
# Newtonian restitution reset. MuJoCo is used purely for *visualization*: we set
# the rendered ball positions from the Jaxonomy trajectory each frame (kinematic
# `mj_forward`, no `mj_step`).
#
# Editor note: uses `_`-prefixed locals in render cells so marimo does not
# report duplicate variable names across cells. Falls back from `__file__` to
# cwd because some marimo runs leave it undefined.

import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import io
    import pathlib

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import mujoco
    from mujoco import MjData, MjModel, MjvOption, Renderer, mjtVisFlag
    import numpy as np

    import jaxonomy
    from jaxonomy import DiagramBuilder, SimulatorOptions, simulate

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle, Circle
    from PIL import Image

    # --- Staircase geometry (matches mujoco/assets/bouncing_ball_stairs.xml) ---
    # 5 contiguous treads, width 0.60 m, tops descending 1.40 -> 0.60 m by 0.20 m.
    STAIR_W = 0.60          # tread width in x
    STEP_TOP0 = 1.40        # top surface z of step 1
    STEP_DROP = 0.20        # z drop per step
    N_STEPS = 5
    GRAV = 9.81

    def floor_height(x):
        """Top surface z of the tread directly under x (clipped to a landing
        platform at the last step's height beyond the staircase)."""
        k = jnp.clip(jnp.floor((x + STAIR_W / 2) / STAIR_W), 0.0, N_STEPS - 1.0)
        return STEP_TOP0 - STEP_DROP * k

    class BallOnStairs(jaxonomy.LeafSystem):
        """A point-mass ball bouncing down the staircase — a hybrid system.

        Continuous state ``[x, z, vx, vz]``; free fall between contacts. A
        zero-crossing on the signed distance to the tread below triggers a
        Newtonian restitution reset ``vz -> -e * vz`` (with a light tangential
        factor ``mu_t`` on ``vx``). Different ``e`` per ball reproduces the
        heavy/medium/light bounce characters without any contact solver.
        """

        def __init__(self, e, r, mu_t=0.98, g=GRAV, name="ball"):
            super().__init__(name=name)
            self.e, self.r, self.mu_t = e, r, mu_t
            self.declare_continuous_state(4, ode=self.ode)
            self.declare_continuous_state_output(name="q")
            self.declare_dynamic_parameter("g", g)
            self.declare_zero_crossing(
                guard=self._guard,
                reset_map=self._reset,
                name="bounce",
                direction="positive_then_non_positive",
            )

        def ode(self, time, state, **params):
            x, z, vx, vz = state.continuous_state
            return jnp.array([vx, vz, 0.0, -params["g"]])

        def _guard(self, time, state, **params):
            x, z, vx, vz = state.continuous_state
            return z - self.r - floor_height(x)          # 0 when the ball touches the tread

        def _reset(self, time, state, **params):
            x, z, vx, vz = state.continuous_state
            z_contact = floor_height(x) + self.r
            return state.with_continuous_state(
                jnp.array([x, z_contact, self.mu_t * vx, -self.e * vz])
            )

    # Per-ball metadata. ``qpos_off`` follows the body order in the MJCF (7 floats
    # per free joint) so the visualization qpos vector lines up with the model.
    #   name, qpos_off, y_offset, face_color, plot_color, mass(kg), radius(m), e, label
    BALLS = [
        ("bocce",    0, -0.30, "#8c5a1f", "C1", 0.920, 0.080, 0.55, "Bocce (heavy, least elastic)"),
        ("golf",     7,  0.00, "#dddddd", "C2", 0.046, 0.050, 0.70, "Golf (medium, lively)"),
        ("pingpong", 14, 0.30, "#f08c1a", "C3", 0.005, 0.040, 0.85, "Ping-pong (light, elastic)"),
    ]

    # Release: dropped from just above step 1, walking rightward down the stairs.
    X0 = -0.20
    VX0 = 1.15
    DROP = 0.25             # release height above the first tread contact
    T_END = 2.3


@app.cell
def __mo():
    import marimo as mo

    return (mo,)


@app.cell
def __intro(mo):
    mo.md(r"""
    # Three balls down a staircase — Jaxonomy dynamics, MuJoCo visualization

    A **bocce**, a **golf ball**, and a **ping-pong ball** are released above the
    same five-step staircase. The interesting physics is the **bounce** — and here
    it is computed by **Jaxonomy as a hybrid dynamical system**, not by a contact
    solver. Each ball is a point mass with state $[x, z, \dot x, \dot z]$ in free
    fall, plus a **zero-crossing event** whenever it meets the tread below:

    $$
    g(x, z) = z - r - z_{\mathrm{floor}}(x) \;\xrightarrow{\;=0\;}\;
    \dot z \mapsto -e\,\dot z .
    $$

    The restitution coefficient $e$ is the only thing that differs between balls
    ($e = 0.85 / 0.70 / 0.55$ for ping-pong / golf / bocce), and it alone produces
    the "light-and-elastic" vs. "heavy-and-thuddy" characters.

    | Ball       | Mass (kg) | Radius (m) | Restitution $e$ |
    |------------|----------:|-----------:|----------------:|
    | Bocce      |   0.92    |   0.08     | 0.55            |
    | Golf       |   0.046   |   0.05     | 0.70            |
    | Ping-pong  |   0.005   |   0.04     | 0.85            |

    **Jaxonomy** integrates the dynamics (adaptive ODE solver + event bisection);
    **MuJoCo** is used only to *render* the result — each frame sets the ball
    positions from the Jaxonomy trajectory (`mj_forward`, no `mj_step`). This is
    the digital-twin split: the engine owns the physics, the renderer owns the
    picture. Contrast with the Jaxonomy-native [bouncing ball](bouncing_ball.ipynb)
    notebook (single ball, flat floor).
    """)
    return


@app.cell
def __paths():
    _here = globals().get("__file__")
    _here_dir = pathlib.Path(_here).resolve().parent if _here else pathlib.Path.cwd()
    candidates = [
        _here_dir / "mujoco" / "assets" / "bouncing_ball_stairs.xml",
        pathlib.Path.cwd() / "docs" / "examples" / "mujoco" / "assets" / "bouncing_ball_stairs.xml",
        pathlib.Path.cwd() / "mujoco" / "assets" / "bouncing_ball_stairs.xml",
    ]
    xml_path = next((p for p in candidates if p.is_file()), None)
    assert xml_path is not None, (
        "Could not locate bouncing_ball_stairs.xml. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )
    xml = str(xml_path)
    return (xml,)


@app.cell
def __simulate(mo):
    """Integrate all three balls with Jaxonomy in one diagram, then repackage the
    trajectories into a MuJoCo ``qpos`` history for the renderer."""
    _builder = DiagramBuilder()
    _blocks = {}
    for _name, _qo, _yoff, _fc, _pc, _mass, _r, _e, _label in BALLS:
        _blocks[_name] = _builder.add(BallOnStairs(_e, _r, name=_name))
    _diagram = _builder.build()

    _ctx = _diagram.create_context()
    _ics = []
    for _name, _qo, _yoff, _fc, _pc, _mass, _r, _e, _label in BALLS:
        _ics.append(jnp.array([X0, STEP_TOP0 + _r + DROP, VX0, 0.0]))
    _ctx = _ctx.with_continuous_state(_ics)

    _opts = SimulatorOptions(max_major_steps=8000, max_major_step_length=0.004)
    _rec = {_name: _blocks[_name].output_ports[0] for _name, *_ in BALLS}
    _sol = simulate(_diagram, _ctx, (0.0, T_END), options=_opts, recorded_signals=_rec)

    t_arr = np.asarray(_sol.time)
    ball_state = {_name: np.asarray(_sol.outputs[_name]) for _name, *_ in BALLS}  # (T,4): x,z,vx,vz

    # Build a MuJoCo qpos history (T, nq=21) from the Jaxonomy trajectory:
    # [x, y_offset, z, quat=identity] per free joint.
    _T = len(t_arr)
    q_hist = np.zeros((_T, 21))
    for _name, _qo, _yoff, _fc, _pc, _mass, _r, _e, _label in BALLS:
        _s = ball_state[_name]
        q_hist[:, _qo + 0] = _s[:, 0]      # x
        q_hist[:, _qo + 1] = _yoff         # y (fixed track, for visual separation)
        q_hist[:, _qo + 2] = _s[:, 1]      # z
        q_hist[:, _qo + 3] = 1.0           # quat w (identity)

    # Bounce instants per ball from vz sign change through impact.
    bounces_by_ball = {}
    for _name, *_ in BALLS:
        _vz = ball_state[_name][:, 3]
        _idx = np.where((_vz[:-1] < -0.2) & (_vz[1:] > 0.2))[0] + 1
        bounces_by_ball[_name] = t_arr[_idx]

    _rows = []
    for _name, _qo, _yoff, _fc, _pc, _mass, _r, _e, _label in BALLS:
        _s = ball_state[_name]
        _rows.append(f"| {_label} | {_e:.2f} | {len(bounces_by_ball[_name]):d} | "
                     f"{_s[-1, 0]:.2f} | {_s[-1, 1]:.3f} |")
    mo.md(rf"""
    Integrated **{T_END:.1f} s** with Jaxonomy's adaptive solver + event bisection
    ({len(t_arr)} recorded samples). Each *bounce* is a zero-crossing event the
    solver localized and applied a restitution reset to.

    | Ball | $e$ | Bounces | Final x [m] | Final z [m] |
    |------|----:|--------:|------------:|------------:|
    {chr(10).join(_rows)}

    Lower $e$ = fewer, deader bounces (bocce mostly drops tread-to-tread); higher
    $e$ = many lively hops (ping-pong).
    """)
    return (t_arr, ball_state, q_hist, bounces_by_ball)


@app.cell
def __plots(t_arr, ball_state, bounces_by_ball, mo):
    """Height traces + normalized mechanical energy, straight from the Jaxonomy state."""
    _fig, _axs = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

    for _name, _qo, _yoff, _fc, _pc, _mass, _r, _e, _label in BALLS:
        _s = ball_state[_name]
        _axs[0].plot(t_arr, _s[:, 1], color=_pc, lw=1.4, label=_label)
        for _t in bounces_by_ball[_name]:
            _axs[0].plot([_t], [_r], "v", color=_pc, markersize=4, alpha=0.5)
    _axs[0].set_ylabel("z [m]")
    _axs[0].set_title("Ball height (Jaxonomy) — markers (▽) at localized bounce events")
    _axs[0].grid(alpha=0.25)
    _axs[0].legend(loc="upper right", fontsize=8)

    for _name, _qo, _yoff, _fc, _pc, _mass, _r, _e, _label in BALLS:
        _s = ball_state[_name]
        _z, _vx, _vz = _s[:, 1], _s[:, 2], _s[:, 3]
        _etot = 0.5 * _mass * (_vx**2 + _vz**2) + _mass * GRAV * _z
        _e0 = float(_etot[0]) if float(_etot[0]) > 0 else 1.0
        _axs[1].plot(t_arr, _etot / _e0, color=_pc, lw=1.4, label=_label)
    _axs[1].set_xlabel("t [s]")
    _axs[1].set_ylabel("E(t) / E(0)")
    _axs[1].set_title("Mechanical energy — each bounce removes a factor $(1-e^2)$ of the vertical KE")
    _axs[1].grid(alpha=0.25)
    _axs[1].legend(loc="upper right", fontsize=8)
    _axs[1].set_ylim(0.0, 1.05)

    _fig.tight_layout()
    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=110)
    plt.close(_fig)
    mo.image(_buf.getvalue())
    return


@app.cell
def __model_for_render(xml):
    # MuJoCo model loaded for its geometry only — the renderer draws the staircase
    # and spheres; positions are driven kinematically from the Jaxonomy trajectory.
    m = MjModel.from_xml_path(xml)
    return (m,)


@app.cell
def __animation(m, q_hist, t_arr, mo):
    """Side-view GIF (matplotlib, all balls overlaid) + 3-D MuJoCo Renderer GIF,
    both driven by the Jaxonomy ``q_hist`` (no MuJoCo physics stepping)."""
    _TARGET_FRAMES = 120
    _stride = max(1, len(q_hist) // _TARGET_FRAMES)
    _dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 0.01

    _step_geoms = []
    for _i in range(m.ngeom):
        _name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, _i)
        if _name and _name.startswith("step"):
            _step_geoms.append((m.geom_pos[_i].copy(), m.geom_size[_i].copy()))

    def _stick_rgb(_q):
        _fig, _ax = plt.subplots(figsize=(6.0, 3.6), dpi=90)
        _ax.set_aspect("equal")
        _ax.set_facecolor("#e8eef5")
        _ax.fill_between([-0.6, 3.2], -0.05, 0.0, color="0.78", zorder=0)
        for _pos, _half in _step_geoms:
            _ax.add_patch(Rectangle(
                (float(_pos[0] - _half[0]), float(_pos[2] - _half[2])),
                2 * float(_half[0]), 2 * float(_half[2]),
                facecolor="#bfa888", edgecolor="#7a6a4f", lw=1.0, zorder=1))
        for _name, _qo, _yoff, _fc, _pc, _mass, _radius, _e, _label in BALLS:
            _ax.add_patch(Circle(
                (float(_q[_qo + 0]), float(_q[_qo + 2])), _radius,
                facecolor=_fc, edgecolor="#222", lw=0.8,
                zorder=3 + int(1.0 / _mass)))  # lighter ball on top
        _ax.set_xlim(-0.6, 3.2)
        _ax.set_ylim(-0.1, 2.2)
        _ax.set_xlabel("x [m]")
        _ax.set_ylabel("z [m]")
        _ax.set_title("Side view (XZ) — positions from Jaxonomy, staggered in y")
        _ax.grid(True, alpha=0.2)
        _canvas = FigureCanvasAgg(_fig)
        _canvas.draw()
        _rgba = np.asarray(_canvas.buffer_rgba())
        plt.close(_fig)
        return _rgba[:, :, :3].copy()

    def _quantize(rgb):
        return Image.fromarray(rgb).quantize(colors=128, method=Image.Quantize.MEDIANCUT)

    _pil_frames = [_quantize(_stick_rgb(q_hist[_fi])) for _fi in range(0, len(q_hist), _stride)]
    _duration_ms = max(20, int(1000.0 * _stride * _dt))
    _buf = io.BytesIO()
    _pil_frames[0].save(_buf, format="GIF", save_all=True, append_images=_pil_frames[1:],
                        duration=_duration_ms, loop=0, optimize=True)
    _viz_rows = [
        mo.md(f"**Side view (matplotlib)** — {len(_pil_frames)} frames, "
              f"{len(_buf.getvalue()) / 1024:.0f} KB. Positions come from the "
              f"Jaxonomy trajectory."),
        mo.image(_buf.getvalue()),
    ]

    # Optional 3-D MuJoCo Renderer pass — kinematic: set qpos, mj_forward, render.
    try:
        _gl_stride = max(1, _stride * 2)
        _renderer = Renderer(m, width=480, height=320)
        _scene_option = MjvOption()
        _mjd_gl = MjData(m)
        _gl_frames = []
        for _fi in range(0, len(q_hist), _gl_stride):
            _mjd_gl.qpos[:] = q_hist[_fi]
            _mjd_gl.qvel[:] = 0.0
            mujoco.mj_forward(m, _mjd_gl)          # kinematics only, no dynamics
            _renderer.update_scene(_mjd_gl, scene_option=_scene_option)
            _gl_frames.append(_quantize(_renderer.render()))
        if _gl_frames:
            _buf2 = io.BytesIO()
            _gl_frames[0].save(_buf2, format="GIF", save_all=True, append_images=_gl_frames[1:],
                               duration=_duration_ms * 2, loop=0, optimize=True)
            _viz_rows.append(mo.md(
                f"**3-D MuJoCo Renderer** — {len(_gl_frames)} frames, "
                f"{len(_buf2.getvalue()) / 1024:.0f} KB. The spheres are placed at the "
                f"Jaxonomy positions (bocce at y=-0.3, golf at 0, ping-pong at +0.3); "
                f"MuJoCo runs `mj_forward` only — no contact solver."))
            _viz_rows.append(mo.image(_buf2.getvalue()))
    except Exception as _e:
        _viz_rows.append(mo.md(
            f"*MuJoCo `Renderer` skipped ({type(_e).__name__}: {_e}). "
            "Side-view GIF above is enough for the demo.*"))

    mo.vstack(_viz_rows)
    return


@app.cell
def __coda(mo, bounces_by_ball):
    _bocce_n = len(bounces_by_ball.get("bocce", []))
    _golf_n = len(bounces_by_ball.get("golf", []))
    _ping_n = len(bounces_by_ball.get("pingpong", []))
    mo.md(rf"""
    ---

    ### What you're seeing

    All the motion — **{_bocce_n} bocce**, **{_golf_n} golf**, **{_ping_n} ping-pong**
    bounces — comes from **one Jaxonomy hybrid model** integrated with an adaptive
    ODE solver and zero-crossing event localization. At each tread the solver
    bisects to the contact instant and applies the restitution reset
    $\dot z \mapsto -e\,\dot z$; between contacts the ball is a projectile.

    The energy panel makes the restitution quantitative: each bounce keeps a
    fraction $e^2$ of the vertical kinetic energy, so a **shallower decay slope
    means a bouncier ball**. The bocce ($e = 0.55$, keeps $0.55^2 \approx 30\%$)
    drops tread-to-tread; the ping-pong ($e = 0.85$, keeps $\approx 72\%$) hops
    lively down the whole flight.

    **The split that matters:** Jaxonomy owns the *dynamics* (events, restitution,
    integration — all `jit`/`grad`-able), and MuJoCo owns the *picture* (kinematic
    `mj_forward` + `Renderer`). To experiment, change each ball's restitution `e`
    (or `X0`/`VX0`) in the `app.setup` block and re-run — the trajectories, the
    energy curves, and both animations update together.

    A natural next step: because the whole model is differentiable, you could take
    `jax.grad` of "final x-position" with respect to `e` or the release velocity —
    gradients that flow *through* the discrete bounce events via Jaxonomy's
    event-time sensitivity machinery.
    """)
    return


if __name__ == "__main__":
    app.run()
