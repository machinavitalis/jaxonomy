# Articulated-quadruped MJX benchmark — visualisation script (T-019b followup).
# Run: python docs/examples/articulated_quadruped_render.py
# Deps: mujoco, mjx, jax, jaxonomy, Pillow.
#
# Renders a 2 s rollout of the 12-DoF articulated quadruped used by the
# T-019b benchmark (`benchmarks/public.py:_measure_articulated_quadruped_*`).
# Output: ``docs/examples/articulated_quadruped.gif``.
#
# Why two MuJoCo objects?  Jaxonomy's MJX block produces qpos as JAX
# arrays.  ``mujoco.Renderer`` is a CPU-side renderer that wants a
# ``mujoco.MjModel`` / ``mujoco.MjData`` pair — MJX has no renderer of
# its own.  We load the same XML twice (once for stepping, once for
# rendering) and copy qpos across each frame; this is the same pattern
# used by mujoco_playground / brax for visualisation.
#
# Trajectory generator: pure-MuJoCo CPU stepping (``mjx.step``).
#   Jaxonomy's continuous-time MJX path (``dt=None``) lowers cleanly but
#   the adaptive solver gets stuck taking many tiny steps when started
#   from an air-drop initial condition with the default contact softness
#   (we observed >100 s wall for a 2 s rollout under the airborne IC,
#   versus ~16 s post-compile for the rest-pose IC the benchmark uses).
#   For pure visualisation we just need a faithful trajectory of the
#   same body — ``mjx.step`` from the same MJCF, no jaxonomy framework
#   overhead, gives us that in well under a second.  The MJX/jaxonomy
#   physics throughput numbers in `benchmarks/public.py` remain the
#   benchmarked claim; this script visualises the same body, not the
#   same code path.
#
# GIF budget: 60 frames @ 480x320, palette quantised to 128 colours.
# That lands well under the 2 MB cap suggested in the task description.
from __future__ import annotations

import io
import pathlib
import sys
import time

# Make ``jaxonomy`` importable when this file is run directly (the
# project is editable-installed via ``pip install -e .``, but if the
# user invokes ``python docs/examples/...`` from elsewhere the local
# checkout still needs to be on ``sys.path``).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
XML_PATH = HERE / "mujoco" / "assets" / "articulated_quadruped.xml"
OUT_GIF = HERE / "articulated_quadruped.gif"

# Trajectory: drop from a stand-pose-ish initial condition for 2 s and
# watch the body settle.
T_END = 2.0
FPS = 30
TARGET_FRAMES = int(round(T_END * FPS))  # 60
WIDTH, HEIGHT = 480, 320


def _drop_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Build an initial qpos that drops the trunk from ~0.6 m with legs out.

    qpos layout (free-joint first): [x y z  qw qx qy qz  joint_angles...].
    The MJCF declares 12 hinges (4 legs × 3); we set the trunk to 0.6 m
    high and bend the knees so contact lands on feet rather than belly —
    visually clearer than a face-plant.
    """
    qpos = np.zeros(model.nq)
    qpos[0:3] = (0.0, 0.0, 0.60)            # trunk position
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)        # trunk quaternion (identity)
    # 4 legs * (hip_roll, hip_pitch, knee).  Slight pitch + bent knee.
    leg_block = np.array([0.0, 0.5, -1.0])
    for leg_idx in range(4):
        off = 7 + 3 * leg_idx
        qpos[off:off + 3] = leg_block
    return qpos


def simulate_quadruped() -> tuple[np.ndarray, np.ndarray]:
    """Run the 2 s rollout via pure-MuJoCo ``mj_step`` (CPU).

    Stepping the same MJCF that ``benchmarks/public.py`` benchmarks under
    Jaxonomy MJX, but through the upstream CPU stepper.  Returns
    ``(t, qpos)`` with shapes ``(T,)`` and ``(T, nq)``.
    """
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = _drop_qpos(model)
    data.qvel[:] = 0.0

    dt = float(model.opt.timestep)              # 0.002 s in the MJCF
    n_steps = int(round(T_END / dt))
    log_every = max(1, int(round(1.0 / (FPS * dt))))  # ~1 frame per 33 ms

    times: list[float] = []
    qposes: list[np.ndarray] = []
    print(f"[render] simulating t_end={T_END}s "
          f"({n_steps} mj_step at dt={dt*1e3:.1f} ms) …", flush=True)
    t0 = time.perf_counter()
    for i in range(n_steps + 1):
        if i % log_every == 0:
            times.append(float(data.time))
            qposes.append(data.qpos.copy())
        mujoco.mj_step(model, data)
    print(f"[render] sim took {time.perf_counter() - t0:.2f}s "
          f"({len(times)} samples)", flush=True)
    return np.asarray(times), np.stack(qposes, axis=0)


def _resample_uniform(t: np.ndarray, q: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample qpos onto a uniform-time grid of length ``n_frames``.

    Linear interp per qpos component (quaternions: good enough for
    visualisation at 30 fps; we re-normalise via ``mj_forward`` later).
    """
    t_uniform = np.linspace(float(t[0]), float(t[-1]), n_frames)
    q_uniform = np.empty((n_frames, q.shape[1]), dtype=np.float64)
    for i in range(q.shape[1]):
        q_uniform[:, i] = np.interp(t_uniform, t, q[:, i])
    return q_uniform


def render_frames(qpos_uniform: np.ndarray) -> list[np.ndarray]:
    """Render each qpos frame via offscreen ``mujoco.Renderer``."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)

    # A camera angled to see all four feet + the floor.  Free camera
    # rather than tracking, so the trunk's settle is visible against
    # the static floor checkerboard.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 135.0, -20.0, 1.6
    cam.lookat[:] = (0.0, 0.0, 0.20)

    frames: list[np.ndarray] = []
    for fi in range(qpos_uniform.shape[0]):
        data.qpos[:] = qpos_uniform[fi]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    return frames


def _quantize(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).quantize(colors=128, method=Image.Quantize.MEDIANCUT)


def encode_gif(frames: list[np.ndarray], out_path: pathlib.Path) -> int:
    """Encode frames into an optimised, palette-quantised GIF.  Returns bytes."""
    pil_frames = [_quantize(f) for f in frames]
    duration_ms = int(round(1000.0 / FPS))
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    out_path.write_bytes(buf.getvalue())
    return len(buf.getvalue())


def main() -> None:
    t_arr, q_hist = simulate_quadruped()
    q_uniform = _resample_uniform(t_arr, q_hist, TARGET_FRAMES)
    print(f"[render] resampled to {TARGET_FRAMES} frames @ {FPS} fps", flush=True)

    z = q_uniform[:, 2]
    z_drop = float(z[0])
    z_settle = float(z[-1])
    # Find first frame where the body has come within 1 cm of its
    # final z and stays there — rough "rest" indicator.
    eps = 0.01
    rest_idx = TARGET_FRAMES - 1
    for i in range(TARGET_FRAMES):
        if all(abs(z[j] - z_settle) < eps for j in range(i, TARGET_FRAMES)):
            rest_idx = i
            break
    print(f"[render] z_drop={z_drop:.3f} m, z_settle={z_settle:.3f} m, "
          f"rest_frame={rest_idx}/{TARGET_FRAMES} "
          f"(t_rest≈{rest_idx / FPS:.2f} s)", flush=True)

    print(f"[render] rendering {TARGET_FRAMES} frames "
          f"@ {WIDTH}x{HEIGHT} …", flush=True)
    t0 = time.perf_counter()
    frames = render_frames(q_uniform)
    print(f"[render] render took {time.perf_counter() - t0:.2f}s", flush=True)

    n_bytes = encode_gif(frames, OUT_GIF)
    kb = n_bytes / 1024.0
    print(f"[render] wrote {OUT_GIF} ({kb:.0f} KB, "
          f"{TARGET_FRAMES} frames @ {FPS} fps, {WIDTH}x{HEIGHT})", flush=True)
    if n_bytes > 2 * 1024 * 1024:
        print("[render] WARN: gif exceeds 2 MB — consider lowering frame "
              "count, resolution, or palette size.", flush=True)


if __name__ == "__main__":
    main()
