"""Offline generator for `sensitivity_batch.png` (README §5).

Renders the two claims of "Differentiable Sensitivity & Batch Simulation"
straight from the section's own code:

  * left  — 1000 spring-mass trajectories from a single ``simulate_batch``
             call, colored by the swept mass (a sequential magnitude encoding);
  * right — ``jax.grad`` of the final position w.r.t. every parameter, from a
             single reverse pass (signed bars, sign by color + direction).

Run from inside the jaxonomy repo:  ``python docs/examples/media/sensitivity_batch_offline.py``

Palette follows the data-viz house system: sequential blue ramp for the mass
sweep (clamped to a legible light end), and a cool/warm categorical pair
(aqua = raises x_final, orange = lowers x_final) for the signed gradients.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

import jaxonomy as jx

logging.getLogger("jaxonomy").setLevel(logging.WARNING)

# ── palette (data-viz reference instance) ───────────────────────────────────
SURFACE = "#f6f8fb"       # light card, matches the other README figures
INK = "#22303f"           # primary ink
INK_2 = "#52514e"         # secondary ink
MUTED = "#898781"         # axis / labels
GRID = "#e1e0d9"          # hairline grid
AXIS = "#c3c2b7"          # baseline / axis
BLUE_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#104281"]  # seq, light→dark
AQUA = "#1baf7a"          # positive gradient  (raises x_final)
ORANGE = "#eb6834"        # negative gradient  (lowers x_final)
blue_cmap = LinearSegmentedColormap.from_list("jx_blue", BLUE_RAMP)


class SpringMass(jx.LeafSystem):
    def __init__(self, mass=1.0, damping=0.1, stiffness=10.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("mass", mass)
        self.declare_dynamic_parameter("damping", damping)
        self.declare_dynamic_parameter("stiffness", stiffness)
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 0.0]), ode=self._ode
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, *inputs, **params):
        x, v = state.continuous_state
        a = -(params["stiffness"] * x + params["damping"] * v) / params["mass"]
        return jnp.array([v, a])


# ── sensitivity: ∂(final position)/∂(all parameters) in one reverse pass ─────
plant = SpringMass(name="plant")
grad_opts = jx.SimulatorOptions(enable_autodiff=True, max_major_steps=200)


def final_position(theta):
    ctx = plant.create_context()
    for name, value in theta.items():
        ctx.parameters[name] = value
    res = jx.simulate(plant, ctx, (0.0, 5.0), options=grad_opts)
    return res.context.continuous_state[0]


grads = jax.grad(final_position)({"mass": 1.0, "damping": 0.1, "stiffness": 10.0})
grads = {k: float(v) for k, v in grads.items()}

# ── Monte Carlo ensemble: 1000 trajectories over a mass sweep (vmap) ─────────
N = 1000
masses = np.linspace(0.5, 2.0, N)
builder = jx.DiagramBuilder()
plant_b = builder.add(SpringMass(name="plant"))
diagram = builder.build()

results = jx.simulate_batch(
    diagram,
    t_span=(0.0, 5.0),
    param_batches={"plant.mass": jnp.linspace(0.5, 2.0, N)},
    options=jx.SimulatorOptions(math_backend="jax", max_major_steps=200),
    recorded_signals={"x": plant_b.output_ports[0]},
)
t = np.asarray(results.time)
if t.ndim == 2:
    t = t[0]
X = np.asarray(results.outputs["x"])[:, :, 0]   # (N, T) position channel

# ── figure ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig, (ax_fan, ax_bar) = plt.subplots(
    1, 2, figsize=(11.4, 4.7), gridspec_kw={"width_ratios": [2.45, 1.0]}
)
fig.patch.set_facecolor(SURFACE)
for ax in (ax_fan, ax_bar):
    ax.set_facecolor(SURFACE)


def _spines(ax, keep=("left", "bottom")):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)
        sp.set_color(AXIS)
        sp.set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=3)


# ---- left: batch fan (sequential magnitude = mass) --------------------------
segs = [np.column_stack([t, X[i]]) for i in range(N)]
lc = LineCollection(segs, linewidths=0.55, alpha=0.35, cmap=blue_cmap)
lc.set_array(masses)
lc.set_clim(0.5, 2.0)
ax_fan.add_collection(lc)
ax_fan.set_xlim(float(t.min()), float(t.max()))
pad = 0.06 * (X.max() - X.min())
ax_fan.set_ylim(float(X.min() - pad), float(X.max() + pad))
ax_fan.axhline(0, color=GRID, lw=0.8, zorder=0)
ax_fan.set_xlabel("time  $t$  (s)", color=INK_2, fontsize=11)
ax_fan.set_ylabel("position  $x(t)$", color=INK_2, fontsize=11)
ax_fan.set_title(
    "1000 trajectories · one  simulate_batch  call",
    color=INK, fontsize=12.5, fontweight="bold", pad=10, loc="left",
)
_spines(ax_fan)

sm = ScalarMappable(norm=Normalize(0.5, 2.0), cmap=blue_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_fan, pad=0.015, fraction=0.046)
cbar.set_label("mass  $m$  (kg)", color=INK_2, fontsize=10.5)
cbar.set_ticks([0.5, 1.0, 1.5, 2.0])
cbar.ax.tick_params(colors=MUTED, labelsize=9.5)
cbar.outline.set_edgecolor(AXIS)
cbar.outline.set_linewidth(0.8)
ax_fan.annotate(
    "heavier mass → slower oscillation",
    xy=(0.985, 0.045), xycoords="axes fraction", ha="right", va="bottom",
    color=INK_2, fontsize=9.5, style="italic",
    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=2.5),
)

# ---- right: gradient bars (tornado — sign by color + direction) -------------
order = ["damping", "mass", "stiffness"]            # descending |∂|
vals = [grads[k] for k in order]
ypos = np.arange(len(order))[::-1]                  # damping on top
colors = [AQUA if v >= 0 else ORANGE for v in vals]
ax_bar.barh(ypos, vals, height=0.42, color=colors, zorder=3)
ax_bar.axvline(0, color=AXIS, lw=1.3, zorder=2)
vmax = max(abs(v) for v in vals)
ax_bar.set_xlim(-0.8 * vmax, 1.5 * vmax)
ax_bar.set_ylim(-0.65, len(order) - 0.3)
ax_bar.set_yticks([])
ax_bar.set_xticks([])
for y, v, name in zip(ypos, vals, order):
    ax_bar.text(0, y + 0.34, name, ha="left", va="center",
                color=INK_2, fontsize=10.5)              # param name above its bar
    ha = "left" if v >= 0 else "right"
    dx = 0.04 * vmax * (1 if v >= 0 else -1)
    ax_bar.text(v + dx, y, f"{v:+.2f}", va="center", ha=ha,
                color=INK, fontsize=11, fontweight="bold")  # value at the tip
ax_bar.set_title(
    "$\\partial\\, x_{\\mathrm{final}} / \\partial\\, \\theta$   ·  one reverse pass",
    color=INK, fontsize=12.5, fontweight="bold", pad=10, loc="left",
)
_spines(ax_bar, keep=())
ax_bar.tick_params(length=0)
# legend: identity by swatch + text (never color alone)
handles = [Patch(facecolor=AQUA, label="raises  $x_{\\mathrm{final}}$"),
           Patch(facecolor=ORANGE, label="lowers  $x_{\\mathrm{final}}$")]
ax_bar.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.19),
              ncol=2, frameon=False, fontsize=9.5, handlelength=1.1,
              handleheight=1.1, columnspacing=1.3, labelcolor=INK_2)

fig.suptitle(
    "One SpringMass model —  vmap  fans it out,  grad  differentiates it",
    color=INK, fontsize=14, fontweight="bold", x=0.5, y=0.99,
)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
out = "docs/examples/media/sensitivity_batch.png"
fig.savefig(out, dpi=185, facecolor=SURFACE)
print("gradients:", grads)
print("batch X shape:", X.shape, "| time pts:", t.shape)
print("wrote", out)
