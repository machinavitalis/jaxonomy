#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline publication run for ``pinn_across_stacks_part_2_neural_dae.ipynb``.

Produces ``pinn_across_stacks_part_2_publication.npz`` (this directory) with
both full-fidelity training runs:

* **Direction 1** — the physics-structured neural flow correction inside the
  tank-network DAE, trained for 150 Adam iterations through the implicit BDF
  solve (3 excitations x 3 horizons, terminal-level loss), plus the AD-vs-FD
  gradient check and the flow-correction recovery grid.
* **Direction 2** — the NEUROMANCER policy trained through the jaxonomy DAE
  plant via the dlpack bridge, 15 epochs, plus the bridge gradcheck, the
  robustness-guard counters, and the closed-loop evaluation.

Runs ~25-40 min on a developer laptop (CPU). The notebook loads the NPZ in
<1 s so the reader's experience stays fast. Run with:

    PYTHONHASHSEED=0 python pinn_across_stacks_part_2_publication_offline.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for pinn_tank_network_dae
OUT_NPZ = HERE / "pinn_across_stacks_part_2_publication.npz"

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import torch
import torch.nn as nn

import jaxonomy as jx
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.library import Constant
from jaxonomy.library.neural_dae import add_neural_correction
from jaxonomy.simulation.dae_projection import project_constraints

from pinn_tank_network_dae import build_network

t_script0 = time.perf_counter()
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Shared plant configuration (must match the notebook)
# ---------------------------------------------------------------------------
H1_IC, H2_IC = 0.40, 0.15
K12_NOMINAL = 3e-3
K12_TRUE = 0.55 * K12_NOMINAL      # clogged orifice, unknown to the model
CENTER = {"pump_power": 50.0, "valve_a_opening": 0.5, "valve_b_opening": 0.5}

AD_OPTS = jx.SimulatorOptions(
    math_backend="jax", ode_solver_method="bdf",
    rtol=1e-7, atol=1e-9, enable_autodiff=True, max_major_steps=400,
)
TRUTH_OPTS = jx.SimulatorOptions(
    math_backend="jax", ode_solver_method="bdf",
    rtol=1e-8, atol=1e-10, enable_autodiff=False, max_major_steps=600,
)


def build_and_wire(k_12, nn_pair=None):
    """Compile the acausal network and wire centered Constant placeholders."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    build_network(ev, ad, h1_ic=H1_IC, h2_ic=H2_IC, k_12=k_12)
    system = AcausalCompiler(ev, ad, scale=True)(name="tank_dae", leaf_backend="jax")
    if nn_pair is not None:
        add_neural_correction(system, nn_pair[0], nn_pair[1], param_name="nn_theta")
    builder = jx.DiagramBuilder()
    s = builder.add(system)
    consts = {}
    for i, p in enumerate(system.input_ports):
        c = builder.add(Constant(np.array(CENTER[p.name])))
        builder.connect(c.output_ports[0], s.input_ports[i])
        consts[p.name] = c
    return builder.build(), s, consts


def set_excitation(diagram, diagram_ctx, consts, sid, exc):
    """Set (pump, valve_a, valve_b) inputs and re-project the algebraic state.

    Projection is applied value-only: AD through the unrolled Newton
    pollutes gradients (FD-verified), and the solver re-enforces g=0 anyway.
    """
    vals = dict(zip(("pump_power", "valve_a_opening", "valve_b_opening"), exc))
    ctx = diagram_ctx
    for name, v in vals.items():
        cid = consts[name].system_id
        ctx = ctx.with_subcontext(cid, ctx[cid].with_parameter("value", jnp.asarray(v)))
    proj = project_constraints(diagram, ctx, tol=1e-12, max_iter=32)
    raw = ctx[sid].continuous_state
    fixed = raw + jax.lax.stop_gradient(proj[sid].continuous_state - raw)
    return ctx.with_subcontext(sid, ctx[sid].with_continuous_state(fixed))


# ===========================================================================
# Direction 1 — learn the clogged orifice inside the DAE
# ===========================================================================
print("=" * 70)
print("Direction 1: physics-structured neural correction inside the DAE")
print("=" * 70)

HORIZONS = (8.0, 16.0, 32.0)
EXCITATIONS = ((50.0, 0.5, 0.5), (80.0, 0.8, 0.2), (30.0, 0.2, 0.8))
FLOW_SCALE = 0.15   # max |dm| in kg/s; the injected flow deficit peaks ~0.08 kg/s
N_ITERS = 150
LR = 8e-3

# ---- truth data (clogged plant) -------------------------------------------
truth_diagram, truth_s, truth_consts = build_and_wire(K12_TRUE)
truth_ctx0 = truth_diagram.create_context()
TRUTH_SID = truth_s.system_id

def truth_levels(exc, T):
    ctx = set_excitation(truth_diagram, truth_ctx0, truth_consts, TRUTH_SID, exc)
    r = jx.simulate(truth_diagram, ctx, (0.0, T), options=TRUTH_OPTS)
    o1 = truth_s.output_ports[0].eval(r.context)
    o2 = truth_s.output_ports[1].eval(r.context)
    return np.array([float(o1), float(o2)])

targets = {(e, T): truth_levels(e, T) for e in EXCITATIONS for T in HORIZONS}
print("truth targets:")
for (e, T), v in targets.items():
    print(f"  exc={e} T={T}: {v.round(4).tolist()}")
    assert np.isfinite(v).all(), f"truth NaN at exc={e} T={T}"

# ---- model with physics-structured neural correction ----------------------
key = jax.random.PRNGKey(SEED)
k1, k2 = jax.random.split(key)
W_H = 8
params0 = {
    "w1": 0.5 * jax.random.normal(k1, (2, W_H)),
    "b1": jnp.zeros(W_H),
    "w2": 0.1 * jax.random.normal(k2, (W_H, 1)),
    "b2": jnp.zeros(1),
}
theta0, unravel = ravel_pytree(params0)

RHO_, A1_, A2_ = 1000.0, 0.02, 0.03
# scale=True normalizes each differential state by its IC, so the scaled
# rows are h/h_ic and d(scaled)/dt = (dh/dt)/h_ic
SC_ROWS = jnp.array([1.0 / H1_IC, 1.0 / H2_IC])

def nn_fn(t, x_diff, theta):
    """Physics-structured correction: one scalar flow dm(h1,h2) [kg/s],
    scattered into the two level equations with the fixed physical ratio
    (-1/(rho*A1), +1/(rho*A2)) — the rank-1 'unknown is a flow' prior."""
    p = unravel(theta)
    z = jnp.tanh(x_diff @ p["w1"] + p["b1"])
    dm = FLOW_SCALE * jnp.tanh(z @ p["w2"] + p["b2"])[0]
    dh = jnp.array([-dm / (RHO_ * A1_), dm / (RHO_ * A2_)])
    return dh * SC_ROWS  # convert physical dh/dt to scaled-state rows

model_diagram, model_s, model_consts = build_and_wire(K12_NOMINAL, nn_pair=(nn_fn, theta0))
model_ctx0 = model_diagram.create_context()
SID = model_s.system_id

def terminal_levels(theta, exc, T):
    ctx = set_excitation(model_diagram, model_ctx0, model_consts, SID, exc)
    ctx = ctx.with_subcontext(SID, ctx[SID].with_parameter("nn_theta", theta))
    r = jx.simulate(model_diagram, ctx, (0.0, T), options=AD_OPTS)
    o1 = model_s.output_ports[0].eval(r.context)
    o2 = model_s.output_ports[1].eval(r.context)
    return jnp.array([o1, o2]).reshape(-1)

targets_j = {k: jnp.asarray(v) for k, v in targets.items()}

def loss_fn(theta):
    l = 0.0
    for e in EXCITATIONS:
        for T in HORIZONS:
            l = l + jnp.sum((terminal_levels(theta, e, T) - targets_j[(e, T)]) ** 2)
    return l / (len(EXCITATIONS) * len(HORIZONS))

vg = jax.jit(jax.value_and_grad(loss_fn))

t0 = time.perf_counter()
l0, g0 = vg(theta0)
l0 = float(l0)
first_grad_s = time.perf_counter() - t0
print(f"loss(theta0)={l0:.6e}; first value_and_grad (incl jit): {first_grad_s:.1f}s")

# baseline: zero correction (tanh MLP with zero weights outputs dm=0)
theta_zero = jnp.zeros_like(theta0)
l_uncorrected = float(vg(theta_zero)[0])
print(f"loss with zero correction: {l_uncorrected:.6e}")

# AD-vs-FD gradient check through the implicit BDF solve (reuses compiled vg)
eps = 1e-4
GRAD_IDX = (0, 10, 25)
grad_ad, grad_fd = [], []
for i in GRAD_IDX:
    e = jnp.zeros_like(theta0).at[i].set(eps)
    fd = (float(vg(theta0 + e)[0]) - float(vg(theta0 - e)[0])) / (2 * eps)
    grad_ad.append(float(g0[i]))
    grad_fd.append(fd)
    print(f"grad[{i}]: AD={float(g0[i]):+.5e}  FD={fd:+.5e}")

# ---- training loop ---------------------------------------------------------
opt = optax.adam(LR)
theta = theta0
state = opt.init(theta)
hist, best = [], (np.inf, theta0)
t0 = time.perf_counter()
for it in range(N_ITERS):
    l, g = vg(theta)
    lf = float(l)
    if not np.isfinite(lf):
        print(f"iter {it}: non-finite loss, stopping"); break
    if lf < best[0]:
        best = (lf, theta)
    upd, state = opt.update(g, state)
    theta = optax.apply_updates(theta, upd)
    hist.append(lf)
    if it % 10 == 0 or it == N_ITERS - 1:
        print(f"iter {it:3d}  loss {lf:.6e}")
d1_train_wall = time.perf_counter() - t0
theta_best = best[1]
print(f"training {len(hist)} iters: {d1_train_wall:.0f}s "
      f"({d1_train_wall/len(hist):.2f}s/iter); "
      f"loss {hist[0]:.3e} -> {best[0]:.3e} ({hist[0]/best[0]:.1f}x); "
      f"vs uncorrected {l_uncorrected:.3e} ({l_uncorrected/best[0]:.1f}x better)")

# ---- validation: recovered flow correction vs injected flow deficit --------
RHO, G_N = 1000.0, 9.81
EPS_V = 1.0   # must match SqrtValve eps
def injected_dm(h1, h2):
    dP = RHO * G_N * (h1 - h2)
    return (K12_TRUE - K12_NOMINAL) * dP / (dP**2 + EPS_V) ** 0.25

def learned_dm(h1, h2, theta):
    p = unravel(theta)
    x_scaled = jnp.array([h1, h2]) * SC_ROWS
    z = jnp.tanh(x_scaled @ p["w1"] + p["b1"])
    return float(FLOW_SCALE * jnp.tanh(z @ p["w2"] + p["b2"])[0])

h1s = np.linspace(0.26, 0.47, 12)
h2s = np.linspace(0.13, 0.24, 12)
inj, lrn = [], []
for h1 in h1s:
    for h2 in h2s:
        inj.append(injected_dm(h1, h2))
        lrn.append(learned_dm(h1, h2, theta_best))
inj, lrn = np.array(inj), np.array(lrn)
d1_rel = np.abs(lrn - inj).mean() / np.abs(inj).mean()
d1_corr = np.corrcoef(lrn, inj)[0, 1]
print(f"flow-correction recovery (visited region): mean|inj|={np.abs(inj).mean():.4f} kg/s, "
      f"rel err={d1_rel:.2%}, corr={d1_corr:.3f}")

# ===========================================================================
# Direction 2 — NEUROMANCER policy trained through the jaxonomy DAE
# ===========================================================================
print("=" * 70)
print("Direction 2: torch policy through the DAE via the dlpack bridge")
print("=" * 70)

DT = 2.0            # communication interval / control period [s]
NSTEPS = 25         # rollout horizon in training
PUMP_MAX = 100.0
D2_EPOCHS = 15

# plant: nominal network, no neural correction
plant_diagram, plant_s, plant_consts = build_and_wire(K12_NOMINAL)
plant_ctx0 = plant_diagram.create_context()
PSID = plant_s.system_id

BRIDGE_OPTS = jx.SimulatorOptions(
    math_backend="jax", ode_solver_method="bdf",
    rtol=1e-7, atol=1e-9, enable_autodiff=True, max_major_steps=64,
)

cs0 = jnp.asarray(plant_ctx0[PSID].continuous_state)
scale = np.asarray(cs0[:2]) / np.array([H1_IC, H2_IC])   # diff rows = scale * level
print("state scale factors:", scale, " full state size:", cs0.shape)

_pump_id = plant_consts["pump_power"].system_id
_va_id = plant_consts["valve_a_opening"].system_id
_vb_id = plant_consts["valve_b_opening"].system_id

def _step(x_levels, u):
    """One DT interval of the DAE. x_levels (2,), u = (pump in [0,1], split v)."""
    n = cs0.at[0].set(scale[0] * x_levels[0]).at[1].set(scale[1] * x_levels[1])
    ctx = plant_ctx0.with_subcontext(PSID, plant_ctx0[PSID].with_continuous_state(n))
    ctx = ctx.with_subcontext(_pump_id, ctx[_pump_id].with_parameter("value", u[0] * PUMP_MAX))
    ctx = ctx.with_subcontext(_va_id, ctx[_va_id].with_parameter("value", 1.0 - u[1]))
    ctx = ctx.with_subcontext(_vb_id, ctx[_vb_id].with_parameter("value", u[1]))
    ctx_p = project_constraints(plant_diagram, ctx, tol=1e-12, max_iter=32)
    state_used = n + jax.lax.stop_gradient(ctx_p[PSID].continuous_state - n)
    ctx = ctx.with_subcontext(PSID, ctx[PSID].with_continuous_state(state_used))
    r = jx.simulate(plant_diagram, ctx, (0.0, DT), options=BRIDGE_OPTS)
    cse = r.context[PSID].continuous_state
    return jnp.array([cse[0] / scale[0], cse[1] / scale[1]])

step_batch = jax.jit(jax.vmap(_step))

def step_vjp(x, u, ct):
    y, pull = jax.vjp(lambda a, b: jax.vmap(_step)(a, b), x, u)
    return pull(ct)
step_vjp = jax.jit(step_vjp)

def t2j(t): return jax.dlpack.from_dlpack(t.contiguous())
def j2t(x): return torch.from_dlpack(x)

class JaxDAEStep(torch.autograd.Function):
    nan_grad_events = 0

    @staticmethod
    def forward(ctx, x, u):
        x64, u64 = t2j(x.detach().double()), t2j(u.detach().double())
        y = step_batch(x64, u64)
        ctx.save_for_backward(x, u)
        return j2t(y).float()

    @staticmethod
    def backward(ctx, gy):
        x, u = ctx.saved_tensors
        gx, gu = step_vjp(t2j(x.detach().double()), t2j(u.detach().double()),
                          t2j(gy.contiguous().double()))
        gx, gu = j2t(gx).float(), j2t(gu).float()
        if not (torch.isfinite(gx).all() and torch.isfinite(gu).all()):
            JaxDAEStep.nan_grad_events += 1
            gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
            gu = torch.nan_to_num(gu, nan=0.0, posinf=0.0, neginf=0.0)
        return gx, gu

# ---- bridge smoke test: forward timing + gradcheck -------------------------
B = 4
x = torch.rand(B, 2) * 0.4 + 0.1
u = torch.rand(B, 2) * 0.5 + 0.25
u.requires_grad_(True)

t0 = time.perf_counter()
y = JaxDAEStep.apply(x, u)
first_fwd_s = time.perf_counter() - t0
print(f"first batched DAE step (incl jit): {first_fwd_s:.1f}s; y[0]={y[0].detach().numpy()}")
t0 = time.perf_counter()
y = JaxDAEStep.apply(x, u)
steady_fwd_s = time.perf_counter() - t0
print(f"steady forward: {steady_fwd_s:.3f}s")

l = (y ** 2).sum()
t0 = time.perf_counter()
l.backward()
first_bwd_s = time.perf_counter() - t0
print(f"first backward (incl jit): {first_bwd_s:.1f}s")
g_ad = u.grad.clone()

eps = 1e-3
i, jdx = 1, 0
up = u.detach().clone(); up[i, jdx] += eps
um = u.detach().clone(); um[i, jdx] -= eps
fd = ((JaxDAEStep.apply(x, up) ** 2).sum() - (JaxDAEStep.apply(x, um) ** 2).sum()) / (2 * eps)
d2_grad_ad, d2_grad_fd = float(g_ad[i, jdx]), float(fd)
print(f"du[{i},{jdx}]: AD={d2_grad_ad:+.6e} FD={d2_grad_fd:+.6e}")

# ---- NEUROMANCER closed-loop training through the DAE ----------------------
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.callbacks import Callback

BAD_POINTS = []

class JaxPlant(nn.Module):
    def forward(self, x, u):
        # levels are physically confined to the tank: clamp like the psl
        # reference model does inside its own equations
        xc = x.clamp(0.02, 0.95)
        y = JaxDAEStep.apply(xc, u)
        bad = ~torch.isfinite(y).all(dim=-1)
        if bad.any():
            BAD_POINTS.append(np.concatenate([xc[bad].detach().numpy(),
                                              u[bad].detach().numpy()], axis=1))
            # hold state on solver failure (a co-sim master would retry/hold)
            y = torch.where(bad.unsqueeze(-1), xc.detach(), y)
        return y

plant_node = Node(JaxPlant(), ['x', 'u'], ['x'], name='dae_plant')
# pump floor 0.05: an idling pump keeps the ideal pump equation regular
policy = blocks.MLP_bounds(insize=4, outsize=2, hsizes=[32, 32], nonlin=nn.GELU,
                           min=torch.tensor([0.05, 0.0]), max=torch.ones(2))
policy_node = Node(policy, ['x', 'r'], ['u'], name='policy')
cl = System([policy_node, plant_node], nsteps=NSTEPS)

def make_ds(n, name):
    x0 = torch.rand(n, 1, 2) * 0.4 + 0.05
    r = (torch.rand(n, 1, 2) * 0.5 + 0.15).repeat(1, NSTEPS + 1, 1)
    return DictDataset({'x': x0, 'r': r}, name=name)

train_d, dev_d = make_ds(64, 'train'), make_ds(16, 'dev')
tl = torch.utils.data.DataLoader(train_d, batch_size=16, collate_fn=train_d.collate_fn, shuffle=True)
dl = torch.utils.data.DataLoader(dev_d, batch_size=16, collate_fn=dev_d.collate_fn)

xv, rv, uv = variable('x'), variable('r'), variable('u')
obj = 5.0 * ((xv == rv) ^ 2); obj.name = 'tracking'
du = 0.1 * ((uv[:, 1:, :] == uv[:, :-1, :]) ^ 2); du.name = 'smooth'
problem = Problem([cl], PenaltyLoss([obj, du], []))

class CurveLogger(Callback):
    """Capture the per-epoch mean train / dev losses the Trainer computes."""
    def __init__(self):
        self.train_curve, self.dev_curve = [], []

    def begin_eval(self, trainer, output):
        self.train_curve.append(float(output['mean_train_loss']))
        if 'mean_dev_loss' in output:
            self.dev_curve.append(float(output['mean_dev_loss']))

curves = CurveLogger()
opt2 = torch.optim.AdamW(problem.parameters(), lr=3e-3)
t0 = time.perf_counter()
trainer = Trainer(problem, tl, dl, optimizer=opt2, epochs=D2_EPOCHS,
                  train_metric='train_loss', dev_metric='dev_loss',
                  eval_metric='dev_loss', warmup=D2_EPOCHS, callback=curves)
best_state = trainer.train()
d2_train_wall = time.perf_counter() - t0
problem.load_state_dict(best_state)
n_bad = sum(len(b) for b in BAD_POINTS)
bad_pts = (np.concatenate(BAD_POINTS, axis=0) if BAD_POINTS
           else np.zeros((0, 4)))
print(f"{D2_EPOCHS} epochs through the DAE: {d2_train_wall:.1f}s; "
      f"train_loss {curves.train_curve[0]:.3f} -> {curves.train_curve[-1]:.3f}; "
      f"sanitized NaN-gradient batches: {JaxDAEStep.nan_grad_events}; "
      f"held-state solver failures: {n_bad}")

# ---- closed-loop evaluation -------------------------------------------------
with torch.no_grad():
    data = {'x': torch.tensor([[[0.40, 0.15]]]), 'r': torch.full((1, 41, 2), 0.30)}
    cl.nsteps = 40
    out = cl(data)
xs = out['x'][0].numpy()
us = out['u'][0].numpy()
print("final levels:", xs[-1], " target 0.30")
d2_rms = float(np.sqrt(((xs[20:] - 0.30) ** 2).mean()))
print(f"steady tracking RMS (last half): {d2_rms:.4f}")

# ===========================================================================
# Save the publication NPZ (<100 KB: optima, histories, grids, stats)
# ===========================================================================
policy_sd = {f"d2_policy::{k}": v.detach().cpu().numpy()
             for k, v in policy.state_dict().items()}
pub_wall = time.perf_counter() - t_script0
np.savez(
    OUT_NPZ,
    # direction 1
    d1_excitations=np.array(EXCITATIONS),
    d1_horizons=np.array(HORIZONS),
    d1_targets=np.array([targets[(e, T)] for e in EXCITATIONS for T in HORIZONS]),
    d1_theta=np.asarray(theta_best),
    d1_hist=np.asarray(hist),
    d1_loss0=l0,
    d1_l_uncorrected=l_uncorrected,
    d1_grad_idx=np.array(GRAD_IDX),
    d1_grad_ad=np.array(grad_ad),
    d1_grad_fd=np.array(grad_fd),
    d1_first_grad_s=first_grad_s,
    d1_train_wall_s=d1_train_wall,
    d1_inj_dm=inj,
    d1_learned_dm=lrn,
    d1_h1_grid=h1s,
    d1_h2_grid=h2s,
    d1_rel_err=d1_rel,
    d1_corr=d1_corr,
    # direction 2
    d2_epochs=D2_EPOCHS,
    d2_train_curve=np.asarray(curves.train_curve),
    d2_dev_curve=np.asarray(curves.dev_curve),
    d2_train_wall_s=d2_train_wall,
    d2_first_fwd_s=first_fwd_s,
    d2_steady_fwd_s=steady_fwd_s,
    d2_first_bwd_s=first_bwd_s,
    d2_grad_ad=d2_grad_ad,
    d2_grad_fd=d2_grad_fd,
    d2_nan_grad_events=JaxDAEStep.nan_grad_events,
    d2_held_state_events=n_bad,
    d2_bad_points=bad_pts,
    d2_rms=d2_rms,
    d2_x_traj=xs,
    d2_u_traj=us,
    **policy_sd,
    # meta
    pub_wall_time_s=pub_wall,
    placeholder_flag=False,
)
print(f"wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1024:.1f} KB) "
      f"in {pub_wall/60:.1f} min total")
