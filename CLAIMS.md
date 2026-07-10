# Claims

**Internal document — not published.** Maps every claim made in a shippable surface
(`README.md`, `docs/**`, `examples/**`, `benchmarks/**`, root-level
`*.ipynb`) to the specific evidence that substantiates it. The full
list of shippable surfaces and the discipline that governs them lives
in `AGENTS/RULES.md` — this file is the evidence ledger that rule
relies on.

## The rule

Two layers of discipline govern shippable-surface claims:

- **New or modified claims** get a row here: point it at an evidence
  file, run the evidence, confirm green. A row with no green evidence
  means the claim comes down.
- **Pre-existing claims** not yet in the table are covered by the
  [adversarial-review pass in AGENTS/RULES.md](AGENTS/RULES.md) until a
  shippable-surface edit touches them — that edit is the moment to
  register the claim with a row. Backfill is opportunistic.

The pre-publication checklist below applies to *registered* claims.

This is the discipline that separates a real engineering project from a
vibe-coded one. Readers don't have to trust us; they can check.

This file is public, alongside its inverse `KNOWN_GAPS.md`: CLAIMS records
what we've proven and the evidence behind it; KNOWN_GAPS records what we
haven't proven yet. Both ship.

---

## Format

| ID | Claim (as written) | Where it appears | Evidence file | Status |
|---|---|---|---|---|
| C-NNN | [exact wording of the claim] | [path or URL] | [path to test/benchmark/log] | green / yellow / red |

**Status meaning:**

- **green** — evidence file exists, runs in CI, passes. Claim is
  publishable.
- **yellow** — evidence file exists but isn't yet in CI, or runs
  locally but not yet committed. Acceptable for short windows only.
- **red** — no evidence yet, evidence is failing, or claim drifted
  from what evidence shows. **Claim must come down within 24 hours.**

---

## Active claims

| ID | Claim (as written) | Where it appears | Evidence file | Status |
|---|---|---|---|---|
| C-002 | DPC training reduces the two-tank tracking loss ~1570x (59.97 → 0.038) and bottom-level tracking RMS 12.8x (1.271 m → 0.099 m), trained policy settles within 24 mm of a 0.6 m setpoint. | `docs/examples/dpc_two_tank_reference_tracking.ipynb` (training + result cells) | same notebook — executed outputs (cells run live, no checkpoint) | green |
| C-003 | `jax.grad` of a terminal cost flows through `simulate_closed_loop` and matches central finite differences to relative error ~5.9e-6. | `docs/examples/dpc_two_tank_reference_tracking.ipynb` (AD-vs-FD cell) | same notebook cell + `test/control/test_t_040_diagram.py::test_gradient_through_simulate_matches_fd` | green |
| C-004 | The trained DPC policy recovers the analytic steady-state command u* = (c2/kp)√r (0.461 vs 0.452 at r=0.6; 0.407 vs 0.369 at r=0.4). | `docs/examples/dpc_two_tank_reference_tracking.ipynb` (steady-state validation cell) | same notebook — executed outputs | green |

*Pre-existing claims are covered by the adversarial-review pass in
`AGENTS/RULES.md`; rows get added here as shippable-surface edits
backfill them.*

Backfill backlog — high-value claims to register first:

- Differentiability claims in `README.md` → tests under `test/autodiff/`
- Solver correctness claims → `test/integrators/` and `validation/`
- Acausal compiler claims → `test/acausal/`
- Cross-tool agreement with `python-control` → tests guarded by
  `pytest.importorskip("control")`
- Block library count ("150+ library blocks" in `README.md`) → audit
  `jaxonomy/library/` against `jaxonomy.library.__all__`. Live count
  is 151 LeafSystem/Diagram subclasses — keep the
  README number conservative relative to the live count.

---

## Pre-publication checklist (for *registered* claims)

Before a shippable-surface change ships, for the subset of claims that
have rows here:

1. Every new claim added by the change has a row in this file.
2. Every existing *registered* claim the change touches has been
   re-verified (evidence re-run green).
3. No row is in red status.
4. Yellow rows have a written plan to reach green within a defined
   window.
5. The corresponding `KNOWN_GAPS.md` entries are updated for anything
   no longer claimed.

If any of the above fails on a registered row, the surface does not
ship. Un-registered pre-existing claims fall back to the
adversarial-review pass in `AGENTS/RULES.md`.

---

## Why this file exists

Two reasons:

1. **Self-discipline.** The failure mode is that claims drift from
   reality without anyone noticing. This file is the forcing function.
   If a row can't be filled, the claim can't be made.

2. **External readiness.** When the conversation gets serious with a
   downstream consumer, an evaluator, or a partner, the question "how
   do you know that's true?" will be asked. This file is the answer
   and the artifact that demonstrates the engineering discipline
   behind it.
