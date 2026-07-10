# AGENTS.md — Jaxonomy

Agent bootstrap and operational notes, for any coding agent (Claude Code,
Codex, Gemini, Cursor, …) and human contributors. This is the **canonical,
tool-neutral entry file**; `CLAUDE.md`, `GEMINI.md`,
`.github/copilot-instructions.md`, and `CONVENTIONS.md` are symlinks to it, and
`.cursor/rules/` points here (see "Entry points" at the end).

Two doors, depending on what you're here to do:

- **Modifying / adding code in Jaxonomy** → follow the read order below,
  starting at `AGENTS/README.md`. Full project orientation lives in `AGENTS/`;
  come back here for the things that specifically save agent time.
- **Using Jaxonomy's public API** in your own code (authoring a tutorial,
  building a demo, writing a downstream library) → read `SKILL.md`, the
  consumer operating manual, instead.

## Read first

1. `AGENTS/README.md` — navigation + which AGENTS file to use when.
2. `AGENTS/CONTEXT.md` — what Jaxonomy is, design philosophy, key abstractions.
3. `AGENTS/PATTERNS.md` — coding conventions (`npa` vs `jnp`, `LeafSystem`
   callback signatures, NamedTuple state, test patterns, naming).
4. `AGENTS/DECISIONS.md` — ADRs; check before re-litigating a settled choice.
5. `AGENTS/RULES.md` — operating principles, shippable-surface rule, claims/gaps
   discipline, self-improvement loop.

For a pure usage session (author a tutorial, build a demo, exercise the public
API), `SKILL.md` is the better starting point.

## Operating discipline (pointers, not a second copy)

The substance lives in two files; this bootstrap defers to them rather than
restating them:

- **`AGENTS/RULES.md`** — the four operating principles (think before coding;
  simplest implementation that fits; surgical changes only; define success then
  loop), the shippable-surface rule + adversarial-review pass, claims/gaps
  discipline, and the self-improvement loop. Read it once.
- **`AGENTS/README.md`** — session protocol, branching/commits, scope
  discipline, and the autonomy/escalation list.

Two reminders that bite most often in an agent session: changes to a
shippable surface (`README.md`, `docs/**`, `examples/**`, `benchmarks/**`, root
`*.ipynb`, `CLAIMS.md`, `KNOWN_GAPS.md`) must be real and evidence-backed —
removed beats fake; and an unrelated bug found mid-task is surfaced to the
maintainer (commit message / PR), not fixed as a drive-by on your branch.

## Where things actually live (non-obvious)

| Looking for                                  | File                                                      |
|----------------------------------------------|-----------------------------------------------------------|
| `findop`, `frequency_response`, `bode_data`, `nyquist_data`, `pole_zero_map`, `step_response`, `impulse_response`, `estimate_frequency_response` | `jaxonomy/library/linearization_workflow.py` (**not** `jaxonomy/optimization/` despite the name) |
| `linearize`, `LinearizedSystem`, `LTISystem`, `TransferFunction`, `PID` | `jaxonomy/library/linear_system.py`                       |
| Standard-library blocks — split by category (was `primitives.py` until the refactor) | `library/sources.py` (sources + stochastic), `library/math_ops.py`, `library/logic.py`, `library/routing.py` (mux/demux/buses), `library/dynamics.py` (integrators + discrete state + filters + PID), `library/nonlinearities.py` (saturate / dead zone / rate limiter / quantizer), `library/tables.py` (lookup family). `library/primitives.py` is a re-export hub for back-compat — `from .primitives import X` keeps working. |
| Container blocks (`EnabledSubsystem`, `TriggeredSubsystem`, `ForEach`) | `jaxonomy/framework/containers.py`                        |
| Unit annotations (`BusUnit`)                 | `jaxonomy/framework/units.py`                             |
| Variants                                     | `jaxonomy/framework/variants.py`                          |
| Lookup-table fitting (`fit_lookup_table_*`)  | `jaxonomy/library/lookup_table_fitting.py`                |
| UQ workflow (Monte Carlo, Sobol, LHS, qMC)   | `jaxonomy/uq/`                                            |
| Diagnostics (dead-store, empty-inputs)       | `jaxonomy/diagnostics.py`                                 |
| Parameter tuning helpers                     | `jaxonomy/optimization/parameter_tuning.py`               |
| Lazy results + DuckDB backend                | `jaxonomy/simulation/lazy_results.py`                     |
| Event-time gradient infrastructure           | `jaxonomy/simulation/event_gradient.py`                   |
| Fast restart                                 | `jaxonomy/simulation/fast_restart.py`                     |
| Provenance manifest                          | `jaxonomy/simulation/provenance.py`                       |

## Testing

- Fast tier: `pytest -m "not slow"`. Use this for regression checks on
  routine work.
- Tests live in `test/` (singular), mirroring the source layout.
- **Baseline: fully green.** A full-suite run (2026-07-09, all 5109 tests,
  fast + slow tiers) had zero unexpected failures — the previously listed
  baseline failures (Kalman filters, state-machine dtype, random
  normal/gamma, fluid, `battery_cell`, `edge_detection_comparator`) all
  pass now. `test_predictor.py` Torch/TF tests *skip* when
  torch/tensorflow aren't installed (currently absent locally). A new
  failure on your branch is therefore yours to explain.
- `pytest.ini` sets a global `--timeout=180` per test via pytest-timeout,
  and its default `addopts` deselect `slow`/`dashboard`/`autodiff_full` —
  pass `-m slow` explicitly to run slow tests. Genuinely long tests are
  marked `@pytest.mark.slow` and, if they can exceed ~180s, carry a
  per-test `@pytest.mark.timeout(N)` override (the global cap applies to
  slow-marked tests too).
- Optional cross-tool deps: `python-control 0.10.x` is available locally
  for SLICOT cross-validation; wrap such tests in `pytest.importorskip("control")`.

## Git conventions

- **Agent-driven work goes on a branch, then merges to `main` when
  acceptance passes.** Worktree sessions inherit a `claude/*` branch;
  longer task work uses `task/T###-short-title`. Direct commits to
  `main` are the maintainer's path for their own infrastructure tweaks.
  Full branching + merge protocol is in `AGENTS/README.md`.
- Infrastructure tweaks (edits to this bootstrap or the `AGENTS/*` docs,
  rule changes) stay uncommitted unless explicitly bundled with feature work.
- Don't `git add -A`; stage specific files. Worktrees occasionally carry
  long-standing untracked drafts — leave anything you don't recognize
  alone.

## Codebase conventions worth knowing

- **`npa` vs `jnp` vs `np` triplet** is a load-bearing
  backend-neutrality invariant. The canonical explanation lives in
  `AGENTS/PATTERNS.md` (JAX Patterns → Backend abstraction); the
  rationale is in `AGENTS/DECISIONS.md` DEC-030. Read those once.
- Helpers that return matplotlib-ready arrays (e.g. `bode_data`,
  `nyquist_data`) return plain dicts — no matplotlib import inside
  `jaxonomy`.
- New analytical helpers on `LinearizedSystem` live in
  `library/linearization_workflow.py`, not a new `analysis/` directory.

## Entry points (why there are several files)

The same content is reachable under every AI tool's expected filename, with no
duplication:

- **`AGENTS.md`** (this file) — the one real, canonical bootstrap. Read
  natively by Codex/ChatGPT and by humans.
- **`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`,
  `CONVENTIONS.md`** — symlinks to this file (Claude Code, Gemini CLI, GitHub
  Copilot, Aider). Edit `AGENTS.md`; the rest follow automatically.
- **`.cursor/rules/`** — a one-line rule pointing Cursor here.
- **`SKILL.md`** — the *consumer* manual (using the API), a separate document;
  it is also surfaced as a Claude skill at `.claude/skills/jaxonomy/SKILL.md`,
  with root `SKILL.md` a symlink into it.
