# RULES.md

> Operating principles and project rules for Jaxonomy. The cheat-sheet
> version of these rules lives in `/CLAUDE.md`; this file is the
> standard. Read this once when you start working on the project; the
> cheat sheet is enough day-to-day.

---

## Four operating principles

1. **Think before coding.** If the task has three or more steps, write
   a plan first. Surface assumptions out loud. If two interpretations
   are possible, ask. Don't silently pick one and execute.
2. **Simplest implementation that fits.** If 200 lines could be 50,
   write 50. No speculative flexibility, no abstractions invented for
   single-use code, no "future-proofing" by adding parameters nobody
   asked for.
3. **Surgical changes only.** Touch only what the task requires. Don't
   reformat adjacent code, don't "improve" nearby comments, don't
   refactor what isn't broken, don't rename existing symbols unless
   asked. Match existing style.
4. **Define success, then loop.** Before writing code, write the
   verification: a test, a benchmark, a script that exercises the
   behavior. Loop until it passes. Don't mark a task done without
   proof.

---

## Shippable-surface rule

Files in these paths are **shippable surfaces**. The world will read
them. Everything in them must be real, accurate, and defensible under
skeptical public scrutiny.

- `README.md`, `CHANGELOG.md`, `docs/**`, `examples/**`
- `benchmarks/**` (results pages, evidence)
- `*.ipynb` in repo root or `examples/`
- `CLAIMS.md`, `KNOWN_GAPS.md`

In shippable surfaces, **never**:

- write `# TODO`, `# FIXME`, `pass`, `raise NotImplementedError`, or
  placeholder values
- fabricate benchmark numbers, timing claims, or success rates
- invent customer names, design partners, or testimonials
- use lorem ipsum, fake URLs, or example.com addresses
- write API examples for functions that don't exist or don't behave as
  written
- write code that wouldn't actually run if pasted into a fresh
  environment

If something can't be implemented or measured right now, **the surface
that depends on it must be removed entirely** until it can. Removed
beats fake.

In non-shippable surfaces (internal scaffolding, exploratory branches,
work-in-progress modules, `AGENTS/` files, root-level
`pytest_output.txt`), the above rules relax — stubs and placeholders
are fine if marked clearly. The distinction is load-bearing.

### Adversarial review pass

Before committing a change to a shippable surface, run a skeptical pass:

> Read this as a skeptical senior engineer evaluating an open-source
> simulation library for production use. Find anything that looks like
> a placeholder, would not survive skeptical public scrutiny, contains
> synthetic data passed off as real, or makes a claim not substantiated
> by the code. Be harsh.

This default applies to shippable-surface changes only. Routine feature
work on `library/` blocks doesn't trigger it; the CHANGELOG bullet
shipped as part of that work does.

---

## Claims discipline

Every claim in a shippable surface should map to a row in `/CLAIMS.md`
backed by an evidence file (test, benchmark, measured run, validation
log); a claim with no evidence comes down.

- **New or modified claims require a row + evidence.** When you add or
  change a claim, add the row in `CLAIMS.md`, add or update the
  evidence file, run it locally, confirm green.
- **Pre-existing claims** not yet registered are covered by the
  [adversarial-review pass](#adversarial-review-pass) below until a
  shippable-surface edit touches them — that edit is the moment to
  backfill the row.

If you find yourself unable to substantiate a claim, **delete the
claim**. Don't soften it. Don't hedge. Remove it.

## Known-gaps discipline

If something doesn't work, doesn't yet work, or only works in a limited
way, it goes in `/KNOWN_GAPS.md`. This file is intentionally public —
honesty here is a feature, not a liability. When you implement
something that closes a known gap, remove the corresponding entry.
When you discover a new gap, add one.

`KNOWN_GAPS.md` is the inverse of `CLAIMS.md`: the former lists what we
don't yet do; the latter lists what we claim with evidence.

---

## How to work on this codebase

**Plan mode is the default for anything non-trivial.** Three or more
steps, or any change to a shippable surface, starts with a written plan
(`plan-<slug>.md` in `notes/`). The plan gets reviewed and annotated
before any code is written. Pour energy into the plan; one-shot the
implementation.

**For internal-only work, just go.** Don't over-plan modules nobody
will see.

**Verification before "done."** No task is complete without proof: a
passing test, a passing benchmark, a working demo, or a measured run.
"I think this works" is not done. "Here is the test that demonstrates
it works" is done.

---

## Conventions

- **Language**: Python with JAX. The `npa` / `jnp` / `np` triplet is a
  load-bearing backend-neutrality invariant — canonical reference in
  `AGENTS/PATTERNS.md`, rationale in `AGENTS/DECISIONS.md` DEC-030.
  Keep functions pure on anything that goes through `jit`, `grad`, or
  `vmap`.
- **Style**: No formatter is currently configured (`pyproject.toml`
  has no `[tool.black]` / `[tool.ruff]`). Match existing style in the
  file you're editing; don't reformat untouched code. If a formatter
  is adopted, an ADR in `AGENTS/DECISIONS.md` is the place to record
  it.
- **Tests**: pytest. Every advertised behavior has a test. Tests live
  in `test/` (singular), mirroring the source layout. See
  `AGENTS/PATTERNS.md` for test patterns.
- **Benchmarks**: separate from tests. Live in `benchmarks/`. Produce
  committed result files. The result file is the artifact, not the
  script.
- **Docs**: MkDocs site in `docs/`; the README is the front door;
  everything else is reachable in two clicks.
- **Dependencies**: minimize. Each new dependency is a liability and
  warrants a DECISIONS.md entry (see DEC-031 for the FMU-export
  precedent of choosing between three viable libraries).
- **Files**: one concept per file. If a file passes ~500 lines, ask
  whether it should be split — but the threshold is a heuristic, not a
  tripwire. The `primitives.py → sources/math_ops/logic/
  routing/dynamics/nonlinearities/tables` split was topic-driven, not
  line-count-driven. Don't refactor mid-task to chase the threshold.

---

## What we don't want

- *New* magic configuration systems, plugin architectures, or registry
  patterns that hide what's executing. Existing ones (`MathDispatcher`,
  `Variants`, FMU dispatchers) are load-bearing and stay — adding a new
  one requires an ADR.
- New frameworks invented inside this codebase. Build on JAX primitives
  and standard tools.
- "Helper" abstractions for things used once.
- Premature optimization. Profile, then optimize, then verify the
  optimization helped.
- Generic error messages. Errors should tell the caller what went wrong
  and what to do about it. See DEC-024 (`ErrorCollector`) for the
  pattern.
- Long docstrings that re-state what the code obviously does.
  Docstrings should explain *why*, edge cases, and gotchas.
- Drive-by changes outside the task's scope. If you discover an
  unrelated bug, surface it to the maintainer (commit message / PR)
  rather than fixing it on the current branch.

---

## Self-improvement loop

After any correction or mistake, propose a new rule for this file (or
a tightening of an existing one). Surface the proposal to the user
before editing — these rules affect every future session.

The proposed rule should be:

- specific (about the actual failure mode, not the category)
- short (one sentence preferred)
- placed in the most relevant section

If a rule is ignored repeatedly, the rule is probably wrong — flag it
for a rewrite rather than reinforcing it. Don't compound the friction.

Code-path-specific dev gotchas (a JAX trace surprise on a specific
function, a subtle precision interaction with a specific solver) go
in the affected code's docstring tagged with the followup task ID,
not in this file. Rules are general; code-path gotchas are local and
travel with the code.

---

## When in doubt

Ask. The cost of a clarifying question is one round-trip; the cost of
building the wrong thing is much higher. If the request is ambiguous,
surface the ambiguity before writing any code. For the full
escalation list (intent ambiguity, ADR-worthy decisions, breaking API
changes, simulation-correctness invariants, irrecoverable test
failures), see `AGENTS/README.md`'s "Autonomy and escalation" section.
