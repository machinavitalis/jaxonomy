# AGENTS/

Orientation and operating manual for AI coding agents (Claude Code, Cursor, and
similar) and human contributors working on Jaxonomy. Fresh session: read this
file first, then follow the session protocol below. The structure is
tool-agnostic — it works for any agent and for humans.

## Files in this directory

Knowing which file to consult vs. update is half the discipline.

- **CONTEXT.md** — what Jaxonomy is, design philosophy, architecture, key
  abstractions, invariants. The orientation document; read before writing code.
- **PATTERNS.md** — coding conventions (`LeafSystem` callbacks, NamedTuple
  state, `npa` vs `jnp`, naming, imports, test patterns). Consult before any
  non-trivial code.
- **DECISIONS.md** — Architectural Decision Records. Check before re-litigating
  a design choice; append an ADR when you settle one future sessions shouldn't
  re-debate.
- **RULES.md** — operating principles, shippable-surface rule, claims/gaps
  discipline, self-improvement loop. Read once; the cheat sheet in `/CLAUDE.md`
  is enough day-to-day.

(`CHANGELOG.md` at repo root is the user-visible delta against CONTEXT.md —
append a bullet when user-visible work lands on `main`.)

## Session protocol

1. **Inspect repo state** — `git status`, `git log --oneline -20`, current
   branch. Red tests on the branch are your starting hole; don't ignore them.
2. **Read CONTEXT.md + PATTERNS.md before writing code.**
3. **Check DECISIONS.md before architectural choices** — new dependency, public
   API change, module restructure, picking between approaches. If settled,
   follow it; if not, consider whether your choice deserves an ADR.
4. **End green and committed.** Run the suite, commit in-flight work. Anything
   needing human eyes goes in a commit message or PR description — not a local
   scratchpad file.
5. **Capture dev-side code-path gotchas** in a followup-tagged docstring on the
   affected code before ending the session.

## Operating rules

### Branching and commits

- One task per branch (`task/T###-short-title`), branched from `main`, merged
  back when acceptance criteria pass and the full suite is green.
- Commit at every acceptance criterion, not at the end — small atomic commits
  bisect and review better.
- Don't merge red. Fix it in-session or flag it in the commit message.
- Update CHANGELOG.md `[Unreleased]` when merging user-visible work.

### Scope discipline

- Stay in scope. An unrelated bug or improvement found mid-task is surfaced to
  the maintainer (commit message / PR), not a drive-by fix on your branch.
- Stop and ask if scope is unclear or acceptance criteria don't match intent.

### Worktrees

Each worktree's AGENTS/ files reflect *its branch's* state.

### Autonomy and escalation

Work autonomously. Don't pause for approval on routine work — implementing
specs, fixing bugs, refactoring within a module, adding tests, updating docs,
merging completed branches when acceptance passes. Escalate only for:

- **Scope or intent ambiguity** you can't resolve from context, or a spec that
  contradicts CONTEXT.md / DECISIONS.md.
- **ADR-worthy decisions** — new dependency, public-API change, module
  restructure, a tradeoff future sessions shouldn't re-debate. Write the ADR
  proposal and flag it before merging the code.
- **Changes to the AGENTS/ files themselves** — institutional memory; propose,
  don't merge unilaterally.
- **Backward-incompatible public API changes** — even if the spec implies them.
- **Discoveries that materially change the task** — significantly bigger,
  smaller, or different than the spec.
- **Anything touching simulation-correctness invariants** — event handling,
  gradient flow, determinism, vmap safety, numerical precision.
- **Test failures you can't resolve in-session.**

Rule of thumb: "would a reasonable project owner want a heads-up before this
lands?" If yes, surface it; if it's a routine fix/test/refactor/doc update,
just merge.

**How to escalate.** Prefix the commit subject (or PR title) with `[NEEDS HUMAN
INPUT]` and use the body to state the question, the options, your
recommendation, and what you're blocked on. Use the maintainer-visible surface
(`git log`, PRs), not a per-worktree local file. Don't block on a synchronous
reply — continue with independent work.

## When you change AGENTS/ itself

These files are institutional memory. Edits to CONTEXT/PATTERNS/DECISIONS/RULES
or this README are small focused PRs to `main`, not bundled with feature work.
Add an ADR alongside the work that prompts it. New PATTERNS reflect actual code
(include the refactor that establishes them), not aspiration.

## What this directory is not

Not a wiki or historical archive; not user documentation (that's `docs/`); not
strategy/commercial planning (outside this repo); not a Jira or
project-management system.

## Quick reference

| Task | File |
|---|---|
| Start of session | `/CLAUDE.md` → `git status` / `git log` |
| Before writing code | CONTEXT.md, PATTERNS.md |
| Before touching a shippable surface | RULES.md, `/CLAIMS.md`, `/KNOWN_GAPS.md` |
| Before an architectural decision | DECISIONS.md |
| Dev gotcha in a specific code path | docstring on the code, tagged with the followup task ID |
| Made a design choice | DECISIONS.md (ADR) |
| Established a new convention | PATTERNS.md |
| Repeated correction → new rule | RULES.md (self-improvement loop) |
| Escalate to the human | `[NEEDS HUMAN INPUT]` commit/PR prefix + body |
| End of session | commit; CHANGELOG bullet if user-visible |
