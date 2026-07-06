# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately via GitHub's **Report a vulnerability** flow:
**Security → Advisories → Report a vulnerability** on this repository
(`https://github.com/machinavitalis/jaxonomy/security/advisories/new`).

Please include enough detail to reproduce: affected version/commit, a minimal
proof of concept, and the impact you observed. We aim to acknowledge a report
within a few business days and will coordinate a fix and disclosure timeline
with you.

## Supported versions

Security fixes target the latest `main` and the most recent tagged release.
Older tags are not maintained.

## Scope notes for users

A few behaviors are insecure **by design** if misused — they are not
vulnerabilities, but know what you're running:

- **Loading untrusted models.** Deserializing a model from JSON reconstructs
  blocks by name and can execute code paths defined by the model. Treat a model
  file like code: only load models from sources you trust.
- **External function blocks** (FMU import via `fmpy`, ONNX/host-callback
  blocks) execute third-party binaries/graphs. Only run FMUs and models you
  trust.

If you find a way to escalate beyond these documented behaviors (e.g. code
execution from a path documented as safe), that *is* a vulnerability — please
report it.
