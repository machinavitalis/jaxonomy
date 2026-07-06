# Third-Party Notices

Jaxonomy is licensed under the [MIT License](LICENSE.md). It **depends on** the
third-party packages below (declared in `requirements.in`) but does not bundle
or redistribute their source. Each is governed by its own license; install-time
dependencies remain under their respective terms.

All runtime dependencies are under permissive, MIT-compatible licenses
(Apache-2.0, BSD, or MIT family). **No copyleft (GPL/LGPL/AGPL) dependency is
present**, so there is no license-compatibility conflict with Jaxonomy's MIT
license.

> License identifiers below are provided for convenience and reflect each
> project's stated license to the best of our knowledge. The authoritative
> license is always the one published by the upstream project — verify there if
> it matters for your use.

## Runtime dependencies

| Package | License (typical SPDX) |
|---|---|
| jax, jaxlib | Apache-2.0 |
| diffrax | Apache-2.0 |
| equinox | Apache-2.0 |
| jaxtyping | Apache-2.0 |
| optax | Apache-2.0 |
| brax | Apache-2.0 |
| jaxopt | Apache-2.0 |
| evosax | Apache-2.0 |
| requests | Apache-2.0 |
| opencv-python | MIT (wrapper) / Apache-2.0 (OpenCV) |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| networkx | BSD-3-Clause |
| sympy | BSD-3-Clause |
| click | BSD-3-Clause |
| cloudpickle | BSD-3-Clause |
| fmpy | BSD-2-Clause |
| dataclasses-json | MIT |
| dataclasses-jsonschema | MIT |
| simpleeval | MIT |
| ts-type | MIT |
| StrEnum | MIT |

## Provenance

Jaxonomy is derived from the MIT-licensed open-source package **pycollimator**
by **Collimator, Inc.** (see [README.md](README.md) and [LICENSE.md](LICENSE.md)).
