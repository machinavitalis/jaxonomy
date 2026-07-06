# SPDX-License-Identifier: MIT

"""CLI helper for inspecting and applying variant configurations (T-111 phase 3).

Two subcommands under ``jaxonomy variants``:

* ``list`` — print every named variant in a built diagram alongside
  its currently-active choice and the full list of available choices.
* ``apply`` — read a variant-config JSON file produced by
  ``dump_variant_config_to_json`` (or written by hand) and emit the
  re-bound config back to stdout (or a destination file) after
  validating it against the diagram.

The CLI needs a Python diagram to operate on; users point it at a
zero-argument builder function via ``--builder package.module:function``
(the standard ``entry_points``-style ``module:name`` locator). The
builder is imported lazily and invoked once per CLI run, so the cost
mirrors any other Python ``import`` of the same module.

Why builder-from-Python instead of model JSON: variant choices in
``Variant.choices`` are zero-argument callables — they cannot
round-trip through the model JSON format today (the JSON only encodes
the *binding*, see T-111 phase 2). The CLI is a stop-gap that lets
release pipelines and reproducibility manifests capture / replay
configurations without spinning up a notebook.
"""

from __future__ import annotations

import importlib
import json
import sys

import click

from jaxonomy.framework.variants import (
    dump_variant_config,
    dump_variant_config_to_json,
    list_variants,
    load_variant_config_from_json,
)


def _load_builder(builder_spec: str):
    """Resolve ``module.path:attr`` into a callable.

    Mirrors the conventions of ``console_scripts`` entry points and
    ``setuptools``-style spec strings. Raises ``click.ClickException``
    (which Click renders as a 1-line CLI error) on import failure.
    """
    if ":" not in builder_spec:
        raise click.ClickException(
            f"--builder must be of the form 'package.module:function'; "
            f"got {builder_spec!r}."
        )
    module_path, _, attr_name = builder_spec.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise click.ClickException(
            f"--builder: failed to import module {module_path!r}: {exc}"
        ) from exc
    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:
        raise click.ClickException(
            f"--builder: module {module_path!r} has no attribute {attr_name!r}."
        ) from exc
    if not callable(attr):
        raise click.ClickException(
            f"--builder: {builder_spec!r} resolved to a {type(attr).__name__}, "
            f"not a callable."
        )
    return attr


@click.group(name="variants", help="Inspect and apply variant configurations on a built diagram.")
def jaxonomy_variants() -> None:
    """``jaxonomy variants`` — see subcommands."""


@jaxonomy_variants.command(name="list", help="List every named variant in the diagram and its active choice.")
@click.option(
    "--builder",
    required=True,
    help="Zero-arg diagram builder, in 'package.module:function' form.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format. 'text' is human-readable; 'json' is "
    "machine-readable (a list of {name, choices, active} objects).",
)
def variants_list(builder: str, output_format: str) -> None:
    """List every named variant in the diagram and its active choice."""
    builder_fn = _load_builder(builder)
    diagram = builder_fn()
    rows = list_variants(diagram)

    if output_format.lower() == "json":
        payload = [
            {"name": name, "choices": list(choices), "active": active}
            for name, choices, active in rows
        ]
        click.echo(json.dumps(payload, indent=2, sort_keys=False))
        return

    if not rows:
        click.echo("<no named variants in diagram>")
        return
    for name, choices, active in rows:
        label = name if name is not None else "<anonymous>"
        click.echo(f"{label}: active={active!r}  choices={list(choices)!r}")


@jaxonomy_variants.command(
    name="dump",
    help="Dump the current variant configuration of the diagram as JSON.",
)
@click.option(
    "--builder",
    required=True,
    help="Zero-arg diagram builder, in 'package.module:function' form.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    default=None,
    help="Output file (defaults to stdout).",
)
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    help="JSON indent. Pass 0 (or negative) for the compact one-line form.",
)
def variants_dump(builder: str, output_path: str | None, indent: int) -> None:
    """Dump ``{variant_name: active_choice}`` for the diagram."""
    builder_fn = _load_builder(builder)
    diagram = builder_fn()
    indent_arg = indent if indent and indent > 0 else None
    payload = dump_variant_config_to_json(diagram, indent=indent_arg)
    if output_path is None:
        click.echo(payload)
    else:
        with open(output_path, "w") as f:
            f.write(payload)
            if not payload.endswith("\n"):
                f.write("\n")


@jaxonomy_variants.command(
    name="apply",
    help="Apply a variant-config JSON file to the diagram and emit the resulting config (validation step).",
)
@click.option(
    "--builder",
    required=True,
    help="Zero-arg diagram builder, in 'package.module:function' form.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to a variant-config JSON file (matching the dump format).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    default=None,
    help="Where to write the re-bound config (defaults to stdout).",
)
def variants_apply(builder: str, config_path: str, output_path: str | None) -> None:
    """Validate a variant-config file against the diagram.

    Reads the JSON config, applies it to the freshly-built diagram via
    :func:`load_variant_config_from_json` (raising on unknown variant
    names or invalid choices), then dumps the resulting binding. This
    is a check-and-canonicalize pass for CI / release pipelines.
    """
    builder_fn = _load_builder(builder)
    diagram = builder_fn()
    with open(config_path, "r") as f:
        json_str = f.read()
    try:
        applied = load_variant_config_from_json(diagram, json_str)
    except Exception as exc:
        raise click.ClickException(f"variants apply: {exc}") from exc
    payload = json.dumps(dump_variant_config(applied), indent=2, sort_keys=True)
    if output_path is None:
        click.echo(payload)
    else:
        with open(output_path, "w") as f:
            f.write(payload)
            if not payload.endswith("\n"):
                f.write("\n")


if __name__ == "__main__":  # pragma: no cover - manual CLI invocation
    jaxonomy_variants()
