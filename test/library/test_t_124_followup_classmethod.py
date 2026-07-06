# SPDX-License-Identifier: MIT

"""T-124-followup-classmethod-on-LookupTable1d.

Verifies the ergonomic ``LookupTable1d.fit_from_data`` classmethod is a
faithful pure-delegation wrapper around
:func:`jaxonomy.library.fit_lookup_table_1d`.  No behaviour change vs.
the standalone helper — these tests just pin the parity contract so a
future refactor doesn't silently drift.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import LookupTable1d, fit_lookup_table_1d


class TestClassmethodReturnsBlock:
    def test_returns_lookup_table_1d_instance(self):
        xp = jnp.linspace(0.0, 10.0, 6)
        x_data = jnp.linspace(0.0, 10.0, 51)
        y_data = 2.0 * x_data + 3.0
        block = LookupTable1d.fit_from_data(xp, x_data, y_data)
        assert isinstance(block, LookupTable1d)


class TestParityWithStandaloneHelper:
    """The classmethod must return a block whose fitted output matches the
    standalone helper bit-for-bit (it's pure delegation)."""

    def test_default_kwargs_match(self):
        xp = jnp.linspace(0.0, 10.0, 11)
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 2.0 * x_data + 3.0
        block_cls = LookupTable1d.fit_from_data(xp, x_data, y_data)
        block_fn = fit_lookup_table_1d(xp, x_data, y_data)
        # Build both blocks into tiny diagrams (so ``initialize`` runs and
        # populates ``output_array``) and compare the fitted tables.
        for block in (block_cls, block_fn):
            builder = jaxonomy.DiagramBuilder()
            b = builder.add(block)
            src = builder.add(library.Constant(0.0))
            builder.connect(src.output_ports[0], b.input_ports[0])
            builder.build().create_context()
        assert jnp.allclose(
            block_cls.output_array, block_fn.output_array, atol=0.0
        )
        assert jnp.allclose(
            block_cls.input_array, block_fn.input_array, atol=0.0
        )

    def test_smoothness_kwarg_forwarded(self):
        xp = jnp.linspace(0.0, 1.0, 9)
        x_data = jnp.linspace(0.05, 0.95, 100)
        y_data = jnp.sin(2.0 * math.pi * x_data)
        block_cls = LookupTable1d.fit_from_data(
            xp, x_data, y_data, smoothness=0.5
        )
        block_fn = fit_lookup_table_1d(
            xp, x_data, y_data, smoothness=0.5
        )
        # Initialize so output_array is populated.
        for block in (block_cls, block_fn):
            builder = jaxonomy.DiagramBuilder()
            b = builder.add(block)
            src = builder.add(library.Constant(0.5))
            builder.connect(src.output_ports[0], b.input_ports[0])
            builder.build().create_context()
        assert jnp.allclose(
            block_cls.output_array, block_fn.output_array, atol=0.0
        )

    def test_weights_kwarg_forwarded(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        x_truth = jnp.linspace(0.05, 0.95, 50)
        y_truth = 2.0 * x_truth
        x_junk = jnp.linspace(0.05, 0.95, 50)
        y_junk = jnp.full_like(x_junk, -50.0)
        x_data = jnp.concatenate([x_truth, x_junk])
        y_data = jnp.concatenate([y_truth, y_junk])
        weights = jnp.concatenate([jnp.ones(50), jnp.zeros(50)])
        block_cls = LookupTable1d.fit_from_data(
            xp, x_data, y_data, weights=weights
        )
        block_fn = fit_lookup_table_1d(
            xp, x_data, y_data, weights=weights
        )
        for block in (block_cls, block_fn):
            builder = jaxonomy.DiagramBuilder()
            b = builder.add(block)
            src = builder.add(library.Constant(0.0))
            builder.connect(src.output_ports[0], b.input_ports[0])
            builder.build().create_context()
        assert jnp.allclose(
            block_cls.output_array, block_fn.output_array, atol=0.0
        )

    def test_block_kwargs_forwarded(self):
        # Forward ``interpolation=`` and ``name=`` through to the
        # constructor.
        xp = jnp.linspace(0.0, math.pi, 9)
        x_data = jnp.linspace(0.0, math.pi, 201)
        y_data = jnp.sin(x_data)
        block = LookupTable1d.fit_from_data(
            xp, x_data, y_data, interpolation="pchip", name="fit_table"
        )
        assert block.name == "fit_table"


class TestFittedBlockIntegratesInDiagram:
    def test_fitted_block_evaluates_in_diagram(self):
        xp = jnp.linspace(0.0, 10.0, 11)
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 2.0 * x_data + 3.0
        block = LookupTable1d.fit_from_data(xp, x_data, y_data)

        builder = jaxonomy.DiagramBuilder()
        b = builder.add(block)
        src = builder.add(library.Constant(5.5))
        builder.connect(src.output_ports[0], b.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        out = b.output_ports[0].eval(ctx)
        assert float(out) == pytest.approx(2.0 * 5.5 + 3.0, abs=1e-4)

    def test_fitted_block_simulates(self):
        # Drive the fitted block with a constant input and run a tiny
        # simulation — exercises the full block lifecycle (build,
        # context, simulate).
        xp = jnp.linspace(0.0, 10.0, 11)
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 3.0 * x_data
        block = LookupTable1d.fit_from_data(xp, x_data, y_data)

        builder = jaxonomy.DiagramBuilder()
        b = builder.add(block)
        src = builder.add(library.Constant(2.0))
        builder.connect(src.output_ports[0], b.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(diagram, ctx, (0.0, 0.1))
        # The block is feedthrough — final output reflects current input.
        out = b.output_ports[0].eval(results.context)
        assert float(out) == pytest.approx(3.0 * 2.0, abs=1e-4)
