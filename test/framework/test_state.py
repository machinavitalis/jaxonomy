# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest

import jax
import jax.numpy as jnp

from collimator.framework import LeafState

pytest.mark.minimal


@pytest.fixture
def x():
    return jnp.array([1.0, 2.0, 3.0])


@pytest.fixture
def y():
    return jnp.array([4.0, 5.0])


@pytest.fixture
def z():
    return jnp.array([0.1, 0.2, 0.3, 0.4])


@jax.jit
def f(x):
    return jnp.exp(jnp.cos(x))


@jax.jit
def f_state(state: LeafState) -> LeafState:
    return state.with_continuous_state(f(state.continuous_state))


class TestLeafState:
    def test_init_xc(self, x):
        state = LeafState(name="x", continuous_state=x)
        assert state.name == "x"

        assert jnp.allclose(state.continuous_state, x)
        assert state.continuous_state.size == 3

        assert state.discrete_state is None

    def test_set_xc(self, x):
        state = LeafState(name="x", continuous_state=x)
        y = 2.5 * x
        state = state.with_continuous_state(y)

        assert jnp.allclose(state.continuous_state, y)

    def test_function_update_leaf(self, x):
        state = LeafState(name="x", continuous_state=x)
        state = f_state(state)

        assert jnp.allclose(state.continuous_state, f(x))
