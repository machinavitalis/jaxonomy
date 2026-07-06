# SPDX-License-Identifier: MIT
"""T-036e: Custom*Block port-name diagnostics.

`CustomJaxBlock` and `CustomPythonBlock` use `init_script` /
`user_statements` strings (compiled + `exec`'d), so the user surface is
the `inputs` / `outputs` port-name lists rather than a callable
signature. This file pins six concrete opaque-error paths that are now
caught eagerly with a clear ``BlockParameterError`` naming the block
and the offending entry.
"""

from __future__ import annotations

import pytest

from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library import CustomJaxBlock, CustomPythonBlock


# Run each case against both block subclasses (CustomPythonBlock inherits
# CustomJaxBlock.__init__, so the validation is shared).
BLOCK_CLASSES = [CustomJaxBlock, CustomPythonBlock]


def _make(cls, *, inputs=None, outputs=None, name="custom_block", init_script=""):
    return cls(
        dt=0.1,
        time_mode="discrete",
        init_script=init_script,
        user_statements="",
        inputs=inputs,
        outputs=outputs,
        name=name,
    )


def _make_well_formed(cls, *, inputs, outputs, name="custom_block"):
    """Helper for sanity tests: provides an init_script that initialises
    every declared output (CustomJaxBlock requires this in discrete mode)."""
    init_lines = [f"{o} = 0.0" for o in outputs]
    return _make(cls, inputs=inputs, outputs=outputs, name=name,
                 init_script="\n".join(init_lines))


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_inputs_must_be_list(cls):
    with pytest.raises(BlockParameterError, match=r"inputs.*must.*list.*str"):
        _make(cls, inputs="x")


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_outputs_must_be_list(cls):
    with pytest.raises(BlockParameterError, match=r"outputs.*must.*list.*tuple"):
        _make(cls, outputs={"y"})


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_inputs_entries_must_be_strings(cls):
    with pytest.raises(BlockParameterError, match=r"must be strings.*int"):
        _make(cls, inputs=[42])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_outputs_entries_must_be_identifiers(cls):
    with pytest.raises(BlockParameterError, match=r"'1bad'.*not a valid Python identifier"):
        _make(cls, outputs=["1bad"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_inputs_entries_must_be_identifiers_with_spaces(cls):
    with pytest.raises(BlockParameterError, match=r"'has space'.*not a valid Python identifier"):
        _make(cls, inputs=["has space"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_inputs_entries_cannot_be_keywords(cls):
    with pytest.raises(BlockParameterError, match=r"'if'.*reserved Python keyword"):
        _make(cls, inputs=["if"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_outputs_entries_cannot_be_keywords(cls):
    with pytest.raises(BlockParameterError, match=r"'def'.*reserved Python keyword"):
        _make(cls, outputs=["def"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_inputs_must_be_unique(cls):
    with pytest.raises(BlockParameterError, match=r"duplicate.*inputs.*entry 'x'"):
        _make(cls, inputs=["x", "y", "x"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_outputs_must_be_unique(cls):
    with pytest.raises(BlockParameterError, match=r"duplicate.*outputs.*entry 'y'"):
        _make(cls, outputs=["y", "z", "y"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_input_output_collision_named(cls):
    with pytest.raises(BlockParameterError, match=r"'shared'.*both `inputs` and `outputs`"):
        _make(cls, inputs=["shared"], outputs=["shared"])


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_well_formed_custom_block_compiles(cls):
    """Sanity: a correctly-shaped block constructs without raising."""
    blk = _make_well_formed(cls, inputs=["u0", "u1"], outputs=["y0"])
    assert blk.input_names == ["u0", "u1"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_empty_io_lists_compile(cls):
    """Sanity: empty inputs and outputs are valid (a self-contained block)."""
    blk = _make(cls, inputs=[], outputs=[])
    assert blk.input_names == []


# ============================================================================
# T-036e (deeper): AST signature diagnostics — strict=True only.
# ============================================================================
#
# These checks fire at __init__ when ``strict=True`` is passed.  They
# AST-walk the init / step / finalize scripts and surface typo'd or
# wrongly-bound symbols before they become a NameError at first eval.


def _strict(cls, *, inputs, outputs, init_script="", user_statements="",
            finalize_script="", name="custom_block", **extra):
    """Helper — construct a Custom*Block in strict mode."""
    return cls(
        dt=0.1,
        time_mode="discrete",
        init_script=init_script,
        user_statements=user_statements,
        finalize_script=finalize_script,
        inputs=inputs,
        outputs=outputs,
        name=name,
        strict=True,
        **extra,
    )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_init_script_undefined_symbol_in_jax_block(cls):
    """Case 1: init_script references a typo'd symbol — fail at __init__."""
    with pytest.raises(
        BlockParameterError,
        match=r"init_script references undefined symbol 'foo'",
    ):
        _strict(
            cls,
            init_script="out_0 = foo + 1.0",
            inputs=["u"],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_init_script_undefined_symbol_lists_allowed_set(cls):
    """The error message must list inputs and outputs to help the user."""
    with pytest.raises(
        BlockParameterError,
        match=r"declared inputs=\['u'\].*outputs=\['out_0'\]",
    ):
        _strict(
            cls,
            init_script="out_0 = qux",
            inputs=["u"],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_user_statements_undefined_symbol(cls):
    """Case 2: user_statements references a typo'd symbol — fail at __init__."""
    with pytest.raises(
        BlockParameterError,
        match=r"user_statements references undefined symbol 'baz'",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            user_statements="out_0 = baz * 2",
            inputs=["u"],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_finalize_script_undefined_symbol(cls):
    """Case 2 (cont.): finalize_script catches typos too."""
    if cls.__name__ == "CustomJaxBlock":
        pytest.skip("finalize_script is only supported on CustomPythonBlock.")
    with pytest.raises(
        BlockParameterError,
        match=r"finalize_script references undefined symbol 'cleanup_fn'",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            finalize_script="cleanup_fn()",
            inputs=[],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_t_vs_time_confusion_hint(cls):
    """Case 4: ``t`` typo'd for ``time`` surfaces a fix-up suggestion."""
    with pytest.raises(
        BlockParameterError,
        match=r"did you mean 'time'",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            user_statements="out_0 = 2 * t",
            inputs=[],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_empty_inputs_with_input_reference_hint(cls):
    """Case 5: empty inputs=[] paired with a ``in_0`` reference — hint."""
    with pytest.raises(
        BlockParameterError,
        match=r"the block has inputs=\[\]; declare 'in_0'",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            user_statements="out_0 = in_0 * 2",
            inputs=[],
            outputs=["out_0"],
        )


# --- negative tests: well-formed scripts must NOT raise ----------------------


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_negative_numpy_import_no_raise(cls):
    """A well-formed script with `import numpy as np` does not raise."""
    blk = _strict(
        cls,
        init_script="import numpy as np\nout_0 = np.zeros(3)",
        user_statements="out_0 = np.sin(in_0) + np.cos(time)",
        inputs=["in_0"],
        outputs=["out_0"],
    )
    assert blk.input_names == ["in_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_negative_jax_import_no_raise(cls):
    """A well-formed script using `jax.numpy as jnp` does not raise."""
    blk = _strict(
        cls,
        init_script="import jax.numpy as jnp\nout_0 = jnp.zeros(2)",
        user_statements="out_0 = jnp.tanh(in_0) * jnp.pi",
        inputs=["in_0"],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_negative_common_math_free_names_no_raise(cls):
    """Bare ``sin``, ``cos``, ``np.zeros`` references don't trigger
    false-positives even without an explicit ``from numpy import ...``."""
    blk = _strict(
        cls,
        init_script="out_0 = 0.0",
        user_statements="out_0 = sin(time) + cos(in_0)",
        inputs=["in_0"],
        outputs=["out_0"],
    )
    assert blk.input_names == ["in_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_negative_function_def_in_init_script_no_raise(cls):
    """A function defined in init_script is visible to user_statements."""
    blk = _strict(
        cls,
        init_script="def double(x): return x * 2\nout_0 = 0.0",
        user_statements="out_0 = double(in_0)",
        inputs=["in_0"],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_negative_from_import_no_raise(cls):
    """`from numpy import sin, cos` introduces names visible to step code."""
    blk = _strict(
        cls,
        init_script="from numpy import sin, cos\nout_0 = 0.0",
        user_statements="out_0 = sin(time) + cos(in_0)",
        inputs=["in_0"],
        outputs=["out_0"],
    )
    assert blk.input_names == ["in_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_negative_comprehension_local_no_raise(cls):
    """List-comp loop variable is local — not flagged as undefined."""
    blk = _strict(
        cls,
        init_script="out_0 = [i * 2 for i in range(3)][0]",
        inputs=[],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


# --- backwards compatibility: strict=False is the default --------------------


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_strict_off_by_default_does_not_raise(cls):
    """Without strict=True, scripts with undefined symbols still construct
    (the error fires at first eval, not at __init__)."""
    blk = cls(
        dt=0.1,
        time_mode="discrete",
        init_script="out_0 = 0.0",
        user_statements="out_0 = totally_made_up_symbol",
        inputs=[],
        outputs=["out_0"],
        name="lax_block",
        # strict not passed → defaults to False
    )
    assert blk.output_names == ["out_0"]


# ============================================================================
# T-036e (case 3): silent dead-store detection — strict=True only.
# ============================================================================
#
# When init_script writes to a name that (a) isn't an output, (b) isn't a
# private ``_*`` scratch var, (c) is never read elsewhere, and (d) looks
# like a typo of an output, raise with a "did you mean ..." hint.


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_silent_dead_store_with_typo_suggestion(cls):
    """Case 3: ``output_a = 1.0`` with ``outputs=["out_a"]`` is a typo
    silently dropped at exec — surface as ``BlockParameterError`` with
    "did you mean 'out_a'?"."""
    with pytest.raises(
        BlockParameterError,
        match=r"init_script assigns to 'output_a'.*Did you mean 'out_a'",
    ):
        _strict(
            cls,
            init_script="output_a = 1.0\nout_a = 0.0",
            inputs=[],
            outputs=["out_a"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_silent_dead_store_pure_scratch_no_error(cls):
    """Negative: ``tmp = 1.0; out_0 = tmp`` is fine — ``tmp`` IS read."""
    blk = _strict(
        cls,
        init_script="tmp = 1.0\nout_0 = tmp",
        inputs=[],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_silent_dead_store_underscore_scratch_no_error(cls):
    """Negative: a leading-underscore name is a private-scratch
    convention; never flagged."""
    blk = _strict(
        cls,
        init_script="_tmp = 1.0\nout_0 = 0.0",
        inputs=[],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_silent_dead_store_unrelated_name_no_error(cls):
    """Negative: an unread name with no close match in outputs is
    legitimate — possibly a documented constant (e.g. ``MY_CONST = 42``)
    or a forgotten leftover.  Conservative-by-design — only typo-suspect
    names are flagged."""
    blk = _strict(
        cls,
        init_script="MY_CONST = 42\nout_0 = 0.0",
        inputs=[],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_silent_dead_store_read_in_user_statements_no_error(cls):
    """Negative: a name written in init_script and read in user_statements
    is not dead — cross-script flow is allowed."""
    blk = _strict(
        cls,
        init_script="scale = 2.0\nout_0 = 0.0",
        user_statements="out_0 = scale * time",
        inputs=[],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_silent_dead_store_input_typo_no_error(cls):
    """Negative: writing to ``inputs[i]`` (overshadowing) is not a
    dead-store — the name appears in declared inputs so we leave it alone
    (a different check would catch the shadow case)."""
    blk = _strict(
        cls,
        init_script="u = 0.0\nout_0 = 0.0",
        user_statements="out_0 = u",
        inputs=["u"],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


# ============================================================================
# T-036e (case 5): full empty-inputs+read diagnostic — strict=True only.
# ============================================================================
#
# When ``inputs=[]`` and the user references an undefined symbol that
# looks like an input read (`u`, `x`, `signal_*`, etc.), the case-1
# error message suggests declaring it in ``inputs=[...]`` rather than
# the generic "undefined symbol" hint.


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_empty_inputs_with_input_like_read_error(cls):
    """Case 5: ``inputs=[]`` + ``out_0 = u + 1.0`` — surface a hint
    asking the user to declare ``u`` in ``inputs=[...]``."""
    with pytest.raises(
        BlockParameterError,
        match=r"declare 'u' in.*`inputs=\[\.\.\.\]`",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            user_statements="out_0 = u + 1.0",
            inputs=[],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_empty_inputs_with_signal_like_read_error(cls):
    """Case 5: ``signal_in`` is also recognised as an input-like name."""
    with pytest.raises(
        BlockParameterError,
        match=r"declare 'signal_in' in.*`inputs=\[\.\.\.\]`",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            user_statements="out_0 = signal_in",
            inputs=[],
            outputs=["out_0"],
        )


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_empty_inputs_well_formed_no_error(cls):
    """Negative: ``inputs=[]`` paired with ``out_0 = time`` is well-formed
    — ``time`` is framework-bound, not an input."""
    blk = _strict(
        cls,
        init_script="out_0 = 0.0",
        user_statements="out_0 = time",
        inputs=[],
        outputs=["out_0"],
    )
    assert blk.output_names == ["out_0"]


@pytest.mark.parametrize("cls", BLOCK_CLASSES)
def test_empty_inputs_unrecognised_name_inputs_hint(cls):
    """Case 5 generalised: any unrecognised symbol with ``inputs=[]``
    gets an "is this meant to come from an upstream block?" hint, since
    that's the most plausible user intent."""
    with pytest.raises(
        BlockParameterError,
        match=r"if 'foobar' is meant to come from an upstream block",
    ):
        _strict(
            cls,
            init_script="out_0 = 0.0",
            user_statements="out_0 = foobar",
            inputs=[],
            outputs=["out_0"],
        )

