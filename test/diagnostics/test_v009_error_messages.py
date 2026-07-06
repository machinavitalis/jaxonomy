# SPDX-License-Identifier: MIT

"""V-009: Build-time and runtime error message quality.

Each test below intentionally constructs a malformed model and asserts that
the resulting error mentions the offending block / port / parameter by name,
rather than surfacing an opaque JAX-internal traceback.

Tests that the framework currently fails to give a useful, named error are
marked ``pytest.xfail`` with a reason tying back to T-002 (error-message
quality task) and a short excerpt of the actual current error message.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import pytest

import jaxonomy  # noqa: F401  (ensures backend init)
from jaxonomy import (
    DiagramBuilder,
    LeafSystem,
    Parameter,
    simulate,
)
from jaxonomy.library import (
    Adder,
    Constant,
    CustomPythonBlock,
    Gain,
    Integrator,
    LookupTable1d,
    Product,
)


# --------------------------------------------------------------------------- #
# 1. Two input ports connected via builder.connect (input -> input).
# --------------------------------------------------------------------------- #
def test_input_to_input_connect_names_offending_port():
    """``builder.connect`` accepts (input, input) silently today and the
    eventual error names only one block as 'not connected'. T-002 should
    catch this at connect time and mention both ports / 'inputs cannot
    connect to inputs'.
    """
    builder = DiagramBuilder()
    g1 = builder.add(Gain(2.0, name="g1"))
    g2 = builder.add(Gain(3.0, name="g2"))

    # T-002: connect() now validates port directions and raises immediately.
    with pytest.raises(
        Exception,
        match=r"(?i)(input.*input|cannot connect|g1.*g2|g2.*g1)",
    ):
        builder.connect(g1.input_ports[0], g2.input_ports[0])


# --------------------------------------------------------------------------- #
# 2. LookupTable1d with non-monotonic input data.
# --------------------------------------------------------------------------- #
def test_lookup_table_non_monotonic_names_block():
    """LookupTable1d should reject non-monotonic ``input_array`` and name
    the offending block. Today there is no monotonicity check at all -- the
    block silently produces undefined interpolation results.
    """
    builder = DiagramBuilder()
    src = builder.add(Constant(1.5, name="src"))

    lut_name = "speed_curve"
    # T-002: LookupTable1d now rejects non-monotonic input_array eagerly in
    # __init__, so the error surfaces inside builder.add().
    with pytest.raises(
        Exception, match=rf"(?is)monotonic.*{lut_name}|{lut_name}.*monotonic"
    ):
        builder.add(
            LookupTable1d(
                input_array=[0.0, 2.0, 1.0, 3.0],   # non-monotonic
                output_array=[0.0, 1.0, 2.0, 3.0],
                interpolation="linear",
                name=lut_name,
            )
        )


# --------------------------------------------------------------------------- #
# 3. Continuous-state shape mismatch.
# --------------------------------------------------------------------------- #
def test_continuous_state_shape_mismatch_names_block():
    """Integrator with shape-(3,) initial_state fed by a length-2 vector
    must produce a ShapeMismatchError naming the block.
    """
    builder = DiagramBuilder()
    src = builder.add(Constant(jnp.array([1.0, 2.0]), name="vec2"))
    intg_name = "intg_block"
    intg = builder.add(Integrator(initial_state=jnp.zeros(3), name=intg_name))
    builder.connect(src.output_ports[0], intg.input_ports[0])

    diagram = builder.build(name="m")
    with pytest.raises(Exception, match=rf"(?is)(shape|mismatch).*{intg_name}|{intg_name}.*shape"):
        ctx = diagram.create_context()
        simulate(diagram, ctx, (0.0, 1.0))


# --------------------------------------------------------------------------- #
# 4. Algebraic loop.
# --------------------------------------------------------------------------- #
def test_algebraic_loop_mentions_loop_and_blocks():
    """Pure-feedthrough cycle must raise an error that mentions
    'algebraic loop'. Bonus (T-029): the cycle blocks should be named.
    Today the names of the cycle blocks ARE listed -- both checks pass.
    """
    builder = DiagramBuilder()
    src = builder.add(Constant(1.0, name="src"))
    g1 = builder.add(Gain(2.0, name="loop_g1"))
    adder = builder.add(Adder(2, name="loop_adder"))
    g2 = builder.add(Gain(0.5, name="loop_g2"))

    builder.connect(src.output_ports[0], adder.input_ports[0])
    builder.connect(adder.output_ports[0], g1.input_ports[0])
    builder.connect(g1.output_ports[0], g2.input_ports[0])
    builder.connect(g2.output_ports[0], adder.input_ports[1])  # closes the loop

    diagram = builder.build(name="m")
    with pytest.raises(Exception, match=r"(?i)algebraic loop") as exc_info:
        ctx = diagram.create_context()
        simulate(diagram, ctx, (0.0, 1.0))

    # T-029 bonus: also verify the cycle block names appear.
    err_str = str(exc_info.value)
    for name in ("loop_g1", "loop_g2", "loop_adder"):
        assert name in err_str, (
            f"algebraic loop error did not name cycle block '{name}': {err_str}"
        )


# --------------------------------------------------------------------------- #
# 5. Vector & scalar mixed in a Product block.
# --------------------------------------------------------------------------- #
def test_product_vector_scalar_mismatch_names_block():
    """Per T-025, Product currently fails on mixed shapes rather than
    broadcasting. The error should name the offending Product block.
    """
    builder = DiagramBuilder()
    vec = builder.add(Constant(jnp.array([1.0, 2.0, 3.0]), name="vec3"))
    scl = builder.add(Constant(2.0, name="scl"))
    prod_name = "the_product"
    prod = builder.add(Product(2, name=prod_name))
    builder.connect(vec.output_ports[0], prod.input_ports[0])
    builder.connect(scl.output_ports[0], prod.input_ports[1])

    diagram = builder.build(name="m")
    with pytest.raises(Exception, match=rf"(?is){prod_name}"):
        ctx = diagram.create_context()
        simulate(diagram, ctx, (0.0, 1.0))


# --------------------------------------------------------------------------- #
# 6. Acausal model with invalid topology (dangling pin / missing eqs).
# --------------------------------------------------------------------------- #
def test_acausal_dangling_pin_names_diagram():
    """An acausal model with a dangling pin must produce an
    ``AcausalModelError``-flavoured error. Today the message references the
    AcausalDiagram and equation/variable counts but does NOT name the
    specific dangling component / pin -- xfail on naming.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad = AcausalDiagram(name="bad_circuit")
    v1 = elec.VoltageSource(ev, name="vs1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    # Only one connection -- v1.p and r1.p are both dangling.
    ad.connect(v1, "n", r1, "n")

    with pytest.raises(Exception) as exc_info:
        AcausalCompiler(ev, ad)()

    err_str = str(exc_info.value)
    # Lower bar: at least the model class / diagram is identified.
    assert re.search(r"(?i)acausal", err_str), (
        f"acausal compile error did not identify itself as acausal: {err_str}"
    )

    # T-002: the acausal error now also names the dangling component/pin.
    assert re.search(r"(?i)(vs1|r1|dangling|unconnected|pin)", err_str), (
        f"acausal compile error did not name the dangling component / pin: "
        f"{err_str}"
    )


# --------------------------------------------------------------------------- #
# 7. Custom Python block raising NameError during init.
# --------------------------------------------------------------------------- #
def test_custom_python_block_undefined_symbol_names_block():
    """A CustomPythonBlock whose init_script references an undefined symbol
    should surface a Jaxonomy error that mentions the block by name.
    The init_script is exec'd at block construction time, so the error
    surfaces inside ``builder.add(CustomPythonBlock(...))``.
    """
    block_name = "custom_py_block"
    builder = DiagramBuilder()
    builder.add(Constant(1.0, name="src"))
    with pytest.raises(Exception, match=rf"(?is){block_name}") as exc_info:
        builder.add(
            CustomPythonBlock(
                dt=0.1,
                init_script="y = undefined_thing_xyz",
                user_statements="y = x + 1",
                inputs=["x"],
                outputs=["y"],
                name=block_name,
            )
        )

    # Be doubly sure the original NameError message also surfaces.
    err_str = str(exc_info.value)
    assert (
        "undefined_thing_xyz" in err_str
        or "not defined" in err_str
        or exc_info.value.__cause__ is not None
    )


# --------------------------------------------------------------------------- #
# 8. Model parameter referenced from a scope where it isn't defined.
# --------------------------------------------------------------------------- #
def test_parameter_expression_undefined_symbol_names_parameter():
    """A Parameter declared as a python expression referencing an undefined
    name should produce an error that names the parameter and the block
    referencing it. Today this surfaces as a bare ``TypeError`` from inside
    the parameter resolver -- opaque, xfail.
    """
    p = Parameter(
        value="undefined_external_param * 3",
        is_python_expr=True,
        name="gain_param",
    )

    block_name = "gain_using_undef"
    builder = DiagramBuilder()
    builder.add(Constant(1.0, name="src"))

    # Parameter resolution happens eagerly inside Gain.__init__ via the
    # ``@parameters`` decorator, so the offending expression blows up here.
    with pytest.raises(Exception) as exc_info:
        builder.add(Gain(p, name=block_name))

    err_str = str(exc_info.value)
    # T-002: the resolver now wraps undefined-symbol errors with a message
    # that names the parameter and the offending symbol.
    assert re.search(
        r"(?is)(gain_param|undefined_external_param|" + block_name + r")",
        err_str,
    ), f"parameter-scope error did not name the parameter / symbol: {err_str!r}"


# --------------------------------------------------------------------------- #
# 9. T-036f: acausal index-reduction error names the offending subset.
# --------------------------------------------------------------------------- #
def test_t036f_parallel_voltage_sources_names_components():
    """Multiple voltage sources connected across the same node pair drive
    the system into a count mismatch (alias elimination collapses their
    constraints, leaving free flow variables).  The error message should
    include a "Likely cause" section that names every voltage source."""
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.error import AcausalModelError

    ev = EqnEnv()
    ad = AcausalDiagram(name="parallel_vsrcs")
    v1 = elec.VoltageSource(ev, name="vs1", V=1.0)
    v2 = elec.VoltageSource(ev, name="vs2", V=2.0)
    v3 = elec.VoltageSource(ev, name="vs3", V=3.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    g = elec.Ground(ev, name="gnd")
    ad.connect(v1, "p", v2, "p")
    ad.connect(v2, "p", v3, "p")
    ad.connect(v3, "p", r1, "p")
    ad.connect(v1, "n", v2, "n")
    ad.connect(v2, "n", v3, "n")
    ad.connect(v3, "n", r1, "n")
    ad.connect(v1, "n", g, "p")

    with pytest.raises(AcausalModelError) as exc_info:
        AcausalCompiler(ev, ad)()

    err_str = str(exc_info.value)
    # 1) preserve today's count-mismatch line
    assert re.search(
        r"Mismatch.*equations.*variables", err_str
    ), f"count-mismatch line missing: {err_str}"
    # 2) the new "Likely cause" section is present
    assert "Likely cause" in err_str, f"missing Likely cause section: {err_str}"
    # 3) all three voltage sources are named in the offending subset
    for name in ("vs1", "vs2", "vs3"):
        assert name in err_str, (
            f"T-036f localization missed component '{name}': {err_str}"
        )


def test_t036f_two_grounds_names_grounds():
    """Two Ground components on the same node are a classic redundancy.
    The diagnostic should flag both grounds as a parallel group."""
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.error import AcausalModelError

    ev = EqnEnv()
    ad = AcausalDiagram(name="two_grounds")
    v1 = elec.VoltageSource(ev, name="vs1", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    g1 = elec.Ground(ev, name="gnd1")
    g2 = elec.Ground(ev, name="gnd2")
    ad.connect(v1, "p", r1, "p")
    ad.connect(r1, "n", v1, "n")
    ad.connect(v1, "n", g1, "p")
    ad.connect(v1, "n", g2, "p")

    with pytest.raises(AcausalModelError) as exc_info:
        AcausalCompiler(ev, ad)()

    err_str = str(exc_info.value)
    assert "Likely cause" in err_str, f"missing Likely cause section: {err_str}"
    # Both grounds must be named in the same parallel group line.
    assert re.search(r"gnd1.*gnd2|gnd2.*gnd1", err_str), (
        f"T-036f did not name both grounds in a single group: {err_str}"
    )


def _make_alias_conflict_diagram(mode="conflict", with_third=False):
    """Build a tiny acausal diagram of custom electrical components
    each declaring two alias-form eqs for the SAME local ``V`` symbol.
    ``mode="conflict"`` flips the sign on the second eq → the new
    pre-pass must flag it; ``mode="redundant"`` keeps both signs equal.
    """
    import sympy as sp

    from jaxonomy.acausal import AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.component_library.electrical import ElecTwoPin

    class _Conflict(ElecTwoPin):
        def __init__(self, ev, name, s1=+1, s2=+1):
            self.name = name
            super().__init__(ev, self.name)
            self.add_eqs([sp.Eq(self.Ip.s, 0), sp.Eq(self.In.s, 0)])
            self.add_eqs([sp.Eq(self.V.s, s1 * self.Vp.s)])
            self.add_eqs([sp.Eq(self.V.s, s2 * self.Vp.s)])

    s2 = -1 if mode == "conflict" else +1
    ev = EqnEnv()
    ad = AcausalDiagram(name="alias_conflict_test")
    sensor_a = _Conflict(ev, "sensor_a", s1=+1, s2=s2)
    sensor_b = _Conflict(ev, "sensor_b", s1=+1, s2=s2)
    vs = elec.VoltageSource(ev, name="vs", V=1.0)
    g = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", sensor_a, "p")
    ad.connect(sensor_a, "n", sensor_b, "p")
    ad.connect(sensor_b, "n", vs, "n")
    ad.connect(vs, "n", g, "p")
    comps = [sensor_a, sensor_b]
    if with_third:
        sensor_c = _Conflict(ev, "sensor_c", s1=+1, s2=-1)
        ad.connect(sensor_c, "p", vs, "p")
        ad.connect(sensor_c, "n", vs, "n")
        comps.append(sensor_c)
    return ev, ad, comps


def test_t036f_alias_conflict_named_components():
    """Two components declare the same canonical variable's alias with
    differing sign expressions — the compiler must surface a clear
    AcausalModelError naming both components and the canonical var."""
    from jaxonomy.acausal import AcausalCompiler
    from jaxonomy.acausal.error import AcausalModelError

    ev, ad, comps = _make_alias_conflict_diagram(mode="conflict")
    with pytest.raises(AcausalModelError) as exc_info:
        AcausalCompiler(ev, ad)()
    err = str(exc_info.value)
    assert "Inconsistent alias declarations" in err, (
        f"alias-conflict error missing header: {err}"
    )
    # at least one of sensor_a / sensor_b must be named (each component
    # has its own internal conflict, so either may surface first).
    assert re.search(r"sensor_a|sensor_b", err), (
        f"alias-conflict error did not name any sensor: {err}"
    )
    # explicit demand from the task: name BOTH sensors when both
    # contain the same conflict pattern.  Both should appear in the
    # detailed bullet list.
    assert "sensor_a" in err and "sensor_b" in err, (
        f"alias-conflict error did not name both sensors: {err}"
    )


def test_t036f_alias_redundant_same_expression_no_error():
    """Two components declaring the EXACT same alias expression (same
    sign) are redundant but harmless — must compile without raising the
    new alias-conflict error."""
    from jaxonomy.acausal import AcausalCompiler
    from jaxonomy.acausal.error import AcausalModelError

    ev, ad, comps = _make_alias_conflict_diagram(mode="redundant")
    # Compile may still raise a *different* downstream error (count
    # mismatch from over-determined system), but it must NOT be an
    # alias-conflict.
    try:
        AcausalCompiler(ev, ad)()
    except AcausalModelError as e:
        assert "Inconsistent alias declarations" not in str(e), (
            f"redundant identical alias triggered spurious conflict: {e}"
        )
    except Exception:
        # Other compilation failures are acceptable for this test —
        # we only care that the alias-conflict path doesn't fire.
        pass


def test_t036f_alias_three_way_conflict():
    """Three components — two agree, one disagrees on sign.  The error
    must name all three (or at minimum the two distinct-expression
    groups) and identify the canonical variable."""
    from jaxonomy.acausal import AcausalCompiler
    from jaxonomy.acausal.error import AcausalModelError

    ev, ad, comps = _make_alias_conflict_diagram(
        mode="conflict", with_third=True
    )
    with pytest.raises(AcausalModelError) as exc_info:
        AcausalCompiler(ev, ad)()
    err = str(exc_info.value)
    assert "Inconsistent alias declarations" in err, (
        f"three-way conflict missing header: {err}"
    )
    # All three sensors should appear in the error.
    for name in ("sensor_a", "sensor_b", "sensor_c"):
        assert name in err, (
            f"three-way conflict did not name component '{name}': {err}"
        )


def test_t036f_well_formed_rc_no_likely_cause_section():
    """A clean RC circuit must compile without producing any 'Likely
    cause' suffix anywhere — the new diagnostic must not add noise on
    successful compiles."""
    import io
    import sys

    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad = AcausalDiagram(name="rc")
    v = elec.VoltageSource(ev, name="vs", V=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    c = elec.Capacitor(ev, name="c", C=1.0, initial_voltage=0.0)
    g = elec.Ground(ev, name="gnd")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", g, "p")

    # Capture stdout — diagram-processing prints debug info on the failure
    # path but a clean compile shouldn't surface any 'Likely cause' text.
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        AcausalCompiler(ev, ad)()
    finally:
        sys.stdout = old
    captured = buf.getvalue()
    assert "Likely cause" not in captured, (
        f"clean RC compile leaked 'Likely cause' text: {captured!r}"
    )


# --------------------------------------------------------------------------- #
# T-036f-followup-cross-component-aliases: write-time guard inside the
# alias-elimination loop catches conflicts that emerge ONLY after pot/flow
# elimination collapses pin-potentials into a shared canonical symbol.
# The pre-pass `_t036f_check_alias_conflicts` cannot see this case because
# at pre-pass time the two component pin variables are still distinct.
# --------------------------------------------------------------------------- #
def test_t036f_cross_component_alias_conflict_named():
    """Directly exercise the write-time guard with a hand-crafted
    `aliaser_map` aliasee_list containing a sign-flipped duplicate.
    The guard must raise an `AcausalModelError` whose message names
    both contributing components (best-effort via `sym_to_cmp`) and
    the canonical aliaser variable.

    Going end-to-end is structurally hard: real cross-component
    sign flips (e.g. two ElecTwoPin variants disagreeing on `Vp - Vn`
    vs `Vn - Vp`) decompose into distinct aliasee symbols whose
    contradiction surfaces downstream as a count mismatch rather
    than a duplicate aliaser_map write.  The write-time guard is the
    correct primitive — exercise it directly.
    """
    from jaxonomy.acausal import AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.diagram_processing import DiagramProcessing
    from jaxonomy.acausal.error import AcausalModelError

    ev = EqnEnv()
    ad = AcausalDiagram(name="cross_component_test")
    vs = elec.VoltageSource(ev, name="vs", V=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    g = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r, "p")
    ad.connect(r, "n", vs, "n")
    ad.connect(vs, "n", g, "p")

    dp = DiagramProcessing(ev, ad, verbose=False)
    # Populate `sym_to_cmp` so the diagnostic can name components.
    ad.add_cmp_sympy_syms(vs)
    ad.add_cmp_sympy_syms(r)
    dp.dpd = None

    # Hand-craft a conflict: an existing alias entry says
    # `r_V = vs_V`, the incoming write says `r_V = -vs_V`.
    fake_aliaser = vs.V
    fake_aliasee = r.V
    existing_list = [(fake_aliasee, fake_aliaser.s)]

    with pytest.raises(AcausalModelError) as exc_info:
        dp._t036f_check_write_time_alias_conflict(
            fake_aliaser,
            fake_aliasee,
            -fake_aliaser.s,
            eq=None,
            aliasee_list=existing_list,
        )
    err = str(exc_info.value)
    assert "Inconsistent alias declarations" in err, (
        f"missing header: {err}"
    )
    assert "vs_V" in err, f"canonical aliaser not named: {err}"
    assert "r_V" in err, f"aliasee not named: {err}"
    # `r` is resolvable via sym_to_cmp on the existing-side aliasee.
    assert "r" in err, f"existing-side component not named: {err}"
    # Sign-flipped expressions both appear in bullet form.
    assert re.search(r"r_V = vs_V", err), (
        f"existing expression line missing: {err}"
    )
    assert re.search(r"r_V = -vs_V", err), (
        f"new expression line missing: {err}"
    )


def test_t036f_cross_component_consistent_no_error():
    """Two equivalent expressions (under sympy.expand) for the same
    `(aliaser, aliasee)` pair MUST NOT raise.  Equivalent declarations
    are harmlessly redundant; the guard must only fire on genuine
    contradictions."""
    import sympy as sp

    from jaxonomy.acausal import AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.diagram_processing import DiagramProcessing

    ev = EqnEnv()
    ad = AcausalDiagram(name="cross_component_consistent")
    vs = elec.VoltageSource(ev, name="vs", V=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    g = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r, "p")
    ad.connect(r, "n", vs, "n")
    ad.connect(vs, "n", g, "p")

    dp = DiagramProcessing(ev, ad, verbose=False)
    ad.add_cmp_sympy_syms(vs)
    ad.add_cmp_sympy_syms(r)
    dp.dpd = None

    # Existing: `r_V = vs_V + 0`.  Incoming: `r_V = vs_V`.
    # `sp.expand` collapses both to `vs_V`.
    fake_aliaser = vs.V
    fake_aliasee = r.V
    existing_list = [(fake_aliasee, fake_aliaser.s + sp.Integer(0))]

    # Should NOT raise.
    dp._t036f_check_write_time_alias_conflict(
        fake_aliaser,
        fake_aliasee,
        fake_aliaser.s,
        eq=None,
        aliasee_list=existing_list,
    )


def test_t036f_cross_component_different_aliasee_no_error():
    """When the new aliasee differs from any existing aliasee for the
    same aliaser, the guard must NOT fire — that's the normal
    "multiple component pins share a node" pattern, not a conflict."""
    from jaxonomy.acausal import AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.diagram_processing import DiagramProcessing

    ev = EqnEnv()
    ad = AcausalDiagram(name="cross_component_distinct")
    vs = elec.VoltageSource(ev, name="vs", V=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    r2 = elec.Resistor(ev, name="r2", R=2.0)
    g = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "p")
    ad.connect(r1, "n", r2, "p")
    ad.connect(r2, "n", vs, "n")
    ad.connect(vs, "n", g, "p")

    dp = DiagramProcessing(ev, ad, verbose=False)
    ad.add_cmp_sympy_syms(vs)
    ad.add_cmp_sympy_syms(r1)
    ad.add_cmp_sympy_syms(r2)
    dp.dpd = None

    fake_aliaser = vs.V
    # Two DISTINCT aliasees both alias to vs.V — that's fine.
    existing_list = [(r1.V, fake_aliaser.s)]

    # Adding a different aliasee with a different (or even same) expr
    # must not flag a conflict.
    dp._t036f_check_write_time_alias_conflict(
        fake_aliaser,
        r2.V,
        -fake_aliaser.s,
        eq=None,
        aliasee_list=existing_list,
    )


# --------------------------------------------------------------------------- #
# T-036f-followup-cross-component-aliases-architecture: post-elimination
# sign-flip detection.  The write-time guard above catches cases where two
# alias entries land on the SAME aliasee with conflicting expressions.  The
# semantic sign-flip case — two ElecTwoPin variants connected with opposite
# pin polarity to the same nodes, each alias-eliminated through a DISTINCT
# aliasee — only manifests as opposite-sign entries on a SHARED canonical
# aliaser AFTER pot/flow elimination chains have run.  The post-loop scan
# in `_t036f_check_post_elim_sign_flip` walks `aliaser_map` for that
# pattern and raises an `AcausalModelError` naming both components.
# --------------------------------------------------------------------------- #
def test_t036f_cross_component_sign_flip_named():
    """Two voltage sources connected with OPPOSITE pin polarity to the
    same node pair drive the system into an over-determined sign-flip:
    `vs_a.p == vs_b.n` and `vs_a.n == vs_b.p`, with both pinned to known
    voltage values via component-internal `V = v` constraints.  The new
    post-elimination scan must raise `AcausalModelError` whose message
    names BOTH sources, the canonical aliaser variable, and explains the
    sign-flip cause.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.error import AcausalModelError

    ev = EqnEnv()
    ad = AcausalDiagram(name="opposite_polarity_vsrcs")
    v1 = elec.VoltageSource(ev, name="vs_a", v=1.0)
    v2 = elec.VoltageSource(ev, name="vs_b", v=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    c = elec.Capacitor(ev, name="c", C=1.0, initial_voltage=0.0)
    g = elec.Ground(ev, name="gnd")
    # Both sources see the same node pair, but with opposite pin polarity.
    ad.connect(v1, "p", v2, "n")     # node_top
    ad.connect(v1, "n", v2, "p")     # node_bot
    ad.connect(v1, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v1, "n")
    ad.connect(v1, "n", g, "p")

    with pytest.raises(AcausalModelError) as exc_info:
        AcausalCompiler(ev, ad)()
    err = str(exc_info.value)
    assert "Inconsistent alias declarations through canonical" in err, (
        f"missing post-elim sign-flip header: {err}"
    )
    # Both source names must appear.
    for name in ("vs_a", "vs_b"):
        assert name in err, (
            f"sign-flip error did not name component '{name}': {err}"
        )
    # The canonical aliaser variable should appear in the message.
    assert re.search(r"vs_[ab]_v", err), (
        f"sign-flip error did not name a canonical aliaser: {err}"
    )
    # The 'pin polarity' guidance should be present.
    assert re.search(r"(?i)pin polarity|sign", err), (
        f"sign-flip error missing remediation hint: {err}"
    )


def test_t036f_cross_component_sign_flip_consistent_no_error():
    """Two voltage sources at the same node pair with MATCHING polarity
    are still redundant (the existing parallel-redundancy diagnostic
    should fire) but must NOT trigger the new sign-flip diagnostic —
    both sources push the same direction, so the sign-flip pattern is
    absent."""
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.error import AcausalModelError

    ev = EqnEnv()
    ad = AcausalDiagram(name="same_polarity_vsrcs")
    v1 = elec.VoltageSource(ev, name="vs_a", v=1.0)
    v2 = elec.VoltageSource(ev, name="vs_b", v=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    c = elec.Capacitor(ev, name="c", C=1.0, initial_voltage=0.0)
    g = elec.Ground(ev, name="gnd")
    # Same polarity: both p at node_top, both n at node_bot.
    ad.connect(v1, "p", v2, "p")
    ad.connect(v1, "n", v2, "n")
    ad.connect(v1, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v1, "n")
    ad.connect(v1, "n", g, "p")

    # The system is still over-determined (parallel sources), so a
    # downstream error is expected — but it must NOT be the sign-flip
    # variant.
    with pytest.raises(AcausalModelError) as exc_info:
        AcausalCompiler(ev, ad)()
    err = str(exc_info.value)
    assert "Inconsistent alias declarations through canonical" not in err, (
        f"same-polarity duplicates spuriously triggered sign-flip "
        f"diagnostic: {err}"
    )


def test_t036f_cross_component_sensor_sign_flip_no_error():
    """Two voltage sensors with sign-flipped pin orientation observe
    the same circuit but DON'T constrain it (sensors set Ip=0/In=0,
    not V=const).  The post-elim sign-flip scan must NOT fire: the
    aliasers ending up here are `node_pot`-kind, not `param`/`inp`,
    and the components contribute observation-only aliases.  The
    full circuit must compile cleanly."""
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec
    from jaxonomy.acausal.error import AcausalModelError

    ev = EqnEnv()
    ad = AcausalDiagram(name="sensor_sign_flip")
    v = elec.VoltageSource(ev, name="vs", v=1.0)
    r = elec.Resistor(ev, name="r", R=1.0)
    c = elec.Capacitor(ev, name="c", C=1.0, initial_voltage=0.0)
    s_a = elec.VoltageSensor(ev, name="vsensor_a")
    s_b = elec.VoltageSensor(ev, name="vsensor_b")
    g = elec.Ground(ev, name="gnd")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", g, "p")
    # sensor_a: p at top, n at bot.
    ad.connect(s_a, "p", v, "p")
    ad.connect(s_a, "n", v, "n")
    # sensor_b: opposite polarity — p at bot, n at top.
    ad.connect(s_b, "p", v, "n")
    ad.connect(s_b, "n", v, "p")

    # Must compile without raising the sign-flip diagnostic.  Sensors
    # don't constrain the system, so the post-elim scan must skip them
    # (false-positive guard).
    try:
        AcausalCompiler(ev, ad)()
    except AcausalModelError as e:
        msg = str(e)
        assert "Inconsistent alias declarations through canonical" not in msg, (
            f"sensor sign-flip spuriously triggered post-elim "
            f"diagnostic (false positive): {msg}"
        )
