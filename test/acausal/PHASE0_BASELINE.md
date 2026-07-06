# Phase 0 baseline — `test/acausal`

This file records a reproducible snapshot after adding the golden integration tests in `test_golden_integrations.py`.

## How to reproduce

From the repository root:

```bash
python3 -m pytest test/acausal/ -q --tb=no -ra
```

## Last recorded run

- **Result:** `57 passed`, `5 skipped`
- **Duration:** ~133 s (wall clock; varies by machine)

## Golden tests (regression wall)

| Test | Closed form |
|------|-------------|
| `test_golden_electrical_rc_lowpass` | \(V_C(t) = V_s(1 - e^{-t/(RC)})\), \(V_s=R=C=1\) |
| `test_golden_rotational_inertia_damper_constant_torque` | \( \omega(t) = \frac{\tau}{D}(1 - e^{-Dt/J}) \) |
| `test_golden_translational_mass_spring_damper_step_force` | Underdamped step \(M\ddot x + D\dot x + Kx = F\) |
| `test_golden_thermal_rc_step_to_setpoint` | \(T(t) = T_{\mathrm{src}} + (T_0 - T_{\mathrm{src}})e^{-t/(RC)}\) |

## Skipped tests (at baseline)

| Location | Reason |
|----------|--------|
| `test_electrical.py` (line ~714) | Diagram processing / WC-389 |
| `test_fluid.py` (several) | Marked flakey or experimental |

Re-run with `-ra` after any acausal change to refresh the skip list.

## Follow-up (later phases)

- Revisit isolated thermal **constant heat flow into one capacitor** once the single-node heat-source topology is structurally sound in `DiagramProcessing` / index reduction.
- Optionally add a `pytest.ini` marker (e.g. `golden_acausal`) for fast CI subsets.
