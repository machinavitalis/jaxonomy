# Hybrid Dynamical System Tutorial: Thermostat-Controlled Room
# Uses Jaxonomy (jaxonomy). Install: pip install marimo jaxonomy matplotlib numpy

import marimo as mo

app = mo.App()

with app.setup:
    import numpy as np
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import jaxonomy
    from jaxonomy.framework import LeafSystem
    from jaxonomy.simulation import SimulatorOptions


@app.cell
def __mo():
    import marimo as mo
    return (mo,)


@app.cell
def __intro(mo):
    mo.md(r"""
    # Hybrid Dynamical System Tutorial: Thermostat-Controlled Room

    This notebook uses the **Jaxonomy** (jaxonomy) package to model and simulate
    a **thermostat-controlled room** — a canonical hybrid dynamical system with:

    - **Continuous dynamics**: room temperature $T$ evolves according to a first-order ODE.
    - **Discrete modes**: the heater is either **OFF** (0) or **HEAT** (1).
    - **Guarded transitions**: when temperature crosses a lower threshold the heater turns on;
      when it crosses an upper threshold it turns off (hysteresis).

    The flow and jump structure is:

    - **Flow (OFF)**: $\dot{T} = -\alpha(T - T_{\mathrm{amb}})$
    - **Flow (HEAT)**: $\dot{T} = -\alpha(T - T_{\mathrm{amb}}) + Q$
    - **Jump (OFF → HEAT)**: when $T \leq T_{\mathrm{low}}$
    - **Jump (HEAT → OFF)**: when $T \geq T_{\mathrm{high}}$

    This is a standard example in hybrid systems theory (e.g. Goebel–Sanfelice–Teel).
    """)
    return


@app.cell
def __model(mo):
    OFF, HEAT = 0, 1

    class ThermostatRoom(LeafSystem):
        def __init__(
            self,
            *args,
            alpha=0.1,
            T_amb=20.0,
            Q=5.0,
            T_low=19.0,
            T_high=21.0,
            name="thermostat",
            **kwargs,
        ):
            super().__init__(*args, name=name, **kwargs)

            self.declare_continuous_state(1, ode=self.ode)
            self.declare_continuous_state_output(name="T")
            self.declare_dynamic_parameter("alpha", alpha)
            self.declare_dynamic_parameter("T_amb", T_amb)
            self.declare_dynamic_parameter("Q", Q)
            self.declare_dynamic_parameter("T_low", T_low)
            self.declare_dynamic_parameter("T_high", T_high)

            self.declare_default_mode(OFF)

            self.declare_zero_crossing(
                guard=self._guard_turn_on,
                reset_map=None,
                start_mode=OFF,
                end_mode=HEAT,
                direction="positive_then_non_positive",
                name="turn_on",
            )
            self.declare_zero_crossing(
                guard=self._guard_turn_off,
                reset_map=None,
                start_mode=HEAT,
                end_mode=OFF,
                direction="positive_then_non_positive",
                name="turn_off",
            )

        def ode(self, time, state, **params):
            alpha = params["alpha"]
            T_amb = params["T_amb"]
            Q = params["Q"]
            T = state.continuous_state[0]
            mode = state.mode
            heat_rate = jnp.where(mode == HEAT, Q, 0.0)
            dT = -alpha * (T - T_amb) + heat_rate
            return jnp.array([dT])

        def _guard_turn_on(self, time, state, **params):
            T = state.continuous_state[0]
            T_low = params["T_low"]
            return T - T_low

        def _guard_turn_off(self, time, state, **params):
            T = state.continuous_state[0]
            T_high = params["T_high"]
            return T_high - T

    mo.md(
        r"""
        ## Thermostat model (LeafSystem)

        The thermostat is implemented as a `LeafSystem` with:
        - One continuous state: room temperature $T$
        - Modes: OFF (0) and HEAT (1)
        - Two zero-crossing events: turn heater **on** when $T \leq T_{\mathrm{low}}$,
          **off** when $T \geq T_{\mathrm{high}}$
        """
    )
    return (ThermostatRoom,)


@app.cell
def __params(mo):
    alpha = 0.1
    T_amb = 20.0
    Q = 5.0
    T_low = 19.0
    T_high = 21.0
    T0 = 18.0
    t_span = (0.0, 120.0)

    mo.md(
        r"""
        ## Parameters and initial condition

        - $\alpha$: thermal time constant, $T_{\mathrm{amb}}$: ambient temperature
        - $Q$: heating rate when ON, $T_{\mathrm{low}}$, $T_{\mathrm{high}}$: hysteresis band
        - $T_0$: initial temperature, $t_{\mathrm{span}}$: simulation interval
        """
    )
    return Q, T0, T_amb, T_high, T_low, alpha, t_span


@app.cell
def __simulate(Q, T0, T_amb, T_high, T_low, ThermostatRoom, alpha, t_span):
    system = ThermostatRoom(
        alpha=alpha,
        T_amb=T_amb,
        Q=Q,
        T_low=T_low,
        T_high=T_high,
    )
    context = system.create_context()
    context = context.with_continuous_state(jnp.array([T0]))

    options = SimulatorOptions(
        max_major_steps=500,
        atol=1e-10,
        rtol=1e-8,
        max_minor_step_size=0.05,
    )
    recorded_signals = {"T": system.output_ports[0]}

    results = jaxonomy.simulate(
        system,
        context,
        t_span,
        options=options,
        recorded_signals=recorded_signals,
    )

    t = results.time
    T = results.outputs["T"]
    return T, t


@app.cell
def __plot(T, T_amb, T_high, T_low, t):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(t, T[:, 0], color="C0", label="$T(t)$")
    ax.axhline(T_high, color="gray", ls="--", alpha=0.8, label="$T_{\\mathrm{high}}$")
    ax.axhline(T_low, color="gray", ls=":", alpha=0.8, label="$T_{\\mathrm{low}}$")
    ax.axhline(T_amb, color="green", ls="-", alpha=0.4, label="$T_{\\mathrm{amb}}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Thermostat-controlled room: hybrid dynamics with hysteresis")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def __summary(mo):
    mo.md(r"""
    ## Summary

    - **Hybrid system**: continuous state $T$ and discrete mode (OFF/HEAT).
    - **Zero-crossing events** in Jaxonomy encode the guards (e.g. $T - T_{\mathrm{low}}$)
      and mode transitions via `start_mode` and `end_mode`.
    - The ODE uses `state.mode` to switch the heating term, producing the piecewise
      continuous behavior and limit cycle in the band $[T_{\mathrm{low}}, T_{\mathrm{high}}]$.
    """)
    return


if __name__ == "__main__":
    app.run()
