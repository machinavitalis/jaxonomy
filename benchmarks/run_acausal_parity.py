# SPDX-License-Identifier: MIT
"""
S4 Acausal / Modelica Parity Benchmark.

This script validates:
1. Lowpass RC Ladder Network (index-reduced DAE compiled via AcausalCompiler)
   against the exact linear state-space ODE solution.
2. DC Motor with Inertia and Viscous Damping (multi-domain acausal coupling)
   against the exact linear state-space ODE solution.
3. Trajectory Differentiability: computes exact JAX analytical gradients
   through the compiled acausal simulation DAE solver and checks against finite differences.
4. Outputs the equivalent Modelica (.mo) code for validation in Modelica standard tools.
"""

import os
import sys
import json
import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure jaxonomy package is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import electrical as elec
from jaxonomy.acausal import rotational as rot
from jaxonomy.acausal import thermal as thermal
from jaxonomy.acausal import battery as battery
import jaxonomy
from jaxonomy.simulation import SimulatorOptions


def build_and_simulate_rc_ladder(v_val=5.0, r1_val=1.0, c1_val=0.5, r2_val=2.0, c2_val=1.5, t_end=5.0):
    """Build and simulate a 2-stage RC ladder network using Jaxonomy acausal DAE compilation."""
    ev = EqnEnv()
    ad = AcausalDiagram()

    v1 = elec.VoltageSource(ev, name="v1", v=v_val)
    r1 = elec.Resistor(ev, name="r1", R=r1_val)
    c1 = elec.Capacitor(ev, name="c1", C=c1_val, initial_voltage=0.0, initial_voltage_fixed=True)
    r2 = elec.Resistor(ev, name="r2", R=r2_val)
    c2 = elec.Capacitor(ev, name="c2", C=c2_val, initial_voltage=0.0, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    
    sensV1 = elec.VoltageSensor(ev, name="sensV1")
    sensV2 = elec.VoltageSensor(ev, name="sensV2")

    # Connect components
    ad.connect(v1, "p", r1, "p")
    ad.connect(r1, "n", c1, "p")
    ad.connect(c1, "p", r2, "p")
    ad.connect(r2, "n", c2, "p")
    ad.connect(v1, "n", gnd, "p")
    ad.connect(c1, "n", gnd, "p")
    ad.connect(c2, "n", gnd, "p")
    
    # Voltage sensors across capacitors
    ad.connect(c1, "p", sensV1, "p")
    ad.connect(c1, "n", sensV1, "n")
    ad.connect(c2, "p", sensV2, "p")
    ad.connect(c2, "n", sensV2, "n")

    # Compile acausal system
    ac = AcausalCompiler(ev, ad, verbose=False)
    lpf = ac()

    # Build jaxonomy diagram
    builder = jaxonomy.DiagramBuilder()
    lpf_sys = builder.add(lpf)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # Setup signals to record
    c1_v_idx = lpf_sys.outsym_to_portid[sensV1.get_sym_by_port_name("v")]
    c2_v_idx = lpf_sys.outsym_to_portid[sensV2.get_sym_by_port_name("v")]
    recorded_signals = {
        "vc1": lpf_sys.output_ports[c1_v_idx],
        "vc2": lpf_sys.output_ports[c2_v_idx],
    }

    # Simulate
    opts = SimulatorOptions(ode_solver_method="bdf", rtol=1e-10, atol=1e-12, buffer_length=10000)
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded_signals, options=opts
    )
    return results.time, results.outputs["vc1"], results.outputs["vc2"]


def solve_rc_ladder_state_space(v_val=5.0, r1_val=1.0, c1_val=0.5, r2_val=2.0, c2_val=1.5, t_eval=None):
    """Solve the RC ladder state-space ODEs to high precision via SciPy solve_ivp."""
    # State: x = [vc1, vc2]
    # A = [ -(1/(R1*C1) + 1/(R2*C1)),  1/(R2*C1) ]
    #     [  1/(R2*C2),               -1/(R2*C2) ]
    # B = [ 1/(R1*C1),  0 ]^T
    
    A = np.array([
        [-(1.0 / (r1_val * c1_val) + 1.0 / (r2_val * c1_val)), 1.0 / (r2_val * c1_val)],
        [1.0 / (r2_val * c2_val), -1.0 / (r2_val * c2_val)]
    ])
    B = np.array([1.0 / (r1_val * c1_val), 0.0])

    def odefun(t, x):
        return A @ x + B * v_val

    sol = solve_ivp(odefun, (t_eval[0], t_eval[-1]), [0.0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.y[0, :], sol.y[1, :]


def build_and_simulate_dc_motor(v_val=24.0, r_val=2.0, kt_val=0.5, ke_val=0.5, j_val=0.1, b_val=0.01, t_end=5.0):
    """Build and simulate a DC motor with mechanical inertia using Jaxonomy."""
    ev = EqnEnv()
    ad = AcausalDiagram()

    v1 = elec.VoltageSource(ev, name="v1", v=v_val)
    mot = elec.DCMotorSimple(ev, name="mot", R=r_val, Kt=kt_val, Ke=ke_val, J=j_val, B=b_val,
                             initial_velocity=0.0, initial_velocity_fixed=True,
                             initial_angle=0.0, initial_angle_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    rotSpd = rot.MotionSensor(ev, name="rotSpd", enable_flange_b=False, enable_angle_port=True)
    sensI = elec.CurrentSensor(ev, name="sensI")

    ad.connect(v1, "p", sensI, "p")
    ad.connect(sensI, "n", mot, "pos")
    ad.connect(v1, "n", mot, "neg")
    ad.connect(v1, "n", gnd, "p")
    ad.connect(mot, "shaft", rotSpd, "flange_a")

    # Compile acausal system
    ac = AcausalCompiler(ev, ad, verbose=False)
    motor_sys = ac()

    # Build jaxonomy diagram
    builder = jaxonomy.DiagramBuilder()
    mot_sys = builder.add(motor_sys)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # Setup signals
    w_idx = mot_sys.outsym_to_portid[rotSpd.get_sym_by_port_name("w_rel")]
    theta_idx = mot_sys.outsym_to_portid[rotSpd.get_sym_by_port_name("ang_rel")]
    i_idx = mot_sys.outsym_to_portid[sensI.get_sym_by_port_name("i")]

    recorded_signals = {
        "speed": mot_sys.output_ports[w_idx],
        "angle": mot_sys.output_ports[theta_idx],
        "current": mot_sys.output_ports[i_idx],
    }

    # Simulate
    opts = SimulatorOptions(ode_solver_method="bdf", rtol=1e-10, atol=1e-12, buffer_length=10000)
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded_signals, options=opts
    )
    return results.time, results.outputs["speed"], results.outputs["angle"], results.outputs["current"]


def solve_dc_motor_state_space(v_val=24.0, r_val=2.0, kt_val=0.5, ke_val=0.5, j_val=0.1, b_val=0.01, t_eval=None):
    """Solve the DC Motor state-space ODEs to high precision via SciPy solve_ivp."""
    # State: x = [theta, w]
    # dw/dt = (Kt/J)*I - (B/J)*w
    # I = (V - Ke*w)/R
    # => dw/dt = -(Kt*Ke/(J*R) + B/J)*w + (Kt/(J*R))*V
    # dtheta/dt = w
    
    a_speed = -(kt_val * ke_val / (j_val * r_val) + b_val / j_val)
    b_speed = kt_val / (j_val * r_val)

    def odefun(t, x):
        theta, w = x[0], x[1]
        dtheta = w
        dw = a_speed * w + b_speed * v_val
        return [dtheta, dw]

    sol = solve_ivp(odefun, (t_eval[0], t_eval[-1]), [0.0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    
    # Reconstruct current from state w: I = (V - Ke*w)/R
    speed_traj = sol.y[1, :]
    current_traj = (v_val - ke_val * speed_traj) / r_val
    return speed_traj, sol.y[0, :], current_traj


def run_acausal_gradient_check():
    """Verify analytical JAX gradients computed through the compiled acausal simulation."""
    print("\n==========================================================")
    print("RUNNING ACOUSAL GRADIENT CHECK (JAX AD vs. FINITE DIFF)")
    print("==========================================================")
    
    # We will build the DC motor system and compile it.
    # To compute gradients w.r.t. parameters (R, Kt, B) using jax.grad,
    # we define a functional wrapper that updates parameters in context and simulates.
    
    # Setup jaxonomy backend
    from jaxonomy.backend import set_backend
    set_backend("jax")
    
    import jax
    jax.config.update("jax_enable_x64", True)

    # Define the diagram model functional
    ev = EqnEnv()
    ad = AcausalDiagram()
    v1 = elec.VoltageSource(ev, name="v1", v=24.0)
    mot = elec.DCMotorSimple(ev, name="mot", R=2.0, Kt=0.5, Ke=0.5, J=0.1, B=0.01)
    gnd = elec.Ground(ev, name="gnd")
    rotSpd = rot.MotionSensor(ev, name="rotSpd", enable_flange_b=False)

    ad.connect(v1, "p", mot, "pos")
    ad.connect(v1, "n", mot, "neg")
    ad.connect(v1, "n", gnd, "p")
    ad.connect(mot, "shaft", rotSpd, "flange_a")

    ac = AcausalCompiler(ev, ad, verbose=False)
    motor_sys = ac()
    builder = jaxonomy.DiagramBuilder()
    mot_sys = builder.add(motor_sys)
    diagram = builder.build()
    
    w_idx = mot_sys.outsym_to_portid[rotSpd.get_sym_by_port_name("w_rel")]
    opts = SimulatorOptions(math_backend="jax", ode_solver_method="bdf", rtol=1e-10, atol=1e-12, enable_autodiff=True)
    
    # We want to differentiate the final speed w(t_end) w.r.t. parameters [R, Kt, B]
    # True values: R=2.0, Kt=0.5, B=0.01. Let's vary around this point: params = [2.0, 0.5, 0.01]
    
    def simulate_speed_fn(params):
        R_val, Kt_val, B_val = params[0], params[1], params[2]
        
        ctx = diagram.create_context()
        sub = ctx[mot_sys.system_id]
        # Inject parameters into the compiler context
        sub = sub.with_parameter("mot_R", R_val)
        sub = sub.with_parameter("mot_Kt", Kt_val)
        sub = sub.with_parameter("mot_Ke", Kt_val) # Assume Kt = Ke
        sub = sub.with_parameter("mot_B", B_val)
        ctx = ctx.with_subcontext(mot_sys.system_id, sub)
        
        # We simulate with postprocess=False to avoid NumPy conversion of JAX Tracers
        res = jaxonomy.simulate(diagram, ctx, (0.0, 3.0), options=opts, postprocess=False)
        
        # Return the final speed value by evaluating the output port
        return mot_sys.output_ports[w_idx].eval(res.context)

    # Compute JAX gradient
    jax_grad_fn = jax.grad(simulate_speed_fn)
    test_params = jnp.array([2.0, 0.5, 0.01])
    
    print(f"Test parameters: R={test_params[0]}, Kt={test_params[1]}, B={test_params[2]}")
    print("Computing JAX analytical gradient (this JIT compiles the DAE solver)...")
    
    t0 = time.perf_counter()
    grad_jax = jax_grad_fn(test_params)
    print(f"JAX gradient computed in {time.perf_counter() - t0:.3f}s")
    grad_jax = np.array(grad_jax)
    
    # Compute central finite differences
    eps = 1e-5
    grad_fd = []
    for i in range(len(test_params)):
        p_plus = np.array(test_params)
        p_plus[i] += eps
        val_plus = float(simulate_speed_fn(p_plus))
        
        p_minus = np.array(test_params)
        p_minus[i] -= eps
        val_minus = float(simulate_speed_fn(p_minus))
        
        grad_fd.append((val_plus - val_minus) / (2 * eps))
    grad_fd = np.array(grad_fd)
    
    # Print results
    print("\n--- ACOUSAL DAE GRADIENT TABLE ---")
    headers = ["Parameter", "Analytical Grad (JAX)", "Finite Diff Grad", "Abs Difference"]
    print(f"{headers[0]:<12} {headers[1]:<25} {headers[2]:<20} {headers[3]:<15}")
    print("-" * 75)
    names = ["R (resistance)", "Kt (motor const)", "B (damping)"]
    for i, name in enumerate(names):
        diff = abs(grad_jax[i] - grad_fd[i])
        print(f"{name:<12} {grad_jax[i]:<25.8f} {grad_fd[i]:<20.8f} {diff:<15.8e}")
    print("-" * 75)
    
    max_diff = np.max(np.abs(grad_jax - grad_fd))
    print(f"Max Gradient Absolute Difference: {max_diff:.8e}")
    assert max_diff < 1e-5, f"DAE Gradient validation failed! max_diff={max_diff:.8e} >= 1e-5"
    print("[SUCCESS] Acausal DAE gradient JAX autodiff matches finite differences!")
    return grad_jax.tolist(), grad_fd.tolist(), float(max_diff)


def print_modelica_code():
    """Output the equivalent Modelica standard library code for direct verification reference."""
    modelica_rc_ladder = """
model RCLadder
  Modelica.Electrical.Analog.Basic.Ground ground;
  Modelica.Electrical.Analog.Sources.ConstantVoltage voltageSource(V=5.0);
  Modelica.Electrical.Analog.Basic.Resistor resistor1(R=1.0);
  Modelica.Electrical.Analog.Basic.Capacitor capacitor1(C=0.5, v(start=0.0, fixed=true));
  Modelica.Electrical.Analog.Basic.Resistor resistor2(R=2.0);
  Modelica.Electrical.Analog.Basic.Capacitor capacitor2(C=1.5, v(start=0.0, fixed=true));
equation
  connect(voltageSource.p, resistor1.p);
  connect(resistor1.n, capacitor1.p);
  connect(capacitor1.p, resistor2.p);
  connect(resistor2.n, capacitor2.p);
  connect(voltageSource.n, ground.p);
  connect(capacitor1.n, ground.p);
  connect(capacitor2.n, ground.p);
end RCLadder;
"""

    modelica_dc_motor = """
model DCMotorLoad
  Modelica.Electrical.Analog.Sources.ConstantVoltage voltageSource(V=24.0);
  Modelica.Electrical.Analog.Basic.Ground electricalGround;
  Modelica.Electrical.Analog.Basic.Resistor resistor(R=2.0);
  Modelica.Electrical.Analog.Basic.Inductor inductor(L=1e-6); // Small parasitic L to match DCMotorSimple
  Modelica.Electrical.Analog.Basic.RotationalEMF motor(k=0.5);
  Modelica.Mechanics.Rotational.Components.Inertia inertia(J=0.1, phi(start=0.0, fixed=true), w(start=0.0, fixed=true));
  Modelica.Mechanics.Rotational.Components.Damper damper(d=0.01);
  Modelica.Mechanics.Rotational.Components.Fixed fixedAngle;
equation
  connect(voltageSource.p, resistor.p);
  connect(resistor.n, inductor.p);
  connect(inductor.n, motor.p);
  connect(voltageSource.n, motor.n);
  connect(voltageSource.n, electricalGround.p);
  connect(motor.flange, inertia.flange_a);
  connect(inertia.flange_b, damper.flange_a);
  connect(damper.flange_b, fixedAngle.flange);
end DCMotorLoad;
"""
    print("\n==========================================================")
    print("EQUIVALENT MODELICA CODE (FOR STANDARD PARITY REFERENCE)")
    print("==========================================================")
    print("1. 2-STAGE RC LADDER:")
    print(modelica_rc_ladder)
    print("\n2. DC MOTOR WITH LOAD:")
    print(modelica_dc_motor)
    print("==========================================================")
    return modelica_rc_ladder, modelica_dc_motor


def build_and_simulate_battery_cell(
    r0_val=0.02, r1_val=0.01, c1_val=1000.0, cap_val=2.0, i_discharge=10.0,
    casing_c=80.0, insulator_r=2.0, t_ambient=298.15, t_end=20.0
):
    """Build and simulate a coupled electro-thermal battery cell using Jaxonomy acausal DAE compilation."""
    ev = EqnEnv()
    ad = AcausalDiagram()

    # Create components
    batteryCell = battery.BatteryCellECM(
        ev, name="bat", R0=r0_val, R1=r1_val, C1=c1_val, capacity_Ah=cap_val,
        ocv_soc=(0.0, 1.0), ocv_volts=(3.0, 4.2),
        initial_soc=1.0, initial_soc_fixed=True,
        initial_v_rc=0.0, initial_v_rc_fixed=True,
        enable_heat_port=True, enable_soc_port=True, enable_v_rc_port=True
    )
    
    currentSource = elec.CurrentSource(ev, name="cs", i=i_discharge)
    gnd = elec.Ground(ev, name="gnd")
    
    casing = thermal.HeatCapacitor(
        ev, name="casing", C=casing_c, initial_temperature=t_ambient, initial_temperature_fixed=True
    )
    insulator = thermal.Insulator(ev, name="insulator", R=insulator_r)
    ambient = thermal.TemperatureSource(ev, name="ambient", temperature=t_ambient)
    
    tempSensor = thermal.TemperatureSensor(ev, name="tempSensor", enable_port_b=False)

    # Connections
    ad.connect(currentSource, "p", batteryCell, "p")
    ad.connect(currentSource, "n", batteryCell, "n")
    ad.connect(batteryCell, "n", gnd, "p")
    
    ad.connect(batteryCell, "heat", casing, "port")
    ad.connect(casing, "port", insulator, "port_a")
    ad.connect(insulator, "port_b", ambient, "port")
    ad.connect(casing, "port", tempSensor, "port_a")

    # Compile acausal system
    ac = AcausalCompiler(ev, ad, verbose=False)
    bat_sys_compiled = ac()

    # Build jaxonomy diagram
    builder = jaxonomy.DiagramBuilder()
    bat_sys = builder.add(bat_sys_compiled)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # Setup signals to record
    soc_idx = bat_sys.outsym_to_portid[batteryCell.get_sym_by_port_name("soc")]
    v_rc_idx = bat_sys.outsym_to_portid[batteryCell.get_sym_by_port_name("v_rc")]
    temp_idx = bat_sys.outsym_to_portid[tempSensor.get_sym_by_port_name("T_rel")]
    
    recorded_signals = {
        "soc": bat_sys.output_ports[soc_idx],
        "v_rc": bat_sys.output_ports[v_rc_idx],
        "temp": bat_sys.output_ports[temp_idx],
    }

    # Simulate
    opts = SimulatorOptions(ode_solver_method="bdf", rtol=1e-10, atol=1e-12, buffer_length=10000)
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded_signals, options=opts
    )
    return results.time, results.outputs["soc"], results.outputs["v_rc"], results.outputs["temp"]


def solve_battery_state_space(
    r0_val=0.02, r1_val=0.01, c1_val=1000.0, cap_val=2.0, i_discharge=10.0,
    casing_c=80.0, insulator_r=2.0, t_ambient=298.15, t_eval=None
):
    """Solve the Electro-Thermal Battery cell state-space ODEs to high precision via SciPy solve_ivp."""
    ip = -i_discharge
    
    def odefun(t, x):
        soc, v_rc, temp = x[0], x[1], x[2]
        dsoc = ip / (3600.0 * cap_val)
        dv_rc = ip / c1_val - v_rc / (r1_val * c1_val)
        dtemp = (ip**2 * (r0_val + r1_val) - (temp - t_ambient) / insulator_r) / casing_c
        return [dsoc, dv_rc, dtemp]
        
    sol = solve_ivp(odefun, (t_eval[0], t_eval[-1]), [1.0, 0.0, t_ambient], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.y[0, :], sol.y[1, :], sol.y[2, :]


def run_battery_gradient_check():
    """Verify analytical JAX gradients computed through the compiled acausal simulation for the battery system."""
    from jaxonomy.backend import set_backend
    set_backend("jax")
    
    import jax
    jax.config.update("jax_enable_x64", True)
    
    ev = EqnEnv()
    ad = AcausalDiagram()
    
    batteryCell = battery.BatteryCellECM(
        ev, name="bat", R0=0.02, R1=0.01, C1=1000.0, capacity_Ah=2.0,
        ocv_soc=(0.0, 1.0), ocv_volts=(3.0, 4.2),
        initial_soc=1.0, initial_soc_fixed=True,
        initial_v_rc=0.0, initial_v_rc_fixed=True,
        enable_heat_port=True
    )
    currentSource = elec.CurrentSource(ev, name="cs", i=10.0)
    gnd = elec.Ground(ev, name="gnd")
    casing = thermal.HeatCapacitor(ev, name="casing", C=80.0, initial_temperature=298.15, initial_temperature_fixed=True)
    insulator = thermal.Insulator(ev, name="insulator", R=2.0)
    ambient = thermal.TemperatureSource(ev, name="ambient", temperature=298.15)
    tempSensor = thermal.TemperatureSensor(ev, name="tempSensor", enable_port_b=False)

    ad.connect(currentSource, "p", batteryCell, "p")
    ad.connect(currentSource, "n", batteryCell, "n")
    ad.connect(batteryCell, "n", gnd, "p")
    ad.connect(batteryCell, "heat", casing, "port")
    ad.connect(casing, "port", insulator, "port_a")
    ad.connect(insulator, "port_b", ambient, "port")
    ad.connect(casing, "port", tempSensor, "port_a")

    ac = AcausalCompiler(ev, ad, verbose=False)
    bat_sys_compiled = ac()
    builder = jaxonomy.DiagramBuilder()
    bat_sys = builder.add(bat_sys_compiled)
    diagram = builder.build()
    
    temp_idx = bat_sys.outsym_to_portid[tempSensor.get_sym_by_port_name("T_rel")]
    opts = SimulatorOptions(math_backend="jax", ode_solver_method="bdf", rtol=1e-10, atol=1e-12, enable_autodiff=True)

    def simulate_temp_fn(params):
        R0_val, C_val, R_ins_val = params[0], params[1], params[2]
        ctx = diagram.create_context()
        sub = ctx[bat_sys.system_id]
        sub = sub.with_parameter("bat_R0", R0_val)
        sub = sub.with_parameter("casing_C", C_val)
        sub = sub.with_parameter("insulator_R", R_ins_val)
        ctx = ctx.with_subcontext(bat_sys.system_id, sub)
        
        res = jaxonomy.simulate(diagram, ctx, (0.0, 10.0), options=opts, postprocess=False)
        return bat_sys.output_ports[temp_idx].eval(res.context)

    jax_grad_fn = jax.grad(simulate_temp_fn)
    test_params = jnp.array([0.02, 80.0, 2.0])
    
    print(f"\n[BATTERY] Test parameters: R0={test_params[0]}, casing_C={test_params[1]}, insulator_R={test_params[2]}")
    grad_jax = jax_grad_fn(test_params)
    grad_jax = np.array(grad_jax)

    # Finite differences
    eps = 1e-4
    grad_fd = []
    for i in range(len(test_params)):
        p_plus = np.array(test_params)
        p_plus[i] += eps
        val_plus = float(simulate_temp_fn(p_plus))
        
        p_minus = np.array(test_params)
        p_minus[i] -= eps
        val_minus = float(simulate_temp_fn(p_minus))
        
        grad_fd.append((val_plus - val_minus) / (2 * eps))
    grad_fd = np.array(grad_fd)

    print("\n--- BATTERY DAE GRADIENT TABLE ---")
    headers = ["Parameter", "Analytical Grad (JAX)", "Finite Diff Grad", "Abs Difference"]
    print(f"{headers[0]:<12} {headers[1]:<25} {headers[2]:<20} {headers[3]:<15}")
    print("-" * 75)
    names = ["R0 (ohmic)", "C (casing mass)", "R (insulation)"]
    for i, name in enumerate(names):
        diff = abs(grad_jax[i] - grad_fd[i])
        print(f"{name:<12} {grad_jax[i]:<25.8f} {grad_fd[i]:<20.8f} {diff:<15.8e}")
    print("-" * 75)
    
    max_diff = np.max(np.abs(grad_jax - grad_fd))
    print(f"Max Gradient Absolute Difference: {max_diff:.8e}")
    assert max_diff < 1e-4, f"Battery Gradient validation failed! max_diff={max_diff:.8e} >= 1e-4"
    print("[SUCCESS] Battery electro-thermal DAE gradient JAX autodiff matches finite differences!")
    return grad_jax.tolist(), grad_fd.tolist(), float(max_diff)


def get_modelica_battery_code():
    return """
model ElectroThermalBattery
  Modelica.Electrical.Analog.Basic.Ground ground;
  Modelica.Electrical.Analog.Sources.ConstantCurrent currentSource(I=10.0);
  Jaxonomy.Acausal.Battery.BatteryCellECM batteryCell(
    R0=0.02, R1=0.01, C1=1000.0, capacity_Ah=2.0,
    enable_heat_port=true, initial_soc=1.0, fixed_soc=true
  );
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor casing(C=80.0, T(start=298.15, fixed=true));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor insulator(G=0.5); // G = 1/R = 0.5 W/K
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature ambient(T=298.15);
equation
  connect(currentSource.p, batteryCell.p);
  connect(currentSource.n, batteryCell.n);
  connect(batteryCell.n, ground.p);
  connect(batteryCell.heat, casing.port);
  connect(casing.port, insulator.port_a);
  connect(insulator.port_b, ambient.port);
end ElectroThermalBattery;
"""


def main():
    print("==========================================================")
    print("STARTING S4 ACAUSAL / MODELICA PARITY RUNG 0 BENCHMARKS")
    print("==========================================================")
    
    # Parameters for RC ladder can be customized or run as defaults
    r1_val, c1_val = 1.0, 0.5
    r2_val, c2_val = 2.0, 1.5
    v_val = 5.0

    # 1. RC Ladder lowpass validation
    print("\nSimulating RC Ladder Lowpass Network...")
    t_rc, vc1_jax, vc2_jax = build_and_simulate_rc_ladder(v_val, r1_val, c1_val, r2_val, c2_val, t_end=5.0)
    
    vc1_jax_flat = np.array(vc1_jax).flatten()
    vc2_jax_flat = np.array(vc2_jax).flatten()
    t_rc_np = np.array(t_rc, dtype=np.float64)
    
    # Solve exact state-space ODE at the simulator's exact time steps
    vc1_ref, vc2_ref = solve_rc_ladder_state_space(v_val, r1_val, c1_val, r2_val, c2_val, t_eval=t_rc_np)
    
    rc1_err = np.max(np.abs(vc1_jax_flat - vc1_ref))
    rc2_err = np.max(np.abs(vc2_jax_flat - vc2_ref))
    print(f"RC Ladder Max Absolute Error Node 1: {rc1_err:.8e}")
    print(f"RC Ladder Max Absolute Error Node 2: {rc2_err:.8e}")
    
    assert rc1_err < 1e-6 and rc2_err < 1e-6, "RC Ladder simulation error exceeds 1e-6!"
    print("[SUCCESS] RC Ladder DAE matches state-space ODE reference to < 1e-6.")

    # Parameters for DC Motor
    v_motor_val = 24.0
    r_motor_val = 2.0
    kt_motor_val = 0.5
    ke_motor_val = 0.5
    j_motor_val = 0.1
    b_motor_val = 0.01

    # 2. DC Motor load validation
    print("\nSimulating DC Motor with load...")
    t_motor, speed_jax, angle_jax, current_jax = build_and_simulate_dc_motor(
        v_val=v_motor_val, r_val=r_motor_val, kt_val=kt_motor_val, ke_val=ke_motor_val, j_val=j_motor_val, b_val=b_motor_val, t_end=5.0
    )
    
    speed_jax_flat = np.array(speed_jax).flatten()
    angle_jax_flat = np.array(angle_jax).flatten()
    current_jax_flat = np.array(current_jax).flatten()
    t_motor_np = np.array(t_motor, dtype=np.float64)
    
    # Solve exact state-space ODE at the simulator's exact time steps
    speed_ref, angle_ref, current_ref = solve_dc_motor_state_space(
        v_val=v_motor_val, r_val=r_motor_val, kt_val=kt_motor_val, ke_val=ke_motor_val, j_val=j_motor_val, b_val=b_motor_val, t_eval=t_motor_np
    )
    
    speed_err = np.max(np.abs(speed_jax_flat - speed_ref))
    angle_err = np.max(np.abs(angle_jax_flat - angle_ref))
    current_err = np.max(np.abs(current_jax_flat - current_ref))
    
    print(f"DC Motor Max Speed Error:   {speed_err:.8e}")
    print(f"DC Motor Max Angle Error:   {angle_err:.8e}")
    print(f"DC Motor Max Current Error: {current_err:.8e}")
    
    assert speed_err < 1e-6 and angle_err < 1e-6 and current_err < 1e-6, "DC Motor error exceeds 1e-6!"
    print("[SUCCESS] DC Motor DAE matches state-space ODE reference to < 1e-6.")

    # 3. Battery Cell Electro-Thermal validation
    print("\nSimulating Electro-Thermal Battery Cell...")
    bat_r0, bat_r1, bat_c1, bat_cap = 0.02, 0.01, 1000.0, 2.0
    bat_i_discharge = 10.0
    bat_casing_c = 80.0
    bat_insulator_r = 2.0
    bat_t_ambient = 298.15
    bat_t_end = 20.0
    
    t_bat, soc_jax, v_rc_jax, temp_jax = build_and_simulate_battery_cell(
        r0_val=bat_r0, r1_val=bat_r1, c1_val=bat_c1, cap_val=bat_cap, i_discharge=bat_i_discharge,
        casing_c=bat_casing_c, insulator_r=bat_insulator_r, t_ambient=bat_t_ambient, t_end=bat_t_end
    )
    
    soc_jax_flat = np.array(soc_jax).flatten()
    v_rc_jax_flat = np.array(v_rc_jax).flatten()
    temp_jax_flat = np.array(temp_jax).flatten()
    t_bat_np = np.array(t_bat, dtype=np.float64)
    
    soc_ref, v_rc_ref, temp_ref = solve_battery_state_space(
        r0_val=bat_r0, r1_val=bat_r1, c1_val=bat_c1, cap_val=bat_cap, i_discharge=bat_i_discharge,
        casing_c=bat_casing_c, insulator_r=bat_insulator_r, t_ambient=bat_t_ambient, t_eval=t_bat_np
    )
    
    soc_err = np.max(np.abs(soc_jax_flat - soc_ref))
    v_rc_err = np.max(np.abs(v_rc_jax_flat - v_rc_ref))
    temp_err = np.max(np.abs(temp_jax_flat - temp_ref))
    
    print(f"Battery Max SOC Error:             {soc_err:.8e}")
    print(f"Battery Max V_RC Error:            {v_rc_err:.8e}")
    print(f"Battery Max Casing Temp Error:     {temp_err:.8e}")
    
    assert soc_err < 1e-6 and v_rc_err < 1e-6 and temp_err < 1e-6, "Battery simulation error exceeds 1e-6!"
    print("[SUCCESS] Electro-Thermal Battery Cell DAE matches state-space ODE reference to < 1e-6.")

    # 4. DAE Differentiability test (DC Motor & Battery)
    print("\nRunning DAE gradient checks...")
    motor_grad_jax, motor_grad_fd, motor_max_diff = run_acausal_gradient_check()
    battery_grad_jax, battery_grad_fd, battery_max_diff = run_battery_gradient_check()
    
    # 5. Print Modelica equivalent code
    modelica_rc, modelica_motor = print_modelica_code()
    modelica_battery = get_modelica_battery_code()
    print("3. ELECTRO-THERMAL BATTERY:")
    print(modelica_battery)
    print("==========================================================")
    
    # 6. Generate plots
    print("\nGenerating validation plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # Left plot: RC Ladder
    ax1.plot(t_rc_np, vc1_ref, 'k-', linewidth=3, label="Node 1 Reference (SciPy)")
    ax1.plot(t_rc_np, vc1_jax_flat, 'c--', linewidth=2, label="Node 1 Jaxonomy DAE")
    ax1.plot(t_rc_np, vc2_ref, 'r-', linewidth=3, label="Node 2 Reference (SciPy)")
    ax1.plot(t_rc_np, vc2_jax_flat, 'y--', linewidth=2, label="Node 2 Jaxonomy DAE")
    ax1.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Voltage (V)", fontsize=11, fontweight="bold")
    ax1.set_title("RC Ladder Network Transient Response", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True)
    
    # Middle plot: DC Motor
    ax2.plot(t_motor_np, speed_ref, 'k-', linewidth=3, label="Speed Reference (SciPy)")
    ax2.plot(t_motor_np, speed_jax_flat, 'g--', linewidth=2, label="Speed Jaxonomy DAE")
    ax2.plot(t_motor_np, current_ref, 'r-', linewidth=3, label="Current Reference (SciPy)")
    ax2.plot(t_motor_np, current_jax_flat, 'm--', linewidth=2, label="Current Jaxonomy DAE")
    ax2.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Amplitude (rad/s or A)", fontsize=11, fontweight="bold")
    ax2.set_title("DC Motor Speed and Current Transient Response", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True)

    # Right plot: Battery Casing Temperature
    ax3.plot(t_bat_np, temp_ref - 273.15, 'k-', linewidth=3, label="Casing Temp Reference (SciPy)")
    ax3.plot(t_bat_np, temp_jax_flat - 273.15, 'r--', linewidth=2, label="Casing Temp Jaxonomy DAE")
    ax3.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
    ax3.set_title("Battery Casing Temperature Rise", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(True)
    
    fig.suptitle("Campaign S4: Acausal DAE Simulation Parity & Correctness\n"
                 "Comparing Compiled Index-Reduced DAEs against Exact State-Space Solutions",
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acausal_parity_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved validation plot to {plot_path}")
    
    # Save JSON results
    results_db = {
        "rc_ladder": {
            "r1": r1_val, "c1": c1_val, "r2": r2_val, "c2": c2_val, "v_in": v_val,
            "max_abs_err_node1": float(rc1_err),
            "max_abs_err_node2": float(rc2_err),
        },
        "dc_motor": {
            "v_in": v_motor_val, "r": r_motor_val, "kt": kt_motor_val, "ke": ke_motor_val, "j": j_motor_val, "b": b_motor_val,
            "max_abs_err_speed": float(speed_err),
            "max_abs_err_angle": float(angle_err),
            "max_abs_err_current": float(current_err),
        },
        "battery_cell": {
            "r0": bat_r0, "r1": bat_r1, "c1": bat_c1, "capacity_Ah": bat_cap, "i_discharge": bat_i_discharge,
            "casing_c": bat_casing_c, "insulator_r": bat_insulator_r,
            "max_abs_err_soc": float(soc_err),
            "max_abs_err_v_rc": float(v_rc_err),
            "max_abs_err_temp": float(temp_err),
        },
        "differentiability": {
            "motor_grad_jax": motor_grad_jax,
            "motor_grad_fd": motor_grad_fd,
            "motor_max_grad_diff": float(motor_max_diff),
            "battery_grad_jax": battery_grad_jax,
            "battery_grad_fd": battery_grad_fd,
            "battery_max_grad_diff": float(battery_max_diff)
        },
        "modelica": {
            "rc_ladder": modelica_rc,
            "dc_motor": modelica_motor,
            "battery_cell": modelica_battery
        }
    }
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acausal_parity_results.json")
    with open(json_path, "w") as f:
        json.dump(results_db, f, indent=2)
    print(f"Saved parity results JSON to {json_path}")
    print("\nS4 Acausal Parity Benchmarks Completed Successfully!")


if __name__ == "__main__":
    main()
