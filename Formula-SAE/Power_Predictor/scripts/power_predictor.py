import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys

# Get the Project Root: Power_Predictor/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add root to Python Path
sys.path.append(str(PROJECT_ROOT))

# Get custom functions
from source.restrictorflow import restrictor_mass_flowrate
from source.engine_kinematics import piston_velocity_m_s

# Get data files
from source.data_loader import (
    load_dyno_curves,
    load_fuels,
    load_engine_specs
)
engine_specs = load_engine_specs()
fuels = load_fuels()
dyno_curves = load_dyno_curves()

# GLOBAL CONSTANTS:
R_AIR = 287         # (J/kg*K) Gas Constant
GAMMA_AIR = 1.4     # Ratio of Constant Heats for Air
BRAKE_EFFICIENCY = 0.3  # Estimated engine brake efficiency.
LAMBDA = 0.9        # Lambda value for AFR


# -----------------------------------------
#           ENGINE FUNCTIONS
# -----------------------------------------

def instant_cylinder_volume_change_rate(crankangle, bore, stroke, lconrod, rpm):
    """
    Calculates instantantaneous cylinder volume change rate (dV/dt) at a given crank angle and engine speed.

    INPUTS:
    - crankangle (radians) : crankshaft angle of the piston
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - lconrod (mm) : length of the connecting rod
    - rpm : engine speed in revolutions per minute

    OUTPUT:
    - dVdt (m^3/s) : instantaneous cylinder volume change rate

    REFERENCED FUNCTIONS and VARIABLES:
    - convert_rpm_to_angularspeed_4stroke(rpm) : converts engine speed in RPM to angular speed in radians per second for a 4-stroke engine
    """

    # Convert inputs to SI units
    bore_m = bore / 1000


    # Calculate bore area
    A = np.pi * (bore_m / 2) ** 2

    # Calculate Piston Velocity (m/s) using slider-crank kinematics
    piston_speed = piston_velocity_m_s(stroke, lconrod, rpm, crankangle)

    # Calculate instantaneous volume change rate (dV/dt)
    dVdt = A * piston_speed

    dVdt = np.maximum(dVdt, 0)

    return dVdt

def engine_displacement(bore, stroke, cylinders):
    """
    Calculates the engine displacement volume.

    INPUTS:
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - cylinders : number of cylinders in the engine

    OUTPUT:
    - displacement (m^3) : engine displacement volume
    """
    bore_m = bore / 1000
    stroke_m = stroke / 1000

    area = np.pi * (bore_m / 2) ** 2
    swept_volume_per_cylinder = area * stroke_m
    total_displacement = swept_volume_per_cylinder * cylinders

    return total_displacement

def ideal_air_mass_flowrate(displacement, 
                            rpm, 
                            T_ambient=300, 
                            P_ambient=101325):
    """
    Calculates the ideal air mass flow rate for a four-stroke engine.

    Since motorcycle engines typically use ITBs, it's reasonable to assume that the intake air density is equivalent to ambient air conditions, as there are no plenum affects to consider. If you were doing this for car engines with plenums, this estimation would be more complex and require additional modeling for the plenum pressure and temperature.

    INPUTS:
    - displacement (m^3) : engine displacement volume
    - rpm : engine speed in revolutions per minute
    - T_ambient (K) : ambient air temperature, defaults to 300 K
    - P_ambient (Pa) : ambient air pressure, defaults to 101325 Pa

    OUTPUT:
    - mass_flow_rate (kg/s) : ideal air mass flow rate
    """

    # Convert RPM to RPS (revolutions per second)
    rps = rpm / 60

    # Calculate ambient air density using ideal gas law
    air_density = P_ambient / (R_AIR * T_ambient)

    # Calculate Theoretical Mass Flow Rate:
    mdot = air_density * displacement * rps / 2  # Divide by 2 for 4-stroke engine

    return mdot

def est_OEM_air_mass_flowrate(power_w, 
                              lambda_val, 
                              AFR_stoich, 
                              lower_heating_value, 
                              brake_efficiency):
    """
    Estimate air mass flow rate at a specific power output value, using assumptions about fuel properties and engine efficiency.
    
    INPUTS:
    - power (W) : engine power output
    - lambda_val : lambda value for AFR
    - AFR_stoich : stoichiometric air-fuel ratio for the fuel being used
    - lower_heating_value (J/kg) : lower heating value of the fuel
    - brake_efficiency : estimated engine brake efficiency (dimensionless, between 0 and 1)
    
    OUTPUT:
    - mass_flow_rate (kg/s) : estimated air mass flow rate
    """

    AFR_real = lambda_val * AFR_stoich  # Calculate real air-fuel ratio based on lambda

    fuel_mass_flow_rate = power_w / (lower_heating_value * brake_efficiency)

    air_mass_flow_rate = fuel_mass_flow_rate * AFR_real

    return air_mass_flow_rate

def est_OEM_volumetric_efficiency_arr(power_w_arr, rpm_arr, displacement, T_ambient=300, P_ambient=101325, LHV=43.4e6, lambda_val=0.9, brake_efficiency=0.3, AFR_stoich=14.7):
    """
    Calculates the volumetric efficiency curve of an OEM engine, provided the power curve and displacement.

    ASSUMPTIONS:
    - 93 Octane Fuel w/ LHV = 43.4 MJ/kg
    - Lambda = 0.9 (modifiable)
    - Brake Efficiency = 0.3 (modifiable)
    - Ambient Air Temperature = 300 K (modifiable)
    - Ambient Air Pressure = 101325 Pa (modifiable)

    INPUTS:
    - power_arr (W) : array of engine power output
    - rpm_arr (RPM) : array of engine RPMs
    - displacement (m^3) : engine displacement volume
    - T_ambient (K) : ambient air temperature, defaults to 300 K
    - P_ambient (Pa) : ambient air pressure, defaults to 101325 Pa
    - LHV (J/kg) : lower heating value of the fuel, default 43.4 MJ/kg
    - lambda_val : lambda value for AFR, default 0.9
    - brake_efficiency : estimated engine brake efficiency, default 0.3
    - AFR_stoich : stoichiometric air-fuel ratio, default 14.7 (gasoline)

    OUTPUT:
    - volumetric_efficiency_arr (dimensionless) : array of volumetric efficiency values corresponding to the input power curve
    """

    if not rpm_arr.size == power_w_arr.size:
        raise IndexError("RPM and Power Arrays must be the same length.")

    # Calculate ideal mass flow rate array based on displacement and RPM
    ideal_mass_flow_arr = ideal_air_mass_flowrate(
        displacement=displacement,
        rpm=rpm_arr, 
        T_ambient=T_ambient,
        P_ambient=P_ambient)  

    # Estimate mass flow rate array based on power output and assumptions
    est_mass_flow_arr = est_OEM_air_mass_flowrate(
        power_w_arr, 
        lambda_val=lambda_val, 
        AFR_stoich=AFR_stoich, 
        lower_heating_value=LHV, 
        brake_efficiency=brake_efficiency)  
    
    if not ideal_mass_flow_arr.size == est_mass_flow_arr.size:
        raise IndexError("Ideal and Estimated OEM Mass Flow Rate Arrays must be the same length.")

    return est_mass_flow_arr / ideal_mass_flow_arr

def engine_volume_demand_rate(
    intake_event_arr,
    crankangle_arr,
    ve,
    bore,
    stroke,
    lconrod,
    rpm
):
    """
    Calculates dynamic engine volume demand rate as a function of crank angle.

    VE and RPM are scalar values for the current engine speed. VE is assumed
    constant throughout the engine cycle.

    INPUTS:
    - intake_event_arr (deg) : crank angles where each cylinder intake event begins
    - crankangle_arr (rad) : crank angle array over one full engine cycle
    - ve (dimensionless) : volumetric efficiency at the current RPM
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - lconrod (mm) : connecting rod length
    - rpm : engine speed in revolutions per minute

    OUTPUT:
    - total_vol_demand_rate (m^3/s) : engine volume demand rate vs crank angle
    """

    intake_duration = np.pi      # 180 deg intake event
    max_angle = 4 * np.pi        # 720 deg four-stroke cycle

    angles = np.asarray(crankangle_arr, dtype=float)
    intake_start_angles = np.radians(intake_event_arr) % max_angle

    ve = float(ve)
    rpm = float(rpm)

    total_vol_demand_rate = np.zeros_like(angles, dtype=float)

    for intake_start in intake_start_angles:

        # Local crank angle relative to the start of this cylinder's intake event
        rel_angles = (angles - intake_start) % max_angle

        # Cylinder is actively demanding air during intake stroke
        pulling = rel_angles < intake_duration

        cyl_demand_rate = np.zeros_like(angles, dtype=float)

        cyl_demand_rate[pulling] = (
            ve
            * instant_cylinder_volume_change_rate(
                rel_angles[pulling],
                bore,
                stroke,
                lconrod,
                rpm
            )
        )

        total_vol_demand_rate += cyl_demand_rate

    return total_vol_demand_rate

def engine_volume_demand_rate_at_angle(
    intake_event_arr,
    crankangle,
    ve,
    bore,
    stroke,
    lconrod,
    rpm
):
    """
    Calculates instantaneous engine volume demand rate at one crank angle.

    INPUTS:
    - intake_event_arr (deg) : crank angles where each cylinder intake event begins
    - crankangle (rad) : single crank angle to evaluate
    - ve (dimensionless) : volumetric efficiency at the current RPM
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - lconrod (mm) : connecting rod length
    - rpm : engine speed in revolutions per minute

    OUTPUT:
    - total_vol_demand_rate (m^3/s) : total engine volume demand rate at this crank angle
    """

    intake_duration = np.pi      # 180 deg intake event
    max_angle = 4 * np.pi        # 720 deg four-stroke cycle

    crankangle = float(crankangle) % max_angle
    ve = float(ve)
    rpm = float(rpm)

    intake_start_angles = np.radians(intake_event_arr) % max_angle

    total_vol_demand_rate = 0.0

    for intake_start in intake_start_angles:

        # Crank angle relative to the start of this cylinder's intake event
        rel_angle = (crankangle - intake_start) % max_angle

        # Cylinder only demands air during its intake stroke
        if rel_angle < intake_duration:

            cyl_demand_rate = (
                ve
                * instant_cylinder_volume_change_rate(
                    rel_angle,
                    bore,
                    stroke,
                    lconrod,
                    rpm
                )
            )

            total_vol_demand_rate += cyl_demand_rate

    return total_vol_demand_rate

def parse_crank_phasing(phasing_string):
    """
    Convert crank phasing string from CSV into a NumPy array.

    Example:
        "0_180_360_540" -> array([  0., 180., 360., 540.])
    """

    if phasing_string is None:
        raise ValueError("Crank phasing string is None")

    phasing_string = str(phasing_string).strip()

    if phasing_string == "":
        raise ValueError("Crank phasing string is empty")

    return np.array(
        [float(value) for value in phasing_string.split("_")],
        dtype=float
    )

def get_engine_data(engine_name : str, fuel_name : str):
    # ---------------------------
    # Get Engine Dyno Data Arrays:
    # ---------------------------

    engine_dyno = dyno_curves[dyno_curves["Engine"] == engine_name].copy()

    if engine_dyno.empty:
        raise ValueError(f"No Dyno Curve found for engine: {engine_name}")
    
    engine_dyno = engine_dyno.sort_values("RPM")

    rpm = engine_dyno["RPM"].to_numpy(dtype=float)
    torque_nm = engine_dyno["Torque_Nm"].to_numpy(dtype=float)
    power_kw = engine_dyno["Power_kW"].to_numpy(dtype=float)

    # ---------------------------
    # Get Engine Spec Values
    # ---------------------------

    rotating_specs = engine_specs[engine_specs["EngineID"] == engine_name]

    if rotating_specs.empty:
        raise ValueError(f"No specifications found for engine: {engine_name}")
    
    engine_spec = rotating_specs.iloc[0] # Ensure you're only grabbing a value, not an array

    cyl = int(engine_spec["Cylinders"])
    bore_mm = float(engine_spec["Bore_mm"])
    stroke_mm = float(engine_spec["Stroke_mm"])
    lconrod_mm = float(engine_spec["Conrod_Length_mm"])
    CR = float(engine_spec["CompRatio"])
    crankphase_str = engine_spec["Crank_Phasing_deg"]
    crankphase_arr = parse_crank_phasing(crankphase_str)

    # ---------------------------
    # Get Engine Fuel Data
    # ---------------------------

    matching_fuel = fuels[fuels["Fuel"] == fuel_name]

    if matching_fuel.empty:
        raise ValueError(f"No data found for fuel: {fuel_name}")
    
    fuel = matching_fuel.iloc[0]    # Ensure only grabbing a value, not array

    fuel_density = float(fuel["Density_kg/L"])
    LHV = float(fuel["LHV_MJ/kg"]) * 1e6   # J/kg
    AFR = float(fuel["AFR"])

    # ---------------------------
    # Return All Engine Data
    # ---------------------------

    return {
        "engine_name" : engine_name,
        "fuel_name" : fuel_name,

        "rpm" : rpm,
        "torque_nm" : torque_nm,
        "power_kw" : power_kw,

        "cyl" : cyl,
        "bore_mm" : bore_mm,
        "stroke_mm" : stroke_mm,
        "lconrod_mm" : lconrod_mm,
        "CR" : CR,
        "crankphases_deg" : crankphase_arr,
        
        "fuel_density" : fuel_density,
        "LHV" : LHV,
        "AFR" : AFR
    }

def get_OEM_Fuel_data():
    fuel_name = "93"
    matching_fuel = fuels[fuels["Fuel"] == fuel_name]

    if matching_fuel.empty:
        raise ValueError(f"No data found for fuel: {fuel_name}")
    
    fuel = matching_fuel.iloc[0]    # Ensure only grabbing a value, not array

    fuel_density = float(fuel["Density_kg/L"])
    LHV = float(fuel["LHV_MJ/kg"])
    AFR = float(fuel["AFR"])

    return{
        "fuel_density" : fuel_density,
        "LHV" : LHV,
        "AFR" : AFR
    }

# -----------------------------------------
#           ITERATIVE SOLVER
# -----------------------------------------

def iteratively_solve_power_output(engine_name : str, fuel_name : str):
    """
    Iteratively solves for the restricted power output of the engine. Will update with further details as the function is built.
    """

    # ------------------------------------
    #       FUNCTION BASE PARAMETERS
    # --------- modify if needed ---------
    # ------------------------------------

    # AMBIENT AIR CONSTANTS:
    T_0 = 300               # (K) Ambient Air Temperature
    P_0 = 101325               # (Pa) Ambient air pressure
    R = R_AIR               # (J/kg*K) Gas Constant

    # PLENUM CONSTANTS:
    T_PLENUM = 310          # (K) Elevated Air Temp of the Plenum
    k = 10                  # Plenum Volume / Engine Displacement

    # ITERATION CONTROL PARAMETERS:
    MAX_ITERS = 1000        # Maximum iterations
    RES_PRESSURE = 10E-4    # (nondimens.) pressure residual
    THETA_STEP = 0.001      # (rad) Crank Angle Step Size

    # CRANK ANGLE ARRAY:
    CRANK_ANGLE_ARRAY_RADIANS = np.arange(0, 4*np.pi + THETA_STEP, THETA_STEP)
    dtheta =  CRANK_ANGLE_ARRAY_RADIANS[1] - CRANK_ANGLE_ARRAY_RADIANS[0]

    # ------------------------------------ 
    #           FUNCTION OPTIONS:
    # --------- modify if needed ---------
    # ------------------------------------ 
    FACTOR_FUEL_RATIO = True   # Option to consider fuel change
    fuel_ratio=1                # Value otherwise

    FACTOR_PUMPING_LOSSES = True   # Option to consider pumping losses
    pumping_loss_power = 0          # Value otherwise

    UNDERRELAX = False      # Option to underrelax solver for stability
    alpha = 0.3             # Under relaxation factor

    # ------------------------------------
    #        GET ENGINE & FUEL DATA
    # ------------------------------------   

    # Get the engine and fuel data from the CSV files. 
    data = get_engine_data(engine_name=engine_name, fuel_name=fuel_name)

    # OEM Dyno Curve Data for the engine
    rpm_arr = data["rpm"]
    oem_power_kw_arr = data["power_kw"]

    # Engine Specs
    bore_mm = data["bore_mm"]
    stroke_mm = data["stroke_mm"]
    lconrod_mm = data["lconrod_mm"] # Connecting rod length
    CYL = data["cyl"]   # number of cylinders

    # Get the phases for the cylinders
    crankphases_deg = data["crankphases_deg"]

    # SELECTED FUEL PROPERTIES:
    LHV = data["LHV"]   # Lower Heating Value (MJ/kg)
    AFR = data["AFR"]   # Air Fuel Ratio
    OEM_fuel_data = get_OEM_Fuel_data()
    OEM_LHV = OEM_fuel_data["LHV"] * 1e6 # J/kg 
    OEM_AFR = OEM_fuel_data["AFR"]
    OEM_LAMBDA = 0.9

    OEM_fuel_energy = OEM_LHV / (OEM_LAMBDA * OEM_AFR)

    # RESTRICTOR SIZE:
    if fuel_name == "E85":
        diam_r = 19 / 1000   # (m) Restrictor Diameter
        
    else:
        diam_r = 20 / 1000  # (m) Restrictor Diameter

    # Calculate engine displacement and plenum volume
    disp_m3 = engine_displacement(bore_mm, stroke_mm, CYL)
    V_PLENUM = k*disp_m3 # (m^3) Plenum displacement

    # Calculate the RPM-based volumetric efficiency
    oem_power_w_arr = oem_power_kw_arr * 1000
    VE_arr = est_OEM_volumetric_efficiency_arr(
        oem_power_w_arr, 
        rpm_arr,
        disp_m3, 
        T_0, 
        P_0, 
        LHV=OEM_LHV, 
        lambda_val=OEM_LAMBDA, 
        brake_efficiency=BRAKE_EFFICIENCY, 
        AFR_stoich=OEM_AFR)

    # Create the restricted power array
    power_restricted_kw = np.zeros_like(oem_power_kw_arr)

    for rpm_idx, speed in enumerate(rpm_arr):
        # Initialize the plenum to be ambient pressure
        P_PLENUM = P_0
        P_PLENUM_ARRAY = np.zeros_like(CRANK_ANGLE_ARRAY_RADIANS)

        # Get the OEM values for VE, Power, Mass Flow
        VE = VE_arr[rpm_idx]
        OEM_power_w = oem_power_kw_arr[rpm_idx] * 1000
        oem_mrate = est_OEM_air_mass_flowrate(
            OEM_power_w, 
            lambda_val=OEM_LAMBDA, 
            AFR_stoich=OEM_AFR, 
            lower_heating_value=OEM_LHV, 
            brake_efficiency=BRAKE_EFFICIENCY)

        # Calculate angular speed in rad/s
        omega = speed * 2 * np.pi / 60

        iter_count = 0  # Initialize the iteration counter
        converged = False

        # Begin the pressure solver
        while iter_count < MAX_ITERS:

            P_START = P_PLENUM
            P_PLENUM_ARRAY[0] = P_START

            for theta_idx in range(len(CRANK_ANGLE_ARRAY_RADIANS) - 1):

                theta = CRANK_ANGLE_ARRAY_RADIANS[theta_idx]
                P_current = P_PLENUM_ARRAY[theta_idx]

                # Calculate plenum air density
                rho_plenum = P_current / (R * T_PLENUM)

                # Calculate engine volume demand
                v_demand_rate = engine_volume_demand_rate_at_angle(
                    crankphases_deg, 
                    theta, 
                    VE, 
                    bore_mm, 
                    stroke_mm, 
                    lconrod_mm, 
                    speed)

                # Calculate Air Mass Flow Demand:
                m_demand_rate = rho_plenum * v_demand_rate

                # Calculate the restrictor mass flow rate
                m_restrictor = restrictor_mass_flowrate(
                    diam_r, 
                    T_0=T_0, 
                    p_0=P_0, 
                    p_plenum=P_current)

                # Calculate the plenum pressure change
                dPdtheta = (
                    R * T_PLENUM / (omega * V_PLENUM)
                    ) * (m_restrictor - m_demand_rate)

                # Update Plenum Pressure
                P_PLENUM_ARRAY[theta_idx + 1] = P_current + (dPdtheta * dtheta)

            P_END = P_PLENUM_ARRAY[-1]
            
            res = abs(P_START - P_END) / P_START

            if res < RES_PRESSURE:

                print(f"Plenum Pressure Solver Converged for {speed} RPM after {iter_count} iterations.")
                
                converged = True
                break

            if not UNDERRELAX:
                P_PLENUM = P_END

            else:
                P_PLENUM = P_START + alpha * (P_END - P_START)

            iter_count += 1
        
        if not converged:
            raise RuntimeError(
                f"ATTENTION: Plenum Pressure Solver did not converge within "
                f"the maximum {MAX_ITERS} iterations.\n"
                f"Engine Speed: {speed}\n"
                f"Residual: {res:.3e}\n"
                f"It is recommended to adjust solution setup parameters.\n"
                f"Solver stopped. Exiting . . ."
            )

        # Calculate the engine mass flow rate array for the converged solution
        m_converged_rates = np.zeros_like(CRANK_ANGLE_ARRAY_RADIANS)
        for m_idx, theta in enumerate(CRANK_ANGLE_ARRAY_RADIANS):

            rho = P_PLENUM_ARRAY[m_idx] / (R * T_PLENUM)

            v_converged_rate = engine_volume_demand_rate_at_angle(
                crankphases_deg,
                theta,
                VE, 
                bore_mm,
                stroke_mm,
                lconrod_mm,
                speed
            )

            m_converged_rates[m_idx] = rho * v_converged_rate
                    
        # Integrate to get effective average mass flow rate
        m_cycle = 0  # Initialize
        dt = dtheta / omega
        m_cycle = np.sum(m_converged_rates[:-1]) * dt
        
        # Effective average air mass flow rate over a cycle
        m_eff_avg_rate = m_cycle * speed / 120

        # Calc the ratio of mass airflows from restricted to OEM
        mass_ratio = m_eff_avg_rate / oem_mrate

        # Optional: Factor energy gain/loss from fuel selection
        if FACTOR_FUEL_RATIO:
            engine_fuel_energy = LHV / (LAMBDA * AFR)

            fuel_ratio = engine_fuel_energy / OEM_fuel_energy

        # Optional: Factor pumping losses due to plenum
        if FACTOR_PUMPING_LOSSES:
            mean_plenum_pressure = 0  # (Pa/rad) Initialize
            mean_plenum_pressure = np.sum(P_PLENUM_ARRAY[:-1]) * dtheta / (4*np.pi)

            pmep = P_0 - mean_plenum_pressure

            pumping_loss_power = pmep * disp_m3 * speed / 120

            if pumping_loss_power < 0:
                raise ValueError("Shits fucked. Higher plenum pressure than ambient, generates negative pumping losses.")

        # Calculate the restricted power output of the engine at the RPM
        power_restricted_kw[rpm_idx] = ((OEM_power_w * mass_ratio * fuel_ratio) - pumping_loss_power) / 1000

    return {
        "rpm" : rpm_arr,
        "power_oem_kw" : oem_power_kw_arr,
        "power_restricted_kw" : power_restricted_kw
    }


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ------------- MAIN FUNCTION -------------
# /////////////////////////////////////////

def main():
    engine = "Daytona675"
    fuel = "E85"
    iteratively_solve_power_output(engine_name=engine, fuel_name=fuel)

if __name__ == "__main__":
    main()