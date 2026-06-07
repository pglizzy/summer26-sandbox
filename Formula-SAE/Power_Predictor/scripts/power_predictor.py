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
from source import restrictor_mass_flowrate
from source import piston_velocity_m_s

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

# GLOBAL FUNCTIONS

# RESIDUAL THRESHOLDS:
RES_PRESSURE = 10E-4    # (nondimens.) pressure residual
RES_MASSFLOW = 10E-4   # (nondimens.) mass flow rate residual

# STEP SIZES:
dTHETA = 0.01       # (radians) Crank angle step size
dN = 50             # (RPM) Engine speed step size 

# CRANK ANGLE ARRAY:
ANGLE_STEPS = int(4 * np.pi / dTHETA) + 1
CRANK_ANGLE_ARRAY_RADIANS = np.linspace(0, 4*np.pi, ANGLE_STEPS) 

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

def ideal_air_mass_flowrate(displacement, rpm, T_ambient=300, P_ambient=101325):
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

def est_OEM_air_mass_flowrate(power, lambda_val, AFR_stoich, lower_heating_value, brake_efficiency):
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

    fuel_mass_flow_rate = power / (lower_heating_value * brake_efficiency)

    air_mass_flow_rate = fuel_mass_flow_rate * AFR_real

    return air_mass_flow_rate

def est_OEM_volumetric_efficiency_arr(power_arr, displacement, T_ambient=300, P_ambient=101325, LHV=43.4e6, lambda_val=0.9, brake_efficiency=0.3, AFR_stoich=14.7):
    """
    Calculates the volumetric efficiency curve of an OEM engine, provided the power curve and displacement.

    ASSUMPTIONS:
    - 93 Octane Fuel w/ LHV = 43.4 MJ/kg
    - Lambda = 0.9 (modifiable)
    - Brake Efficiency = 0.3 (modifiable)
    - Ambient Air Temperature = 300 K (modifiable)
    - Ambient Air Pressure = 101325 Pa (modifiable)

    INPUTS:
    - power_arr (W) : array of engine power output values and RPM values
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
    air_density = P_ambient / (R_AIR * T_ambient)  # Calculate ambient air density using ideal gas law

    ideal_mass_flow_arr = ideal_air_mass_flowrate(displacement, power_arr[:, 1], air_density)  # Calculate ideal mass flow rate array based on displacement and RPM

    est_mass_flow_arr = np.array([est_OEM_air_mass_flowrate(power, lambda_val, AFR_stoich, LHV, brake_efficiency) for power in power_arr[:, 0]])  # Estimate mass flow rate array based on power output and assumptions

    volumetric_efficiency_arr = est_mass_flow_arr / ideal_mass_flow_arr  # Calculate volumetric efficiency array as the ratio of estimated mass flow to ideal mass flow

    return volumetric_efficiency_arr

def engine_volume_demand_rate(intake_event_arr, crankangle_arr, ve_arr, bore, stroke, lconrod, rpm):
    """
    Calculates the dynamic engine volume demand as a function of the engine's intake event order, crank angle array, volumetric efficiency curve, and engine geometry.

    The intake event array is dictated by the engine cylinder count and firing order. See intake_event_arr() for more details. Array length is equal to the number of cylinders, and the element value indicates the angle at which the intake event starts for that cylinder in degrees. 

    For engines with overlapping intake events, volume demand will be higher than the instant cylinder volume demand rate at that crank angle.

    INPUTS:
    - Intake_event_arr (degrees) : array of crank angles at which each cylinder intake event begins.
    - crankangle_arr (radians) : array of crank angles over which to calculate the volume demand
    - ve_arr (dimensionless) : VE of engine at each crank angle
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - lconrod (mm) : length of the connecting rod
    - rpm : engine speed in revolutions per minute

    OUTPUT:
    - volume_demand_arr (m^3) : array of dynamic engine air volume demand rate throughout the engine cycle.

    """

    intake_duration = np.pi    # Length of intake event

    angles = np.asarray(crankangle_arr) # (radian) angle array
    max_angle = 4*np.pi # (radian) max angle of engine cycle
    ve_arr = np.asarray(ve_arr) # array of VE values at each crank angle

    if ve_arr.shape != angles.shape:
        raise ValueError("VE array and crank angle array must have the same shape.")

    # Ensure the intake event array occurs within the engine cycle
    intake_start_angles = np.radians(intake_event_arr) % max_angle

    # Initialize total volume demand array
    total_vol_demand_rate = np.zeros_like(angles, dtype=float)

    # Iterate through intake events and calculate volume demand contribution from each event at each crank angle
    for intake_start in intake_start_angles:
        
        # Show where the crank is relative to the start of intake event, ensure that the angle is positive and wraps around the engine cycle
        rel_angles = (angles - intake_start) % max_angle

        # If relative angle is less than intake duration, we know the intake is pulling air
        pulling = rel_angles < intake_duration

        # Create the cylinder's local volume demand rate array
        cyl_demand_rate = np.zeros_like(angles, dtype=float)

        cyl_demand_rate[pulling] = (
            ve_arr[pulling] 
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

# -----------------------------------------
#           ITERATIVE SOLVER
# -----------------------------------------

def iteratively_solve_power_output(engine : str, fuel : str):
    """
    Iteratively solves for the restricted power output of the engine. Will update with further details as the function is built.
    """

    # Ambient Air Constants:
    T_0 = 300               # (K) Ambient Air Temperature
    P_0 = 101325               # (Pa) Ambient air pressure
    R = R_AIR               # (J/kg*K) Gas Constant
    rho_0 = P_0 / (R * T_0) # (kg/m^3) Ambient Air Density

    # Plenum Constants:
    T_P = 310          # (K) Elevated Air Temp of the Plenum
    k = 10                  # Plenum Volume / Engine Displacement

    # Restrictor Constants:
    if fuel == "E85":
        diam_r = 19 / 1000   # (m) Restrictor Diameter
        
    else:
        diam_r = 20 / 1000  # (m) Restrictor Diameter

    A_r = np.pi * 0.25 * (diam_r**2)
    


    






# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ------------- MAIN FUNCTION -------------
# /////////////////////////////////////////

def main():
    engine = "Daytona675"
    fuel = "E85"
    iteratively_solve_power_output(engine=engine, fuel=fuel)

if __name__ == "__main__":
    main()