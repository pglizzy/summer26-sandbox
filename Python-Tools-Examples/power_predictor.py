import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV IMPORT PATHS:
path_dyno_csv = "N/A"
path_rotating_assy_csv = "N/A"
path_fuel_csv = "N/A"

# GLOBAL CONSTANTS:
R_AIR = 287         # (J/kg*K) Gas Constant
GAMMA_AIR = 1.4     # Ratio of Constant Heats for Air
T_0 = 300           # (K) Ambient Air Temperature
P_0 = 101325        # (Pa) Ambient Air Pressure (Atmospheric Pressure, absolute)
k = 10              # Plenum volume to engine displacement ratio. 
BRAKE_EFFICIENCY = 0.3  # Estimated engine brake efficiency.
LAMBDA = 0.9        # Lambda value for AFR


# GLOBAL FUNCTIONS
PI = np.pi
COS = np.cos
SIN = np.sin

# GLOBAL ARRAYS:
CRANK_ANGLE_ARRAY_RADIANS = np.linspace(0, 4*PI, 1001) # Crank angle array from 0 to 720 degrees (4*pi radians) with 1001 points

# -----------------------------------------
#           RESTRICTOR FUNCTIONS
# -----------------------------------------

def restrictor_mass_flowrate(diam, T_0, p_0, p_plenum, Cd=1):
    """
    Calculate the mass flowrate of the air through the restrictor.

    INPUTS:
    - diam (mm) : throat diameter of the restrictor
    - T_0 (K) : upstream air temperature
    - p_0 (Pa) : upstream air pressure
    - p_plenum (Pa) : plenum (downstream) air pressure
    - Cd : Restrictor discharge coefficient. Assumed perfect restrictor, defaults to 1. Acceptable range [0,1]. Determine experimentally.

    OUTPUT: 
    - res_massflow (kg/s) : Restrictor air mass flow rate
    """
    GAMMA = GAMMA_AIR
    R = R_AIR

    if Cd > 1:
        raise ValueError("Cd must be less than or equal to 1.")
    
    diam_m = diam/1000  # Convert from mm to m
    area = PI * 0.25 * (diam_m ** 2)    # XEC Area, mm^2

    pratio = p_plenum / p_0     # Outlet/Inlet air pressure ratio
    gratio = (2/(GAMMA + 1))**(GAMMA/(GAMMA - 1))   # Heat ratio

    choking = pratio <= gratio  # Choking if ture

    if choking: # Calculate choked mass flow rate
        print("Choked flow. Calculating . . .\n")

        # Break Up the Equation to Make it Easier
        GR = (GAMMA + 1)/2
        GC = GR ** (-GR/(GAMMA - 1))
        GROOT = np.sqrt(GAMMA/R)
        forcepart = Cd * area * p_0

        # Calculate Mass Flow Rate
        res_massflow = (forcepart / np.sqrt(T_0)) * GROOT * GC

    else:
        print("Subsonic flow. Calculating . . .\n")

        # Break up the equation to make it easier
        forcepart = Cd * area * p_0
        energypart = 2 * GAMMA / (R * T_0 * (GAMMA - 1))
        pressurepart = (pratio ** (2/GAMMA)) - (pratio ** ((GAMMA + 1)/GAMMA))

        # Calculate Mass Flow Rate
        res_massflow = forcepart * np.sqrt(energypart * pressurepart)

    return res_massflow

# -----------------------------------------
#           ENGINE FUNCTIONS
# -----------------------------------------

def convert_rpm_to_angularspeed_4stroke(rpm):
    """
    Converts engine speed in RPM to angular speed in radians per second for a 4-stroke engine.

    INPUT:
    - rpm : engine speed in revolutions per minute

    OUTPUT:
    - omega (rad/s) : angular speed of the crankshaft
    """
    return (rpm * 2 * PI) / 60 / 2   # Divide by 2 for 4-stroke engine

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
    stroke_m = stroke / 1000
    lconrod_m = lconrod / 1000
    omega = convert_rpm_to_angularspeed_4stroke(rpm)

    # Calculate bore area
    A = PI * (bore_m / 2) ** 2

    # Calculate the piston velocity as a function of crank angle using slider-crank kinematics
    r = stroke_m / 2  # Crank radius, m
    l = lconrod_m     # Connecting rod length, m
    theta = crankangle  # crank angle, radians

    # Calculate Piston Velocity using slider-crank kinematics
    velocity = - omega * r * (SIN(theta) + r*SIN(theta)*COS(theta)/np.sqrt(l**2 - (r*SIN(theta))**2))

    # Calculate instantaneous volume change rate (dV/dt)
    dVdt = A * velocity

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

    area = PI * (bore_m / 2) ** 2
    swept_volume_per_cylinder = area * stroke_m
    total_displacement = swept_volume_per_cylinder * cylinders

    return total_displacement

def cylinder_head_volume(displacement, compressionratio):
    """
    Calculates the cylinder head (clearance) volume based on engine displacement and compression ratio.

    INPUTS:
    - displacement (m^3) : engine displacement volume
    - compressionratio : engine compression ratio (dimensionless)

    OUTPUT:
    - head_volume (m^3) : cylinder head (clearance) volume
    """
    return displacement / (compressionratio - 1)

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





def update_plenum_pressure(restrictordiam, Cd, crankangle, bore, stroke, lconrod, rpm, p_0=101325, T_0=300):
    """
    Updates the plenum pressure through the intake stroke relative to crankshaft angle of the piston.

    INPUTS:
    - restrictordiam (mm) : throat diameter of the restrictor
    - Cd : Restrictor discharge coefficient. Assumed perfect restrictor, defaults to 1. Acceptable range [0,1]. Determine experimentally.
    - crankangle (degrees) : crankshaft angle of the piston
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - lconrod (mm) : length of the connecting rod
    - rpm : engine speed in revolutions per minute
    - p_0 (Pa) : upstream air pressure, defaults to atmospheric pressure
    - T_0 (K) : upstream air temperature, defaults to 300 K

    OUTPUT:
    - p_plenum (Pa) : plenum pressure at the given crank angle
    """
    pass

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# ------------- MAIN FUNCTION -------------
# /////////////////////////////////////////

def main():
    pass

if __name__ == "__main__":
    main()