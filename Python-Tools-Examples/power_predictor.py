import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# GLOBAL VARIABLES:
R_AIR = 287     # (J/kg*K) Gas Constant
GAMMA_AIR = 1.4     # Ratio of Constant Heats for Air

# GLOBAL FUNCTIONS
PI = np.pi
COS = np.cos
SIN = np.sin

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