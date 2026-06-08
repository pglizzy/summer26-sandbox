import numpy as np

R_AIR = 287         # (J/kg*K) Gas Constant
GAMMA_AIR = 1.4     # Ratio of Specific Heats for Air


def restrictor_mass_flowrate(diam, T_0, p_0, p_plenum, Cd=1.0):
    """
    Calculate the mass flow rate of air through the intake restrictor.

    INPUTS:
    - diam (m): restrictor throat diameter
    - T_0 (K): upstream stagnation/ambient air temperature
    - p_0 (Pa): upstream stagnation/ambient air pressure
    - p_plenum (Pa): downstream plenum pressure
    - Cd: restrictor discharge coefficient. Defaults to 1.0.

    OUTPUT:
    - res_massflow (kg/s): restrictor air mass flow rate

    NOTES:
    - The function uses the standard quasi-steady compressible orifice/nozzle
      equations.
    - Flow is choked when p_plenum / p_0 is below the critical pressure ratio.
    - This function intentionally does not print during normal operation because
      it is called thousands of times inside the plenum solver.
    """
    gamma = GAMMA_AIR
    R = R_AIR

    if not 0 < Cd <= 1:
        raise ValueError("Cd must be in the range (0, 1].")
    if diam <= 0:
        raise ValueError("Restrictor diameter must be positive.")
    if T_0 <= 0:
        raise ValueError("Upstream temperature must be positive.")
    if p_0 <= 0:
        raise ValueError("Upstream pressure must be positive.")
    if p_plenum <= 0:
        raise ValueError("Plenum pressure must be positive.")

    area = np.pi * 0.25 * diam**2

    # Clamp pressure ratio to avoid non-physical reverse flow or numerical issues.
    # For this naturally aspirated model, p_plenum should normally be <= p_0.
    pratio = min(max(p_plenum / p_0, 1e-9), 1.0)
    critical_pratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    if pratio <= critical_pratio:
        # Choked flow: Mach 1 at the throat.
        res_massflow = (
            Cd
            * area
            * p_0
            * np.sqrt(gamma / (R * T_0))
            * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        )
    else:
        pressure_term = pratio ** (2 / gamma) - pratio ** ((gamma + 1) / gamma)
        res_massflow = (
            Cd
            * area
            * p_0
            * np.sqrt((2 * gamma / (R * T_0 * (gamma - 1))) * pressure_term)
        )

    return res_massflow
