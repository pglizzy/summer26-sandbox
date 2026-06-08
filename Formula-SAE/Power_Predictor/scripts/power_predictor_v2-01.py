import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import re

# Get the Project Root: Power_Predictor/
# This assumes this file is located in Power_Predictor/scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add root to Python Path so imports like source.restrictorflow work when this
# script is run directly from the scripts/ folder.
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
R_AIR = 287              # (J/kg*K) Gas Constant for Air
GAMMA_AIR = 1.4          # Ratio of specific heats for air. Kept here for reference.
BRAKE_EFFICIENCY = 0.3   # Estimated engine brake efficiency.
LAMBDA = 0.9             # Lambda value for AFR in the restricted configuration.
OEM_FUEL_NAME = "93"     # Fuel assumed for the OEM dyno curve and OEM VE extraction.


# -----------------------------------------
#           OUTPUT / PLOTTING HELPERS
# -----------------------------------------

def sanitize_for_path(text):
    """
    Convert an arbitrary string into a filesystem-safe folder/file token.

    This is used so the run folder follows the requested ENGINE_FUEL naming
    convention without accidentally creating nested folders or invalid paths.
    """
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text).strip())


def make_output_dir(engine_name, fuel_name):
    """
    Create the output directory for a run.

    IMPORTANT PROJECT RULE:
    All generated files are saved under:
        Power_Predictor/data/output/ENGINE_FUEL/

    No plots, CSVs, or error files are written outside data/output/.
    """
    run_label = f"{sanitize_for_path(engine_name)}_{sanitize_for_path(fuel_name)}"
    output_dir = PROJECT_ROOT / "data" / "output" / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(df, output_dir, filename):
    """
    Save a DataFrame under the run output directory.
    """
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


def select_every_1000_rpm_indices(rpm_arr):
    """
    Select one RPM index per 1000-RPM bin for detailed mass-flow plots.

    If the dyno data contains exact 1000-RPM points, those are selected. If not,
    this selects the closest available point to each 1000-RPM target within the
    data range.
    """
    rpm_arr = np.asarray(rpm_arr, dtype=float)
    if rpm_arr.size == 0:
        return []

    rpm_min = int(np.ceil(np.nanmin(rpm_arr) / 1000.0) * 1000)
    rpm_max = int(np.floor(np.nanmax(rpm_arr) / 1000.0) * 1000)

    selected_indices = []
    used_indices = set()

    for target_rpm in range(rpm_min, rpm_max + 1, 1000):
        idx = int(np.argmin(np.abs(rpm_arr - target_rpm)))
        if idx not in used_indices:
            selected_indices.append(idx)
            used_indices.add(idx)

    return selected_indices


def plot_residual_history(residual_histories, output_dir):
    """
    Plot solver pressure residual vs. cycle iteration for each RPM.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for rpm, residuals in residual_histories.items():
        if len(residuals) == 0:
            continue
        ax.semilogy(np.arange(1, len(residuals) + 1), residuals, label=f"{rpm:.0f} RPM")

    ax.set_title("Plenum Pressure Solver Residual History")
    ax.set_xlabel("Cycle Iteration")
    ax.set_ylabel("Relative Pressure Residual, |P_end - P_start| / P_start")
    ax.grid(True, which="both", alpha=0.35)

    # Avoid unreadable legends if many RPMs are present.
    if len(residual_histories) <= 15:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "solver_residual_history.png", dpi=200)
    plt.close(fig)


def plot_stacked_traces(trace_dict, output_dir, y_key, y_label, title, filename_prefix, max_plots_per_file=8):
    """
    Create stacked crank-angle trace plots, grouped into multiple PNGs if needed.

    This avoids creating a single extremely tall figure when the RPM sweep has a
    large number of speed points.
    """
    rpm_values = sorted(trace_dict.keys())

    for file_idx, start in enumerate(range(0, len(rpm_values), max_plots_per_file), start=1):
        rpm_chunk = rpm_values[start:start + max_plots_per_file]
        fig_height = max(2.2 * len(rpm_chunk), 4.0)
        fig, axes = plt.subplots(
            len(rpm_chunk),
            1,
            figsize=(11, fig_height),
            sharex=True
        )

        if len(rpm_chunk) == 1:
            axes = [axes]

        for ax, rpm in zip(axes, rpm_chunk):
            trace = trace_dict[rpm]
            ax.plot(trace["theta_deg"], trace[y_key])
            ax.set_ylabel(f"{rpm:.0f} RPM\n{y_label}")
            ax.grid(True, alpha=0.35)

        axes[-1].set_xlabel("Crank Angle (deg)")
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"{filename_prefix}_{file_idx:02d}.png", dpi=200)
        plt.close(fig)


def plot_mass_flow_detail(trace_dict, rpm_arr, output_dir):
    """
    Plot converged restrictor mass flow rate and engine mass demand together.

    One plot is created for each 1000-RPM target represented in the dyno data.
    """
    selected_indices = select_every_1000_rpm_indices(rpm_arr)
    rpm_values = np.asarray(rpm_arr, dtype=float)

    for idx in selected_indices:
        rpm = float(rpm_values[idx])
        # Exact float keys from rpm_arr are used when trace_dict is built.
        trace = trace_dict.get(rpm)
        if trace is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(trace["theta_deg"], trace["mdot_restrictor_kg_s"], label="Restrictor Inflow")
        ax.plot(trace["theta_deg"], trace["mdot_engine_kg_s"], label="Engine Demand")
        ax.set_title(f"Converged Mass Flow Rates - {rpm:.0f} RPM")
        ax.set_xlabel("Crank Angle (deg)")
        ax.set_ylabel("Mass Flow Rate (kg/s)")
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"mass_flow_restrictor_vs_engine_{rpm:.0f}rpm.png", dpi=200)
        plt.close(fig)


def plot_mass_balance_error(summary_df, output_dir):
    """
    Plot restrictor-engine cycle mass balance error vs. RPM.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary_df["rpm"], summary_df["mass_balance_error"], marker="o")
    ax.set_title("Restrictor-Engine Cycle Mass Balance Error")
    ax.set_xlabel("Engine Speed (RPM)")
    ax.set_ylabel("Relative Mass Error")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "mass_balance_error_vs_rpm.png", dpi=200)
    plt.close(fig)


def plot_power_comparison(summary_df, output_dir):
    """
    Plot OEM and restricted power curves.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary_df["rpm"], summary_df["power_oem_kw"], label="OEM")
    ax.plot(summary_df["rpm"], summary_df["power_restricted_kw"], label="Restricted")
    ax.set_title("OEM vs. Restricted Power")
    ax.set_xlabel("Engine Speed (RPM)")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "power_oem_vs_restricted.png", dpi=200)
    plt.close(fig)


# -----------------------------------------
#           ENGINE FUNCTIONS
# -----------------------------------------

def instant_cylinder_volume_change_rate(crankangle, bore, stroke, lconrod, rpm):
    """
    Calculates instantaneous cylinder volume change rate (dV/dt) at a given crank angle and engine speed.

    INPUTS:
    - crankangle (radians) : crankshaft angle of the piston
    - bore (mm) : cylinder bore diameter
    - stroke (mm) : cylinder stroke length
    - lconrod (mm) : length of the connecting rod
    - rpm : engine speed in revolutions per minute

    OUTPUT:
    - dVdt (m^3/s) : instantaneous cylinder volume change rate

    REFERENCED FUNCTIONS and VARIABLES:
    - piston_velocity_m_s(stroke, lconrod, rpm, crankangle) : calculates piston velocity using slider-crank kinematics

    SIGN CONVENTION:
    - Positive piston velocity is assumed to increase cylinder volume.
    - Negative dV/dt values are clipped to zero because this reduced-order intake model only treats the engine as an intake demand sink during volume-increasing motion.
    """

    # Convert bore input to SI units. Stroke and connecting rod length are
    # passed directly into piston_velocity_m_s(), which is expected to handle
    # those units consistently with the source.engine_kinematics module.
    bore_m = bore / 1000

    # Calculate bore area.
    A = np.pi * (bore_m / 2) ** 2

    # Calculate piston velocity (m/s) using slider-crank kinematics.
    # The plenum solver uses crank angle in radians, while the shared
    # engine_kinematics.py functions use crank angle in degrees. Convert here
    # so the rest of the solver can stay radian-based.
    piston_speed = piston_velocity_m_s(stroke, lconrod, rpm, np.rad2deg(crankangle))

    # Calculate instantaneous volume change rate (dV/dt).
    dVdt = A * piston_speed

    # Only positive volume change is counted as intake demand.
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

    Since motorcycle engines typically use ITBs, it is reasonable to assume the
    OEM WOT intake reference density is approximately ambient air density. In the
    restricted system, this assumption is replaced by the dynamic plenum density.

    INPUTS:
    - displacement (m^3) : engine displacement volume
    - rpm : engine speed in revolutions per minute
    - T_ambient (K) : ambient air temperature, defaults to 300 K
    - P_ambient (Pa) : ambient air pressure, defaults to 101325 Pa

    OUTPUT:
    - mass_flow_rate (kg/s) : ideal air mass flow rate at 100% volumetric efficiency
    """

    # Convert RPM to RPS (revolutions per second).
    rps = np.asarray(rpm, dtype=float) / 60

    # Calculate ambient air density using ideal gas law.
    air_density = P_ambient / (R_AIR * T_ambient)

    # Calculate theoretical mass flow rate. Divide by 2 because a four-stroke
    # engine ingests one displacement volume every two crankshaft revolutions.
    mdot = air_density * displacement * rps / 2

    return mdot


def est_OEM_air_mass_flowrate(power_w,
                              lambda_val,
                              AFR_stoich,
                              lower_heating_value,
                              brake_efficiency):
    """
    Estimate air mass flow rate at a specific power output value using fuel properties and engine efficiency.

    INPUTS:
    - power_w (W) : engine brake power output
    - lambda_val : lambda value for AFR
    - AFR_stoich : stoichiometric air-fuel ratio for the fuel being used
    - lower_heating_value (J/kg) : lower heating value of the fuel
    - brake_efficiency : estimated engine brake efficiency (dimensionless, between 0 and 1)

    OUTPUT:
    - mass_flow_rate (kg/s) : estimated air mass flow rate
    """

    AFR_real = lambda_val * AFR_stoich
    fuel_mass_flow_rate = np.asarray(power_w, dtype=float) / (lower_heating_value * brake_efficiency)
    air_mass_flow_rate = fuel_mass_flow_rate * AFR_real

    return air_mass_flow_rate


def est_OEM_volumetric_efficiency_arr(power_w_arr,
                                      rpm_arr,
                                      displacement,
                                      T_ambient=300,
                                      P_ambient=101325,
                                      LHV=43.4e6,
                                      lambda_val=0.9,
                                      brake_efficiency=0.3,
                                      AFR_stoich=14.7):
    """
    Calculates the effective volumetric efficiency curve of an OEM engine.

    ASSUMPTIONS:
    - OEM dyno curve is interpreted as a WOT, near-atmospheric ITB breathing curve.
    - Lambda is constant unless supplied otherwise.
    - Brake efficiency is constant unless supplied otherwise.
    - Ambient air density is the OEM reference density.

    INPUTS:
    - power_w_arr (W) : array of engine power output
    - rpm_arr (RPM) : array of engine RPMs
    - displacement (m^3) : engine displacement volume
    - T_ambient (K) : ambient air temperature, defaults to 300 K
    - P_ambient (Pa) : ambient air pressure, defaults to 101325 Pa
    - LHV (J/kg) : lower heating value of the OEM fuel
    - lambda_val : lambda value for AFR
    - brake_efficiency : estimated engine brake efficiency
    - AFR_stoich : stoichiometric air-fuel ratio for OEM fuel

    OUTPUT:
    - volumetric_efficiency_arr (dimensionless) : effective VE values corresponding to the input power curve
    """

    power_w_arr = np.asarray(power_w_arr, dtype=float)
    rpm_arr = np.asarray(rpm_arr, dtype=float)

    if rpm_arr.size != power_w_arr.size:
        raise IndexError("RPM and Power Arrays must be the same length.")

    # Calculate ideal mass flow rate array based on displacement and RPM.
    ideal_mass_flow_arr = ideal_air_mass_flowrate(
        displacement=displacement,
        rpm=rpm_arr,
        T_ambient=T_ambient,
        P_ambient=P_ambient
    )

    # Estimate OEM mass flow rate array based on power output and assumptions.
    est_mass_flow_arr = est_OEM_air_mass_flowrate(
        power_w_arr,
        lambda_val=lambda_val,
        AFR_stoich=AFR_stoich,
        lower_heating_value=LHV,
        brake_efficiency=brake_efficiency
    )

    if ideal_mass_flow_arr.size != est_mass_flow_arr.size:
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

    intake_duration = np.pi      # 180 deg intake event.
    max_angle = 4 * np.pi        # 720 deg four-stroke cycle.

    angles = np.asarray(crankangle_arr, dtype=float)
    intake_start_angles = np.radians(intake_event_arr) % max_angle

    ve = float(ve)
    rpm = float(rpm)

    total_vol_demand_rate = np.zeros_like(angles, dtype=float)

    for intake_start in intake_start_angles:

        # Local crank angle relative to the start of this cylinder's intake event.
        rel_angles = (angles - intake_start) % max_angle

        # Cylinder is actively demanding air during intake stroke.
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

    intake_duration = np.pi      # 180 deg intake event.
    max_angle = 4 * np.pi        # 720 deg four-stroke cycle.

    crankangle = float(crankangle) % max_angle
    ve = float(ve)
    rpm = float(rpm)

    intake_start_angles = np.radians(intake_event_arr) % max_angle

    total_vol_demand_rate = 0.0

    for intake_start in intake_start_angles:

        # Crank angle relative to the start of this cylinder's intake event.
        rel_angle = (crankangle - intake_start) % max_angle

        # Cylinder only demands air during its intake stroke.
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

    In this reduced-order model, these values are treated as intake-event start
    angles over a 720-degree four-stroke cycle.
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


def get_fuel_data(fuel_name: str):
    """
    Return fuel data from the loaded fuel table.

    OUTPUT UNITS:
    - fuel_density: kg/L
    - LHV: J/kg
    - AFR: stoichiometric air/fuel ratio by mass
    """
    matching_fuel = fuels[fuels["Fuel"] == fuel_name]

    if matching_fuel.empty:
        raise ValueError(f"No data found for fuel: {fuel_name}")

    fuel = matching_fuel.iloc[0]

    return {
        "fuel_density": float(fuel["Density_kg/L"]),
        "LHV": float(fuel["LHV_MJ/kg"]) * 1e6,
        "AFR": float(fuel["AFR"]),
    }


def get_engine_data(engine_name: str, fuel_name: str):
    """
    Collect OEM dyno data, engine geometry, and selected restricted-fuel data.
    """

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
    # Get Engine Spec Values:
    # ---------------------------
    # The dyno curve and rotating-assembly tables should ideally use the same
    # engine identifier. Some OEM names differ slightly between the two files
    #; for example, the dyno data may use "Daytona675" while the rotating
    # assembly table may use "Daytona675R". Try an exact match first, then
    # fall back to a conservative prefix/normalized-name match.
    rotating_specs = engine_specs[engine_specs["EngineID"] == engine_name]

    if rotating_specs.empty:
        engine_name_normalized = str(engine_name).lower().replace(" ", "").replace("-", "").replace("_", "")

        engine_id_series = engine_specs["EngineID"].astype(str)
        engine_id_normalized = (
            engine_id_series
            .str.lower()
            .str.replace(" ", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace("_", "", regex=False)
        )

        fallback_mask = (
            engine_id_normalized.str.startswith(engine_name_normalized)
            | engine_id_normalized.eq(engine_name_normalized.rstrip("r"))
        )
        rotating_specs = engine_specs[fallback_mask]

    if rotating_specs.empty:
        available_specs = ", ".join(engine_specs["EngineID"].dropna().astype(str).unique())
        raise ValueError(
            f"No specifications found for engine: {engine_name}. "
            f"Available EngineID values are: {available_specs}"
        )

    if len(rotating_specs) > 1:
        matches = ", ".join(rotating_specs["EngineID"].astype(str).tolist())
        raise ValueError(
            f"Multiple specification rows matched engine {engine_name}: {matches}. "
            "Please make the Engine name in the dyno CSV match one EngineID exactly."
        )

    engine_spec = rotating_specs.iloc[0]

    cyl = int(engine_spec["Cylinders"])
    bore_mm = float(engine_spec["Bore_mm"])
    stroke_mm = float(engine_spec["Stroke_mm"])
    lconrod_mm = float(engine_spec["Conrod_Length_mm"])
    CR = float(engine_spec["CompRatio"])
    crankphase_str = engine_spec["Crank_Phasing_deg"]
    crankphase_arr = parse_crank_phasing(crankphase_str)

    # ---------------------------
    # Get Selected Fuel Data:
    # ---------------------------
    fuel = get_fuel_data(fuel_name)

    # ---------------------------
    # Return All Engine Data:
    # ---------------------------
    return {
        "engine_name": engine_name,
        "fuel_name": fuel_name,

        "rpm": rpm,
        "torque_nm": torque_nm,
        "power_kw": power_kw,

        "cyl": cyl,
        "bore_mm": bore_mm,
        "stroke_mm": stroke_mm,
        "lconrod_mm": lconrod_mm,
        "CR": CR,
        "crankphases_deg": crankphase_arr,

        "fuel_density": fuel["fuel_density"],
        "LHV": fuel["LHV"],
        "AFR": fuel["AFR"],
    }


def get_OEM_Fuel_data():
    """
    Return OEM fuel data for the OEM dyno-derived VE calculation.

    The OEM dyno curve is assumed to have been generated on OEM_FUEL_NAME.
    """
    return get_fuel_data(OEM_FUEL_NAME)


def calculate_restrictor_diameter_m(fuel_name):
    """
    Return restrictor diameter in meters based on selected fuel.

    Current Formula SAE-style rule assumption:
    - E85: 19 mm
    - Other fuels: 20 mm
    """
    if str(fuel_name).upper() == "E85":
        return 19 / 1000
    return 20 / 1000


def sweep_one_cycle(
    p_start,
    crank_angle_array,
    dtheta,
    omega,
    speed,
    ve,
    crankphases_deg,
    bore_mm,
    stroke_mm,
    lconrod_mm,
    diam_r,
    T_0,
    P_0,
    R,
    T_PLENUM,
    V_PLENUM,
    store=False,
):
    """
    Sweep one 720-degree engine cycle using forward Euler integration.

    INPUTS:
    - p_start (Pa): plenum pressure at the beginning of the cycle
    - crank_angle_array (rad): crank-angle array over one full 720-degree cycle
    - dtheta (rad): crank-angle step
    - omega (rad/s): crankshaft angular speed
    - speed (RPM): engine speed
    - ve: OEM-derived effective volumetric efficiency at this RPM
    - crankphases_deg: modeled intake-event start angles
    - bore_mm, stroke_mm, lconrod_mm: engine geometry
    - diam_r (m): restrictor diameter
    - T_0, P_0: upstream atmospheric conditions
    - R, T_PLENUM, V_PLENUM: plenum gas constant, temperature, and volume
    - store: if True, return full trace arrays for plotting and post-processing

    OUTPUT:
    - If store=False: dictionary containing p_end and optional trace values set to None.
    - If store=True: dictionary containing pressure, dp/dtheta, mass-flow, and demand traces.
    """
    n_steps = len(crank_angle_array)

    p_array = np.zeros(n_steps, dtype=float)
    p_array[0] = p_start

    # These arrays are only needed for the final converged cycle.
    dpdtheta_array = np.zeros(n_steps, dtype=float) if store else None
    mdot_engine_array = np.zeros(n_steps, dtype=float) if store else None
    mdot_restrictor_array = np.zeros(n_steps, dtype=float) if store else None
    v_demand_array = np.zeros(n_steps, dtype=float) if store else None

    for theta_idx in range(n_steps - 1):
        theta = crank_angle_array[theta_idx]
        p_current = p_array[theta_idx]

        # Prevent non-physical negative or zero pressure from causing numerical
        # issues in density and restrictor-flow calculations.
        if p_current <= 0:
            raise ValueError(f"Non-physical plenum pressure encountered: {p_current:.3f} Pa")

        # Calculate plenum air density using ideal gas law.
        rho_plenum = p_current / (R * T_PLENUM)

        # Calculate engine volume demand rate from crank-angle phasing and piston motion.
        v_demand_rate = engine_volume_demand_rate_at_angle(
            crankphases_deg,
            theta,
            ve,
            bore_mm,
            stroke_mm,
            lconrod_mm,
            speed
        )

        # Convert volume demand to mass demand using instantaneous plenum density.
        m_demand_rate = rho_plenum * v_demand_rate

        # Calculate restrictor mass flow using current plenum pressure.
        m_restrictor = restrictor_mass_flowrate(
            diam_r,
            T_0=T_0,
            p_0=P_0,
            p_plenum=p_current
        )

        # Isothermal plenum pressure ODE:
        # dp/dtheta = (R*T_plenum / (omega*V_plenum)) * (mdot_in - mdot_out)
        dPdtheta = (
            R * T_PLENUM / (omega * V_PLENUM)
        ) * (m_restrictor - m_demand_rate)

        # Store final-cycle trace values before stepping to the next angle.
        if store:
            dpdtheta_array[theta_idx] = dPdtheta
            mdot_engine_array[theta_idx] = m_demand_rate
            mdot_restrictor_array[theta_idx] = m_restrictor
            v_demand_array[theta_idx] = v_demand_rate

        # Forward Euler pressure update.
        p_array[theta_idx + 1] = p_current + (dPdtheta * dtheta)

    # Fill the final trace point using the final pressure. This makes the arrays
    # convenient for plotting against the full crank-angle array.
    if store:
        theta = crank_angle_array[-1]
        p_current = p_array[-1]
        rho_plenum = p_current / (R * T_PLENUM)
        v_demand_rate = engine_volume_demand_rate_at_angle(
            crankphases_deg,
            theta,
            ve,
            bore_mm,
            stroke_mm,
            lconrod_mm,
            speed
        )
        m_demand_rate = rho_plenum * v_demand_rate
        m_restrictor = restrictor_mass_flowrate(
            diam_r,
            T_0=T_0,
            p_0=P_0,
            p_plenum=p_current
        )
        dPdtheta = (
            R * T_PLENUM / (omega * V_PLENUM)
        ) * (m_restrictor - m_demand_rate)

        dpdtheta_array[-1] = dPdtheta
        mdot_engine_array[-1] = m_demand_rate
        mdot_restrictor_array[-1] = m_restrictor
        v_demand_array[-1] = v_demand_rate

    return {
        "p_end": p_array[-1],
        "pressure_pa": p_array if store else None,
        "dpdtheta_pa_per_rad": dpdtheta_array,
        "mdot_engine_kg_s": mdot_engine_array,
        "mdot_restrictor_kg_s": mdot_restrictor_array,
        "v_demand_m3_s": v_demand_array,
    }


# -----------------------------------------
#           ITERATIVE SOLVER
# -----------------------------------------

def iteratively_solve_power_output(engine_name: str, fuel_name: str):
    """
    Iteratively solves for the restricted power output of an engine.

    SOLVER OVERVIEW:
    1. Use the OEM dyno curve and OEM fuel assumptions to infer an effective VE curve.
    2. For each RPM, solve for the periodic plenum pressure trace over a 720-degree cycle.
    3. Integrate the converged engine mass demand over the cycle to get restricted average air flow.
    4. Scale OEM power by restricted/OEM air flow ratio, selected-fuel energy ratio, and pumping loss.
    5. Save diagnostic CSVs and plots under data/output/ENGINE_FUEL/.
    """

    # ------------------------------------
    #       FUNCTION BASE PARAMETERS
    # --------- modify if needed ---------
    # ------------------------------------

    # AMBIENT AIR CONSTANTS:
    T_0 = 300                  # (K) Ambient air temperature.
    P_0 = 101325               # (Pa) Ambient air pressure.
    R = R_AIR                  # (J/kg*K) Gas constant for air.

    # PLENUM CONSTANTS:
    T_PLENUM = 310             # (K) Elevated air temperature of the plenum.
    k = 10                     # Plenum volume / engine displacement.

    # ITERATION CONTROL PARAMETERS:
    MAX_ITERS = 1000           # Maximum cycle iterations per RPM.
    RES_PRESSURE = 1.0e-4      # Non-dimensional pressure residual.
    THETA_STEP_TARGET = 0.01   # (rad) Target crank-angle step size (~0.57 deg). Use 0.001 for high-resolution studies.

    # CRANK ANGLE ARRAY:
    # Use linspace with an exact 4*pi endpoint so cycle integrations line up
    # exactly with one full four-stroke engine cycle.
    n_intervals = int(np.ceil((4 * np.pi) / THETA_STEP_TARGET))
    CRANK_ANGLE_ARRAY_RADIANS = np.linspace(0, 4 * np.pi, n_intervals + 1)
    dtheta = CRANK_ANGLE_ARRAY_RADIANS[1] - CRANK_ANGLE_ARRAY_RADIANS[0]
    THETA_DEG_ARRAY = np.degrees(CRANK_ANGLE_ARRAY_RADIANS)

    # ------------------------------------
    #           FUNCTION OPTIONS:
    # --------- modify if needed ---------
    # ------------------------------------
    FACTOR_FUEL_RATIO = True       # Option to consider fuel energy-per-air change.
    FACTOR_PUMPING_LOSSES = True   # Option to consider extra pumping losses due to low plenum pressure.

    UNDERRELAX = False             # Option to underrelax cycle-start pressure update.
    alpha = 0.3                    # Under-relaxation factor.

    # ------------------------------------
    #        GET ENGINE & FUEL DATA
    # ------------------------------------
    output_dir = make_output_dir(engine_name, fuel_name)

    # Get the engine and selected fuel data from the CSV files.
    data = get_engine_data(engine_name=engine_name, fuel_name=fuel_name)

    # OEM Dyno Curve Data for the engine.
    rpm_arr = data["rpm"]
    torque_nm_arr = data["torque_nm"]
    oem_power_kw_arr = data["power_kw"]
    oem_power_w_arr = oem_power_kw_arr * 1000

    # Engine Specs.
    bore_mm = data["bore_mm"]
    stroke_mm = data["stroke_mm"]
    lconrod_mm = data["lconrod_mm"]
    CYL = data["cyl"]

    # Get the modeled intake event phases for the cylinders.
    crankphases_deg = data["crankphases_deg"]

    # Selected restricted-fuel properties.
    LHV = data["LHV"]       # (J/kg) Lower heating value.
    AFR = data["AFR"]       # Stoichiometric air-fuel ratio.

    # OEM fuel properties for OEM dyno mass-flow and VE extraction.
    OEM_fuel_data = get_OEM_Fuel_data()
    OEM_LHV = OEM_fuel_data["LHV"]
    OEM_AFR = OEM_fuel_data["AFR"]
    OEM_LAMBDA = 0.9
    OEM_fuel_energy = OEM_LHV / (OEM_LAMBDA * OEM_AFR)

    # Selected fuel energy per unit air mass.
    engine_fuel_energy = LHV / (LAMBDA * AFR)
    fuel_ratio = engine_fuel_energy / OEM_fuel_energy if FACTOR_FUEL_RATIO else 1.0

    # Restrictor Size.
    diam_r = calculate_restrictor_diameter_m(fuel_name)

    # Calculate engine displacement and plenum volume.
    disp_m3 = engine_displacement(bore_mm, stroke_mm, CYL)
    V_PLENUM = k * disp_m3

    # Calculate RPM-based effective volumetric efficiency from the OEM dyno curve.
    VE_arr = est_OEM_volumetric_efficiency_arr(
        power_w_arr=oem_power_w_arr,
        rpm_arr=rpm_arr,
        displacement=disp_m3,
        T_ambient=T_0,
        P_ambient=P_0,
        LHV=OEM_LHV,
        lambda_val=OEM_LAMBDA,
        brake_efficiency=BRAKE_EFFICIENCY,
        AFR_stoich=OEM_AFR
    )

    # Create output arrays and diagnostic storage.
    power_restricted_kw = np.zeros_like(oem_power_kw_arr)
    restricted_mdot_kg_s_arr = np.zeros_like(oem_power_kw_arr)
    oem_mdot_kg_s_arr = np.zeros_like(oem_power_kw_arr)
    mass_ratio_arr = np.zeros_like(oem_power_kw_arr)
    mean_plenum_pressure_pa_arr = np.zeros_like(oem_power_kw_arr)
    pumping_loss_kw_arr = np.zeros_like(oem_power_kw_arr)
    mass_balance_error_arr = np.zeros_like(oem_power_kw_arr)
    convergence_iterations_arr = np.zeros_like(oem_power_kw_arr)
    final_pressure_residual_arr = np.zeros_like(oem_power_kw_arr)

    residual_histories = {}
    trace_dict = {}
    residual_records = []
    error_messages = []

    for rpm_idx, speed in enumerate(rpm_arr):
        # Initialize plenum pressure to ambient pressure for the first cycle.
        P_PLENUM = P_0

        # Get the OEM values for VE, power, and OEM air mass flow.
        VE = VE_arr[rpm_idx]
        OEM_power_w = oem_power_w_arr[rpm_idx]
        oem_mrate = est_OEM_air_mass_flowrate(
            OEM_power_w,
            lambda_val=OEM_LAMBDA,
            AFR_stoich=OEM_AFR,
            lower_heating_value=OEM_LHV,
            brake_efficiency=BRAKE_EFFICIENCY
        )
        oem_mdot_kg_s_arr[rpm_idx] = oem_mrate

        # Calculate angular speed in rad/s.
        omega = speed * 2 * np.pi / 60

        if omega <= 0:
            raise ValueError(f"Invalid engine speed encountered: {speed} RPM")

        residuals_this_rpm = []
        converged = False

        # Begin the periodic plenum pressure solver.
        for iter_count in range(MAX_ITERS):
            P_START = P_PLENUM

            cycle_result = sweep_one_cycle(
                p_start=P_START,
                crank_angle_array=CRANK_ANGLE_ARRAY_RADIANS,
                dtheta=dtheta,
                omega=omega,
                speed=speed,
                ve=VE,
                crankphases_deg=crankphases_deg,
                bore_mm=bore_mm,
                stroke_mm=stroke_mm,
                lconrod_mm=lconrod_mm,
                diam_r=diam_r,
                T_0=T_0,
                P_0=P_0,
                R=R,
                T_PLENUM=T_PLENUM,
                V_PLENUM=V_PLENUM,
                store=False,
            )

            P_END = cycle_result["p_end"]
            res = abs(P_START - P_END) / P_START
            residuals_this_rpm.append(res)
            residual_records.append({
                "rpm": speed,
                "iteration": iter_count + 1,
                "pressure_residual": res,
                "p_start_pa": P_START,
                "p_end_pa": P_END,
            })

            if res < RES_PRESSURE:
                converged = True
                break

            # The next cycle starts from the previous cycle's end pressure.
            if not UNDERRELAX:
                P_PLENUM = P_END
            else:
                P_PLENUM = P_START + alpha * (P_END - P_START)

        residual_histories[float(speed)] = residuals_this_rpm
        convergence_iterations_arr[rpm_idx] = len(residuals_this_rpm)
        final_pressure_residual_arr[rpm_idx] = residuals_this_rpm[-1] if residuals_this_rpm else np.nan

        if not converged:
            message = (
                f"ATTENTION: Plenum Pressure Solver did not converge within "
                f"the maximum {MAX_ITERS} iterations.\n"
                f"Engine Speed: {speed}\n"
                f"Residual: {final_pressure_residual_arr[rpm_idx]:.3e}\n"
                f"It is recommended to adjust solution setup parameters."
            )
            error_messages.append(message)
            raise RuntimeError(message)

        print(f"Plenum Pressure Solver Converged for {speed:.0f} RPM after {len(residuals_this_rpm)} cycle iterations.")

        # Rerun one clean converged cycle while storing all trace quantities.
        final_cycle = sweep_one_cycle(
            p_start=P_PLENUM,
            crank_angle_array=CRANK_ANGLE_ARRAY_RADIANS,
            dtheta=dtheta,
            omega=omega,
            speed=speed,
            ve=VE,
            crankphases_deg=crankphases_deg,
            bore_mm=bore_mm,
            stroke_mm=stroke_mm,
            lconrod_mm=lconrod_mm,
            diam_r=diam_r,
            T_0=T_0,
            P_0=P_0,
            R=R,
            T_PLENUM=T_PLENUM,
            V_PLENUM=V_PLENUM,
            store=True,
        )

        # Extract final-cycle arrays.
        pressure_pa = final_cycle["pressure_pa"]
        dpdtheta_pa_per_rad = final_cycle["dpdtheta_pa_per_rad"]
        mdot_engine_kg_s = final_cycle["mdot_engine_kg_s"]
        mdot_restrictor_kg_s = final_cycle["mdot_restrictor_kg_s"]
        v_demand_m3_s = final_cycle["v_demand_m3_s"]

        # Integrate cycle mass using time step dt = dtheta / omega.
        dt = dtheta / omega
        m_engine_cycle = np.sum(mdot_engine_kg_s[:-1]) * dt
        m_restrictor_cycle = np.sum(mdot_restrictor_kg_s[:-1]) * dt

        # Effective average air mass flow rate over one full four-stroke cycle.
        m_eff_avg_rate = m_engine_cycle * speed / 120
        restricted_mdot_kg_s_arr[rpm_idx] = m_eff_avg_rate

        # Calculate restricted/OEM air mass flow ratio.
        mass_ratio = m_eff_avg_rate / oem_mrate
        mass_ratio_arr[rpm_idx] = mass_ratio

        # Restrictor-engine cycle mass balance error.
        if abs(m_engine_cycle) > 0:
            mass_balance_error = abs(m_restrictor_cycle - m_engine_cycle) / abs(m_engine_cycle)
        else:
            mass_balance_error = np.nan
        mass_balance_error_arr[rpm_idx] = mass_balance_error

        # Optional: Factor pumping losses due to plenum pressure depression.
        mean_plenum_pressure = np.sum(pressure_pa[:-1]) * dtheta / (4 * np.pi)
        mean_plenum_pressure_pa_arr[rpm_idx] = mean_plenum_pressure

        if FACTOR_PUMPING_LOSSES:
            pmep = P_0 - mean_plenum_pressure
            pumping_loss_power = pmep * disp_m3 * speed / 120

            if pumping_loss_power < 0:
                raise ValueError(
                    "Calculated negative pumping loss. Mean plenum pressure is above ambient. "
                    "Check pressure model, restrictor model, or any ram-pressure assumptions."
                )
        else:
            pumping_loss_power = 0.0

        pumping_loss_kw_arr[rpm_idx] = pumping_loss_power / 1000

        # Calculate the restricted power output of the engine at the RPM.
        power_restricted_kw[rpm_idx] = (
            (OEM_power_w * mass_ratio * fuel_ratio) - pumping_loss_power
        ) / 1000

        # Store final-cycle traces for diagnostics and plotting.
        trace_dict[float(speed)] = {
            "rpm": float(speed),
            "theta_rad": CRANK_ANGLE_ARRAY_RADIANS.copy(),
            "theta_deg": THETA_DEG_ARRAY.copy(),
            "pressure_pa": pressure_pa.copy(),
            "pressure_kpa": pressure_pa.copy() / 1000,
            "dpdtheta_pa_per_rad": dpdtheta_pa_per_rad.copy(),
            "mdot_engine_kg_s": mdot_engine_kg_s.copy(),
            "mdot_restrictor_kg_s": mdot_restrictor_kg_s.copy(),
            "v_demand_m3_s": v_demand_m3_s.copy(),
        }

    # ------------------------------------
    #        SAVE CSV OUTPUTS
    # ------------------------------------
    summary_df = pd.DataFrame({
        "rpm": rpm_arr,
        "torque_oem_nm": torque_nm_arr,
        "power_oem_kw": oem_power_kw_arr,
        "power_restricted_kw": power_restricted_kw,
        "ve_effective_oem": VE_arr,
        "oem_mdot_kg_s": oem_mdot_kg_s_arr,
        "restricted_mdot_kg_s": restricted_mdot_kg_s_arr,
        "mass_ratio_restricted_to_oem": mass_ratio_arr,
        "fuel_ratio": fuel_ratio,
        "mean_plenum_pressure_pa": mean_plenum_pressure_pa_arr,
        "mean_plenum_pressure_kpa": mean_plenum_pressure_pa_arr / 1000,
        "pumping_loss_kw": pumping_loss_kw_arr,
        "mass_balance_error": mass_balance_error_arr,
        "solver_iterations": convergence_iterations_arr,
        "final_pressure_residual": final_pressure_residual_arr,
    })

    residual_df = pd.DataFrame(residual_records)

    # Long-format trace CSV. This can be large, but it is useful for debugging
    # and post-processing pressure, dp/dtheta, and mass-flow behavior.
    trace_records = []
    for rpm, trace in trace_dict.items():
        n = len(trace["theta_deg"])
        trace_records.append(pd.DataFrame({
            "rpm": np.full(n, rpm),
            "theta_deg": trace["theta_deg"],
            "theta_rad": trace["theta_rad"],
            "pressure_pa": trace["pressure_pa"],
            "pressure_kpa": trace["pressure_kpa"],
            "dpdtheta_pa_per_rad": trace["dpdtheta_pa_per_rad"],
            "mdot_engine_kg_s": trace["mdot_engine_kg_s"],
            "mdot_restrictor_kg_s": trace["mdot_restrictor_kg_s"],
            "v_demand_m3_s": trace["v_demand_m3_s"],
        }))
    traces_df = pd.concat(trace_records, ignore_index=True) if trace_records else pd.DataFrame()

    save_dataframe(summary_df, output_dir, "restricted_power_summary.csv")
    save_dataframe(residual_df, output_dir, "solver_residual_history.csv")
    save_dataframe(traces_df, output_dir, "converged_cycle_traces.csv")

    # Save a lightweight run metadata file.
    metadata_path = output_dir / "run_metadata.txt"
    with metadata_path.open("w", encoding="utf-8") as f:
        f.write(f"Engine: {engine_name}\n")
        f.write(f"Selected restricted fuel: {fuel_name}\n")
        f.write(f"OEM fuel assumption: {OEM_FUEL_NAME}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Restrictor diameter (m): {diam_r}\n")
        f.write(f"Plenum volume ratio k = V_plenum / displacement: {k}\n")
        f.write(f"Engine displacement (m^3): {disp_m3}\n")
        f.write(f"Plenum volume (m^3): {V_PLENUM}\n")
        f.write(f"Ambient pressure (Pa): {P_0}\n")
        f.write(f"Ambient temperature (K): {T_0}\n")
        f.write(f"Plenum temperature (K): {T_PLENUM}\n")
        f.write(f"Brake efficiency assumption: {BRAKE_EFFICIENCY}\n")
        f.write(f"Restricted lambda assumption: {LAMBDA}\n")
        f.write(f"OEM lambda assumption: {OEM_LAMBDA}\n")
        f.write(f"Fuel ratio: {fuel_ratio}\n")
        f.write(f"Theta step target (rad): {THETA_STEP_TARGET}\n")
        f.write(f"Actual theta step (rad): {dtheta}\n")
        f.write(f"Pressure residual tolerance: {RES_PRESSURE}\n")
        f.write(f"Maximum iterations: {MAX_ITERS}\n")

    if error_messages:
        with (output_dir / "error_log.txt").open("w", encoding="utf-8") as f:
            f.write("\n\n".join(error_messages))

    # ------------------------------------
    #        SAVE PLOTS
    # ------------------------------------
    plot_power_comparison(summary_df, output_dir)
    plot_residual_history(residual_histories, output_dir)
    plot_stacked_traces(
        trace_dict,
        output_dir,
        y_key="pressure_kpa",
        y_label="Pressure (kPa)",
        title="Converged Plenum Pressure Over Crank Cycle",
        filename_prefix="stacked_plenum_pressure"
    )
    plot_stacked_traces(
        trace_dict,
        output_dir,
        y_key="dpdtheta_pa_per_rad",
        y_label="dp/dθ (Pa/rad)",
        title="Converged dp/dθ Over Crank Cycle",
        filename_prefix="stacked_dpdtheta"
    )
    plot_mass_flow_detail(trace_dict, rpm_arr, output_dir)
    plot_mass_balance_error(summary_df, output_dir)

    return {
        "rpm": rpm_arr,
        "power_oem_kw": oem_power_kw_arr,
        "power_restricted_kw": power_restricted_kw,
        "summary_df": summary_df,
        "residual_df": residual_df,
        "traces_df": traces_df,
        "output_dir": output_dir,
    }


# \\\\\\\\\\\\\\\\\\\\\n# ------------- MAIN FUNCTION -------------
# /////////////////////////////////////////

def main():
    engine = "Daytona675"
    fuel = "E85"
    results = iteratively_solve_power_output(engine_name=engine, fuel_name=fuel)
    print(f"Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
