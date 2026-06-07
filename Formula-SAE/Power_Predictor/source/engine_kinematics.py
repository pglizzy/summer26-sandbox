"""
slidercrank.py

Slider-crank piston kinematics.

Inputs:
    stroke_mm        [mm]
    rod_length_mm    [mm]
    rpm              [rev/min]
    crank_angle_deg  [deg]

Outputs:
    position_m       [m]
    velocity_m_s     [m/s]
    acceleration_m_s2 [m/s^2]

Coordinate convention:
    TDC = 0
    BDC = stroke

Positive direction is from TDC toward BDC.
"""

import numpy as np


MM_TO_M = 1e-3


def _validate_geometry(stroke_mm: float, rod_length_mm: float) -> tuple[float, float]:
    """
    Validate slider-crank geometry and return crank radius and rod length in meters.
    """

    if stroke_mm <= 0:
        raise ValueError("stroke_mm must be greater than zero.")

    if rod_length_mm <= 0:
        raise ValueError("rod_length_mm must be greater than zero.")

    crank_radius_mm = stroke_mm / 2.0

    if rod_length_mm <= crank_radius_mm:
        raise ValueError(
            "rod_length_mm must be greater than crank radius. "
            "Check stroke and rod length inputs."
        )

    crank_radius_m = crank_radius_mm * MM_TO_M
    rod_length_m = rod_length_mm * MM_TO_M

    return crank_radius_m, rod_length_m


def piston_position_m(
    stroke_mm: float,
    rod_length_mm: float,
    crank_angle_deg,
):
    """
    Calculate piston position from TDC.

    Returns
    -------
    float or ndarray
        Piston position in meters.
    """

    r, l = _validate_geometry(stroke_mm, rod_length_mm)

    theta = np.deg2rad(crank_angle_deg)

    position_m = (
        r * (1.0 - np.cos(theta))
        + l
        - np.sqrt(l**2 - (r * np.sin(theta))**2)
    )

    return position_m


def piston_velocity_m_s(
    stroke_mm: float,
    rod_length_mm: float,
    rpm: float,
    crank_angle_deg,
):
    """
    Calculate piston velocity.

    Positive velocity means piston is moving from TDC toward BDC.

    Returns
    -------
    float or ndarray
        Piston velocity in m/s.
    """

    r, l = _validate_geometry(stroke_mm, rod_length_mm)

    theta = np.deg2rad(crank_angle_deg)
    omega = rpm * 2.0 * np.pi / 60.0  # rad/s

    root = np.sqrt(l**2 - (r * np.sin(theta))**2)

    dx_dtheta = (
        r * np.sin(theta)
        + (r**2 * np.sin(theta) * np.cos(theta)) / root
    )

    velocity_m_s = dx_dtheta * omega

    return velocity_m_s


def piston_acceleration_m_s2(
    stroke_mm: float,
    rod_length_mm: float,
    rpm: float,
    crank_angle_deg,
):
    """
    Calculate piston acceleration.

    Positive acceleration is toward BDC.

    Returns
    -------
    float or ndarray
        Piston acceleration in m/s^2.
    """

    r, l = _validate_geometry(stroke_mm, rod_length_mm)

    theta = np.deg2rad(crank_angle_deg)
    omega = rpm * 2.0 * np.pi / 60.0  # rad/s

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    root = np.sqrt(l**2 - (r * sin_theta)**2)

    d2x_dtheta2 = (
        r * cos_theta
        + (r**2 * (cos_theta**2 - sin_theta**2)) / root
        + (r**4 * sin_theta**2 * cos_theta**2) / root**3
    )

    acceleration_m_s2 = d2x_dtheta2 * omega**2

    return acceleration_m_s2


def piston_kinematics(
    stroke_mm: float,
    rod_length_mm: float,
    rpm: float,
    crank_angle_deg,
):
    """
    Calculate piston position, velocity, and acceleration together.

    Returns
    -------
    dict
        Dictionary containing MKS outputs.
    """

    return {
        "position_m": piston_position_m(
            stroke_mm,
            rod_length_mm,
            crank_angle_deg,
        ),
        "velocity_m_s": piston_velocity_m_s(
            stroke_mm,
            rod_length_mm,
            rpm,
            crank_angle_deg,
        ),
        "acceleration_m_s2": piston_acceleration_m_s2(
            stroke_mm,
            rod_length_mm,
            rpm,
            crank_angle_deg,
        ),
    }