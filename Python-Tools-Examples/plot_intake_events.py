import numpy as np
import matplotlib.pyplot as plt


def piston_position(theta_deg, stroke, rod_length):
    """
    Slider-crank piston position.

    Args:
        theta_deg: Crank angle in degrees relative to TDC.
        stroke: Engine stroke length.
        rod_length: Connecting rod length.

    Returns:
        Piston position where:
            TDC = stroke
            BDC = 0

        Units match the stroke and rod_length inputs.
    """

    crank_radius = stroke / 2.0
    theta = np.deg2rad(theta_deg)

    # Distance piston has moved down from TDC
    displacement_from_tdc = (
        crank_radius * (1 - np.cos(theta))
        + rod_length
        - np.sqrt(rod_length**2 - (crank_radius * np.sin(theta))**2)
    )

    # Flip so TDC is plot maximum and BDC is plot minimum
    return stroke - displacement_from_tdc


def plot_intake_piston_positions(
    intake_starts,
    stroke,
    rod_length,
    intake_duration=180,
    resolution_deg=1
):
    """
    Plot piston position during intake strokes over a 720-degree 4-stroke cycle.

    Args:
        intake_starts: Array-like. Each element is the intake start angle
                       for that cylinder, in crankshaft degrees.
                       Length of array = number of cylinders.
        stroke: Engine stroke length.
        rod_length: Connecting rod length.
        intake_duration: Intake stroke duration in crank degrees.
                         Default is 180 degrees.
        resolution_deg: Plot resolution in degrees.
    """

    cycle_degrees = 720
    intake_starts = np.asarray(intake_starts, dtype=float)
    num_cylinders = len(intake_starts)

    crank_radius = stroke / 2.0

    if rod_length <= crank_radius:
        raise ValueError("rod_length must be greater than stroke / 2.")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, num_cylinders))

    local_angles = np.arange(
        0,
        intake_duration + resolution_deg,
        resolution_deg
    )

    for i, start_angle in enumerate(intake_starts):
        cyl_num = i + 1
        color = colors[i]

        global_angles = (start_angle + local_angles) % cycle_degrees
        positions = piston_position(local_angles, stroke, rod_length)

        # Split lines if the intake event wraps past 720 degrees
        wrap_indices = np.where(np.diff(global_angles) < 0)[0] + 1
        segment_starts = np.r_[0, wrap_indices]
        segment_ends = np.r_[wrap_indices, len(global_angles)]

        first_segment = True

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            ax.plot(
                global_angles[seg_start:seg_end],
                positions[seg_start:seg_end],
                linewidth=2.5,
                color=color,
                label=f"Cyl {cyl_num}" if first_segment else None
            )
            first_segment = False

    ax.set_xlim(0, cycle_degrees)
    ax.set_ylim(0, stroke)

    ax.set_xlabel("Crankshaft Angle Over 4-Stroke Cycle, degrees")
    ax.set_ylabel("Piston Position")
    ax.set_title("Piston Position During Intake Strokes")

    ax.set_xticks(np.arange(0, cycle_degrees + 1, 90))
    ax.set_yticks([0, stroke])
    ax.set_yticklabels(["BDC", "TDC"])

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Cylinder")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: 4-cylinder engine
    intake_starts = [0, 90, 180, 270, 360, 450, 540, 630]  # degrees, for example

    stroke = 86.0       # mm, for example
    rod_length = 143.0  # mm, for example

    plot_intake_piston_positions(
        intake_starts=intake_starts,
        stroke=stroke,
        rod_length=rod_length
    )