import numpy as np


def circular_mean(angles):
    """
    Calculates the circular (directional) mean of an array of angles.

    Parameters:
    angles (array-like): The input angles in degrees (between 0 and 360).
    in_degrees (bool): If True, treats inputs and outputs as degrees.
                       If False, treats them as radians.

    Returns:
    float: The circular mean angle.
    """
    angles = np.asarray(angles)
    angles = np.radians(angles)

    # Convert degrees to radians for trigonometric functions

    # 1. Convert angles to Cartesian coordinates (x, y) on a unit circle
    # 2. Sum or average the coordinates (summing works the same for arctan2)
    x = np.sum(np.cos(angles))
    y = np.sum(np.sin(angles))

    # 3. Convert the resulting vector back to an angle
    mean_angle = np.arctan2(y, x)

    # Convert back to degrees if requested and wrap to [0, 360)
    mean_angle = np.degrees(mean_angle)
    return mean_angle % 360
