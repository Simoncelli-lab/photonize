import numba
import numpy as np
import pandas as pd


@numba.njit
def apply_drift_correction(
    x, y, frames, drift_x, drift_y, num_photons, max_frame_drift
):
    """
    Subtract drift and align photons to Picasso coordinates (Numba-compiled).

    Alignment formula (16x binning):
        P_c = L_c - (0.5 - 1 / (2 * binning)) = L_c - 0.46875
        P_c = Picasso coordinates
        L_c = LINCam coordinates

    When 16 detector pixels (values 0-15) are binned into one TIF pixel, the
    average value of the 0th pixel is 7.5 / 16 = 0.46875, so this offset aligns
    photon coordinates with the binned localization grid.

    Parameters
    ----------
    x, y : np.ndarray
        Photon coordinates (px).
    frames : np.ndarray
        Frame index of each photon.
    drift_x, drift_y : np.ndarray
        Per-frame drift in x and y (px).
    num_photons : int
        Number of photons.
    max_frame_drift : int
        Number of frames available in the drift arrays; frame indices at or
        beyond this are clamped to the last frame.

    Returns
    -------
    undrifted_x, undrifted_y : np.ndarray
        Drift-corrected, aligned photon coordinates (px).
    """
    undrifted_x = np.empty(num_photons)
    undrifted_y = np.empty(num_photons)
    for i in range(num_photons):
        frame = frames[i]
        if frame >= max_frame_drift:
            frame = max_frame_drift - 1  # Prevent out-of-bounds access
        # Apply drift and fixed correction offset of 0.46875
        undrifted_x[i] = x[i] - (0.46875 + drift_x[frame])
        undrifted_y[i] = y[i] - (0.46875 + drift_y[frame])
    return undrifted_x, undrifted_y


def undrift_photons(photons, drift, offset, int_time=200):
    """
    Subtract drift and align photons to Picasso coordinates.

    Wraps the Numba-compiled `apply_drift_correction`.

    Parameters
    ----------
    photons : pd.DataFrame
        Photons with 'x', 'y', 'dt', 'ms' columns.
    drift : pd.DataFrame
        Per-frame drift with 'x' and 'y' columns.
    offset : float
        Frame offset used to map photon times to frame indices.
    int_time : float, optional
        Camera integration time per frame (ms). Default 200.

    Returns
    -------
    pd.DataFrame
        Drift-corrected photons with 'x', 'y', 'dt', 'ms' columns.
    """
    # Convert DataFrame columns to NumPy arrays
    ms_index = photons["ms"].to_numpy()
    x_photons = photons["x"].to_numpy()
    y_photons = photons["y"].to_numpy()
    # Map each photon's ms to its (oversampled) frame index
    frames = np.floor((offset * ms_index) / int_time).astype(np.int32)
    # Drift arrays
    drift_x = drift["x"].to_numpy()
    drift_y = drift["y"].to_numpy()
    # Get dimensions
    num_photons = len(photons)
    max_frame_drift = len(drift_x)
    # Apply drift correction using the Numba function
    undrifted_x, undrifted_y = apply_drift_correction(
        x_photons, y_photons, frames, drift_x, drift_y, num_photons, max_frame_drift
    )
    # Create a new DataFrame with undrifted coordinates
    photons_undrifted = pd.DataFrame(
        {
            "x": undrifted_x,
            "y": undrifted_y,
            "dt": photons["dt"].to_numpy(),
            "ms": ms_index,
        }
    )
    return photons_undrifted
