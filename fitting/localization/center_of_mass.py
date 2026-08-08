import numpy as np


def localize_com(x_photons, y_photons, return_sd=True):
    """
    Localize an event by the center of mass of its photon positions.

    Parameters
    ----------
    x_photons, y_photons : np.ndarray
        Photon coordinates (px).
    return_sd : bool, optional
        If True, also return the standard error of the mean (SEM) in x and y.
        If False, the returned SEM values are 0.0. Default True.

    Returns
    -------
    pos_x, pos_y : np.float32
        Center-of-mass position (px).
    sd_x, sd_y : np.float32
        Standard error of the mean in x and y (px), or 0.0 if return_sd is False.
    """
    total_photons = len(x_photons)
    pos_x = np.sum(x_photons) / total_photons
    pos_y = np.sum(y_photons) / total_photons
    if return_sd:
        # standard error of the mean: std of the photon positions / sqrt(N)
        sd_x = np.std(x_photons, ddof=1) / np.sqrt(total_photons)
        sd_y = np.std(y_photons, ddof=1) / np.sqrt(total_photons)
        return (
            pos_x.astype(np.float32),
            pos_y.astype(np.float32),
            sd_x.astype(np.float32),
            sd_y.astype(np.float32),
        )
    else:
        sd_x, sd_y = 0.0, 0.0
        return (pos_x.astype(np.float32), pos_y.astype(np.float32), sd_x, sd_y)
