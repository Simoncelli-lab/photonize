import numpy as np


def fit_lifetime(dt_array, dt_peak, distance_array, lt_diameter, sigma=None):
    """
    Estimate lifetime from photon arrival times using a Gaussian-mask weighting.

    Parameters
    ----------
    dt_array : np.ndarray
        Photon arrival times (TCSPC channel number).
    dt_peak : float
        Peak channel to subtract (calibrated reference).
    distance_array : np.ndarray
        Distance of each photon from the localization center.
    lt_diameter : float
        Diameter of the circular ROI (same units as distances).
    sigma : float
        PSF sigma used for the Gaussian weighting. Required.

    Returns
    -------
    lifetime : float
        Estimated lifetime, or np.nan if no photons fall within the ROI.
    lft_sem : float
        Standard error of the lifetime (0.0 if no photons fall within the ROI).
    """
    radius = lt_diameter / 2
    mask = distance_array <= radius
    if not np.any(mask):
        # no photons in the ROI: lifetime undefined
        return np.nan, 0
    dt_mask = dt_array[mask]
    dist_mask = distance_array[mask]
    if sigma is None:
        raise ValueError("sigma must be provided for 'gauss_mask' method")
    # Gaussian weight by radial distance: down-weights background photons far from center
    weights = np.exp(-(dist_mask**2) / (2 * sigma**2))
    weights /= np.sum(weights)
    lifetime = np.sum(weights * dt_mask) - dt_peak
    t = dt_mask - dt_peak
    # effective number of photons and standard error of the weighted-mean lifetime
    n_signal = np.sum(weights) ** 2 / np.sum(weights**2)
    sigma_lft = np.sqrt(np.sum(weights * (t - lifetime) ** 2))
    lft_sem = sigma_lft / np.sqrt(n_signal)
    return lifetime, lft_sem
