from fitting.localization import localize_com


def fit_position(x_start, y_start, x_photons, y_photons, pos_diameter):
    """
    Fit an event position from photon coordinates (center of mass).

    Parameters
    ----------
    x_start, y_start : float
        Initial position estimate (px).
    x_photons, y_photons : np.ndarray
        Photon coordinates (px).
    pos_diameter : float
        Diameter of the circular fitting ROI (px).

    Returns
    -------
    x, y : float
        Fitted position (px).
    sdx, sdy : float
        Positional standard error in x and y (px).
    """
    dx = x_photons - x_start
    dy = y_photons - y_start
    distance_squared = dx**2 + dy**2
    # Keep photons within the circular ROI (radius = pos_diameter / 2) around the start position
    radius_squared = (pos_diameter / 2) ** 2
    mask = distance_squared <= radius_squared
    x_filtered = x_photons[mask]
    y_filtered = y_photons[mask]
    x, y, sdx, sdy = localize_com(x_filtered, y_filtered, return_sd=True)
    return x, y, sdx, sdy
