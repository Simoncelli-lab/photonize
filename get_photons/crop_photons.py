import numpy as np
import pandas as pd

from get_photons import boundaries


def crop_rectangle(photons, x_min=0, x_max=float("inf"), y_min=0, y_max=float("inf")):
    """
    Filter photons to a rectangular x-y region.

    Parameters
    ----------
    photons : pd.DataFrame
        Photons with 'x', 'y', 'ms' columns.
    x_min, x_max : float, optional
        Bounds on x. Defaults span the full range.
    y_min, y_max : float, optional
        Bounds on y. Defaults span the full range.

    Returns
    -------
    pd.DataFrame
        Photons within the specified x-y range.
    """
    mask = (
        (photons.x >= x_min)
        & (photons.x <= x_max)
        & (photons.y >= y_min)
        & (photons.y <= y_max)
    )
    return photons[mask]


def crop_cuboid(
    photons,
    x_min=0,
    x_max=float("inf"),
    y_min=0,
    y_max=float("inf"),
    ms_min=0,
    ms_max=float("inf"),
):
    """
    Filter photons to a rectangular x-y region within a time window.

    Parameters
    ----------
    photons : pd.DataFrame
        Photons with 'x', 'y', 'ms' columns.
    x_min, x_max : float, optional
        Bounds on x. Defaults span the full range.
    y_min, y_max : float, optional
        Bounds on y. Defaults span the full range.
    ms_min, ms_max : float, optional
        Bounds on time (ms). Defaults span the full range.

    Returns
    -------
    pd.DataFrame
        Photons within the specified x-y-time bounds.
    """
    mask = (
        (photons.x >= x_min)
        & (photons.x <= x_max)
        & (photons.y >= y_min)
        & (photons.y <= y_max)
        & (photons.ms >= ms_min)
        & (photons.ms <= ms_max)
    )
    return photons[mask]


def crop_event(event, photons, diameter, more_ms=0, verbose=False):
    """
    Crop photons to a cylinder (circular ROI x time window) around an event.

    Parameters
    ----------
    event : pd.Series
        Event with 'x', 'y', and either ('start_ms', 'end_ms') or
        ('start_ms_fr', 'end_ms_fr') attributes.
    photons : pd.DataFrame
        Photons with 'x', 'y', 'ms' columns.
    diameter : float
        Diameter of the circular ROI around the event position (px).
    more_ms : float, optional
        Extra time included before and after the event (ms). Default 0.
    verbose : bool, optional
        If True, print a warning when the cylinder contains very few photons.
        Default False.

    Returns
    -------
    pd.DataFrame
        Photons within the cylinder, with an added 'distance' column holding the
        squared distance to the event center.

    Raises
    ------
    AttributeError
        If the event lacks both ('start_ms', 'end_ms') and
        ('start_ms_fr', 'end_ms_fr').
    """
    x_min, x_max, y_min, y_max = boundaries.spatial_boundaries(event, diameter)
    if hasattr(event, "start_ms") and hasattr(event, "end_ms"):
        start, end = event.start_ms, event.end_ms
    elif hasattr(event, "start_ms_fr") and hasattr(event, "end_ms_fr"):
        start, end = event.start_ms_fr, event.end_ms_fr
    else:
        raise AttributeError(
            "Required attributes are missing. Expected either 'start_ms', 'end_ms' or 'start_ms_fr', 'end_ms_fr'."
        )
    photons_cropped = pd.DataFrame(
        data=crop_cuboid(
            photons, x_min, x_max, y_min, y_max, (start - more_ms), (end + more_ms)
        )
    )
    x_distance = photons_cropped["x"].to_numpy() - event.x
    y_distance = photons_cropped["y"].to_numpy() - event.y
    total_distance_sq = np.square(x_distance) + np.square(y_distance)
    photons_cropped["distance"] = total_distance_sq
    # circular ROI: keep photons within radius = diameter/2 (compare squared)
    radius_sq = (diameter / 2) ** 2
    photons_cylinder = photons_cropped[photons_cropped.distance <= radius_sq]
    if verbose and len(photons_cylinder) < 30:
        try:
            print(
                f"low photon count for crop_event(): "
                f"len={len(photons_cylinder)}, event={event.event}"
            )
        except AttributeError:
            pass
    return photons_cylinder
