# This is the main function for analysing a single event
from collections import namedtuple

import numpy as np

from fitting.fit_lifetime import fit_lifetime
from fitting.fit_position import fit_position
from fitting.on_off import get_on_off_dur

# Define the named tuple once
EventResult = namedtuple(
    "EventResult",
    [
        "x_fit",
        "y_fit",
        "sdx_fit",
        "sdy_fit",
        "lifetime",
        "lft_sem",
        "start_ms",
        "end_ms",
        "duration_ms",
        "num_photons",
    ],
)


def fit_event(
    photons,
    x_start,
    y_start,
    sigma,
    dt_peak,
    pos_diameter=4.5,
    lt_diameter=7,
):
    """
    Analyze a single photon event: fit position (center of mass) and lifetime
    (Gaussian-mask weighting).

    Parameters
    ----------
    photons : pd.DataFrame
        Photons cropped to the event ROI, with columns 'x', 'y', 'ms', 'dt'.
    x_start, y_start : float
        Initial position estimate (px).
    sigma : float
        PSF width (px) used for the lifetime fit.
    dt_peak : float
        Calibrated peak arrival time; photons before it are excluded.
    pos_diameter : float, optional
        ROI diameter (px) used for position fitting. Default 4.5.
    lt_diameter : float, optional
        ROI diameter (px) used for lifetime estimation. Default 7.

    Returns
    -------
    EventResult
        Named tuple with fields:
        - x_fit, y_fit : fitted position (px)
        - sdx_fit, sdy_fit : positional standard error (px)
        - lifetime : estimated lifetime
        - lft_sem : standard error of the lifetime
        - start_ms, end_ms : event start/end time (ms)
        - duration_ms : event duration (ms)
        - num_photons : number of photons in the event
    """
    # get start and end of event
    start_ms, end_ms, duration_ms = get_on_off_dur(photons)
    # keep only photons inside the event time window and after the pulse peak
    event_photons = photons[
        (photons.ms >= start_ms) & (photons.ms <= end_ms) & (photons.dt >= dt_peak)
    ]
    # extract coordinates and arrival times
    x_photons = np.copy(event_photons.x)
    y_photons = np.copy(event_photons.y)
    dt_photons = np.copy(event_photons.dt)
    x_fit, y_fit, sdx_fit, sdy_fit = fit_position(
        x_start=x_start,
        y_start=y_start,
        x_photons=x_photons,
        y_photons=y_photons,
        pos_diameter=pos_diameter,
    )
    # radial distance of each photon from the fitted position (for the lifetime Gaussian mask)
    distances = np.hypot((x_photons - x_fit), (y_photons - y_fit))
    lifetime, lft_sem = fit_lifetime(
        dt_array=dt_photons,
        dt_peak=dt_peak,
        distance_array=distances,
        lt_diameter=lt_diameter,
        sigma=sigma,
    )
    # pack into a named tuple and return
    return EventResult(
        x_fit=x_fit,
        y_fit=y_fit,
        sdx_fit=sdx_fit,
        sdy_fit=sdy_fit,
        lifetime=lifetime,
        lft_sem=lft_sem,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        num_photons=len(event_photons),
    )
