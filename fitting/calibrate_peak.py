import numpy as np


def calibrate_peak_arrival(event_photons):
    """
    Estimate the peak photon arrival time as the mode of the arrival-time
    histogram.

    Parameters
    ----------
    event_photons : pd.DataFrame
        Photon data with a 'dt' column of arrival times (TCSPC channel number).
        Use at most 1,000,000 photons to keep the histogram fast.

    Returns
    -------
    int
        Index of the histogram bin with the highest count, i.e. the most
        frequent arrival channel.

    Notes
    -----
    The histogram spans the observed dt range (min to max) with unit-width bins.
    Useful for aligning or calibrating arrival-time distributions in FLIM data.
    """
    min_dt = np.min(event_photons.dt)
    max_dt = np.max(event_photons.dt)
    counts, bins = np.histogram(event_photons.dt, bins=np.arange(min_dt, max_dt))
    return np.argmax(counts)
