import numpy as np
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters


def get_on_off_dur(photons, bin_size_ms=10, smoothing_size=5, jump=1):
    """
    Estimate an event's start, end, and duration from photon arrival times.

    Bins the arrival times into a histogram, smooths it with a Lee filter, and
    runs change-point detection to find the on/off transitions.

    Parameters
    ----------
    photons : object with attribute 'ms'
        Photon data where `photons.ms` is an array of arrival times (ms).
    bin_size_ms : int or float, optional
        Bin width for the time histogram (ms). Default 10.
    smoothing_size : int, optional
        Window size for Lee-filter smoothing (must be odd). Default 5.
    jump : int, optional
        Subsampling step for the change-point search. Default 1.

    Returns
    -------
    start : float
        Estimated event start time (ms).
    end : float
        Estimated event end time (ms).
    duration : float
        Estimated event duration (ms).

    Notes
    -----
    Uses the 'Binseg' algorithm from `ruptures` with the L2 model to find two
    change points (on and off). Falls back to the min/max of the photon times
    if segmentation fails.
    """
    bins = np.arange(min(photons.ms), max(photons.ms) + bin_size_ms, bin_size_ms)
    counts, _ = np.histogram(photons.ms, bins=bins)
    smoothed_counts = lee_filter_1d(counts, smoothing_size)
    model = "l2"
    algo = rpt.Binseg(model=model, min_size=1, jump=jump).fit(smoothed_counts)
    try:
        change_points = np.array(algo.predict(n_bkps=2))
        # convert change-point bin indices back to time, with half-bin offsets to
        # bracket the on/off transitions slightly wide
        start = (change_points[0] - 1.5) * bin_size_ms + bins[0]
        end = (change_points[1] + 0.5) * bin_size_ms + bins[0]
        duration = (change_points[1] - change_points[0]) * bin_size_ms
    except BadSegmentationParameters:
        # segmentation failed (too few/flat counts): fall back to full photon time range
        start = min(photons.ms)
        end = max(photons.ms)
        duration = end - start
    return start, end, duration


def lee_filter_1d(data, window_size=5):
    """
    Smooth 1D data with a Lee filter for noise reduction.

    Parameters
    ----------
    data : np.ndarray
        1D array to filter.
    window_size : int, optional
        Sliding-window size (must be odd). Default 5.

    Returns
    -------
    np.ndarray
        Smoothed data.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    padded_data = np.pad(data, pad_width=window_size // 2, mode="reflect")
    local_mean = np.convolve(
        padded_data, np.ones(window_size) / window_size, mode="valid"
    )
    local_var = (
        np.convolve(padded_data**2, np.ones(window_size) / window_size, mode="valid")
        - local_mean**2
    )
    noise_var = np.mean(local_var)
    result = local_mean + (local_var / (local_var + noise_var)) * (data - local_mean)
    return result
