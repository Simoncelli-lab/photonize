import numpy as np

import fitting
from event import analyze_events, create_events
from utilities import helper


def evelyze(
    localizations_file,
    photons_file,
    drift_file,
    offset,
    int_time,
    pos_diameter=4.5,
    lt_diameter=5.5,
    ref_pixel_nm=115,
    frame_size=256,
    suffix="",
    max_dark_frames=1,
    proximity=2,
    filter_single=True,
    norm_brightness=False,
    dt_window=None,
    more_ms=0,
    verbose=False,
    **kwargs,
):
    """
    Run the full event pipeline: link localizations, analyze events, save output.

    Reads the localization, photon, and drift inputs; links localizations into
    events; fits each event's position (center of mass) and lifetime
    (Gaussian-mask weighting); optionally normalizes brightness; and saves the
    result in picasso format with an appended provenance message.

    Parameters
    ----------
    localizations_file : str or pd.DataFrame
        Picked + filtered localizations (picasso .hdf5 file or DataFrame).
    photons_file : str or pd.DataFrame
        Photon index file (picasso .hdf5 file or DataFrame).
    drift_file : str or pd.DataFrame
        Drift file (picasso .txt file or DataFrame).
    offset : float
        Temporal oversampling factor of the localization file (frames per
        integration time); used to convert frame indices to time and to map
        photon times to drift frames.
    int_time : float
        Camera integration time per frame (ms).
    pos_diameter : float, optional
        ROI diameter (px) used for position fitting. Default 4.5.
    lt_diameter : float, optional
        ROI diameter (px) used for lifetime estimation. Default 5.5.
    ref_pixel_nm : float, optional
        Binned pixel size (nm); should match frame_size (115 nm for 256 px).
        Recorded in the output metadata for provenance. Default 115.
    frame_size : int, optional
        Detector frame size in pixels, used for the brightness-normalization
        grid. Default 256.
    suffix : str, optional
        Suffix appended to the output filename. Default ''.
    max_dark_frames : int, optional
        Number of consecutive frames without a localization that may be skipped
        when linking. Default 1.
    proximity : float, optional
        Maximum distance (in units of lpx + lpy) between adjacent localizations
        for them to belong to the same event. Default 2.
    filter_single : bool, optional
        If True, exclude single localizations that cannot be linked. Default True.
    norm_brightness : bool, optional
        If True, normalize brightness using a local brightness map. Events
        with non-finite fitted coordinates are dropped first. Default False.
    dt_window : tuple of float, optional
        Optional (min, max) gate on photon arrival time 'dt'. Default None.
    more_ms : int, optional
        Extra milliseconds cropped before and after each event. Default 0.
    verbose : bool, optional
        If True, print detailed per-group diagnostics. Default False.
    **kwargs
        Forwarded to events_lt_pos.

    Returns
    -------
    None
        The events DataFrame is written to disk in picasso format.
    """
    if verbose:
        print("Starting event analysis: ...")
    # 1) read in files
    localizations = helper.process_input(localizations_file, dataset="locs")
    photons = helper.process_input(photons_file, dataset="photons")
    drift = helper.process_input(drift_file, dataset="drift")
    # 2) create preliminary events by linking localizations
    events = create_events.locs_to_events(
        localizations,
        offset=offset,
        int_time=int_time,
        max_dark_frames=max_dark_frames,
        proximity=proximity,
        filter_single=filter_single,
    )
    # 3) analyze events in main loop (position + lifetime + brightness)
    arrival_time = {}
    events = analyze_events.events_lt_pos(
        events,
        photons,
        drift,
        offset,
        int_time=int_time,
        pos_diameter=pos_diameter,
        lt_diameter=lt_diameter,
        arrival_time=arrival_time,
        dt_window=dt_window,
        more_ms=more_ms,
        verbose=verbose,
        **kwargs,
    )
    # 4) normalize brightness if requested
    if norm_brightness:
        if verbose:
            print("Normalizing brightness...")
        # Degenerate events (too few photons in their ROI) can have non-finite
        # fitted coordinates, which would break the brightness-map KDTree.
        finite_mask = np.isfinite(events["x"]) & np.isfinite(events["y"])
        n_dropped = int((~finite_mask).sum())
        if n_dropped:
            print(f"dropped {n_dropped} events with non-finite coordinates")
            events = events[finite_mask].reset_index(drop=True)
        events = fitting.normalize_brightness(events, frame_size=frame_size)
    # 5) save events
    file_extension = "_event" + suffix
    message = helper.create_append_message(
        function="Evelyze",
        localizations_file=localizations_file,
        photons_file=photons_file,
        drift_file=drift_file,
        offset=offset,
        int_time=int_time,
        pos_method="com",
        pos_diameter=pos_diameter,
        lt_method="gauss_mask",
        lt_diameter=lt_diameter,
        ref_pixel_nm=ref_pixel_nm,
        frame_size=frame_size,
        link_proximity=proximity,
        max_dark_frames=max_dark_frames,
        filter_single=filter_single,
        start_stop_event="ruptures",
        background="picasso",
        peak_arrival_time=arrival_time["start"],
    )
    helper.dataframe_to_picasso(events, localizations_file, file_extension, message)
