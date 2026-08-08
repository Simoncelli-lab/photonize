from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fitting
import get_photons
from fitting.fit_event import fit_event
from utilities import helper


def events_lt_pos(
    event_file,
    photons_file,
    drift_file,
    offset,
    int_time=200,
    pos_diameter=4.5,
    lt_diameter=5.5,
    arrival_time: Optional[Dict[str, Any]] = None,
    dt_window: Optional[Tuple[float, float]] = None,
    more_ms: int = 0,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Tag linked events with lifetime and fitted ROI position.

    For each event, photons are cropped to a circular ROI, the position is fit
    by center of mass and the lifetime estimated by Gaussian-mask weighting, and
    the results are written back into the events DataFrame. 

    Parameters
    ----------
    event_file : str or pd.DataFrame
        Events as a picasso .hdf5 file (must contain a 'group' column).
    photons_file : str or pd.DataFrame
        Photons as a picasso .hdf5 file.
    drift_file : str or pd.DataFrame
        Drift as a picasso .txt file.
    offset : int
        Frame offset used to convert photon times to frame indices.
    int_time : int, optional
        Camera integration time per frame (ms). Default 200.
    pos_diameter : float, optional
        ROI diameter (px) used for position fitting. Default 4.5.
    lt_diameter : float, optional
        ROI diameter (px) used for lifetime estimation. Default 5.5.
    arrival_time : dict, optional
        Dictionary used to return the calibrated peak arrival time under the
        key 'start'. Initialized to an empty dict if None.
    dt_window : tuple of float, optional
        Optional (min, max) gate on photon arrival time 'dt'. Default None.
    more_ms : int, optional
        Extra milliseconds cropped before and after each event. Default 0.
    verbose : bool, optional
        If True, print per-group diagnostics and low-photon warnings.
        Default False (only a progress bar and summary lines are shown).
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    pd.DataFrame
        The events DataFrame tagged with fitted position, lifetime, brightness,
        localization precision, and timing columns.
    """
    # Ensure mutable default is not reused
    if arrival_time is None:
        arrival_time = {}

    # Read input files
    events = helper.process_input(event_file, dataset="locs")
    photons = helper.process_input(photons_file, dataset="photons")
    drift = helper.process_input(drift_file, dataset="drift")
    total_events = len(events)
    print(f"{len(photons)} photons and {total_events} events read in")

    # Preallocate arrays using np.empty (assumes all values will be assigned)
    lifetime = np.empty(total_events, dtype=np.float32)
    total_photons_arr = np.empty(total_events, dtype=np.float32)
    x_position = np.empty(total_events, dtype=np.float32)
    y_position = np.empty(total_events, dtype=np.float32)
    sdx = np.empty(total_events, dtype=np.float32)
    sdy = np.empty(total_events, dtype=np.float32)
    lft_sem = np.empty(total_events, dtype=np.float32)
    duration_ms_arr = np.empty(total_events, dtype=np.float32)
    start_ms_new = np.empty(total_events, dtype=np.float32)
    end_ms_new = np.empty(total_events, dtype=np.float32)
    delta_x = np.empty(total_events, dtype=np.float32)
    delta_y = np.empty(total_events, dtype=np.float32)

    # Define crop diameter for returning photons:
    # crop a slightly larger square (+0.5 px margin) than the largest ROI so edge
    # photons survive the later circular masking in crop_event
    crop_diameter = max(pos_diameter, lt_diameter) + 0.5

    # Use only the first 1M photons to keep the peak-arrival histogram fast;
    # the peak channel is stable, so a subset suffices.
    calib_photons = photons[:1000000]
    if dt_window:
        calib_photons = calib_photons[
            (calib_photons.dt >= dt_window[0]) & (calib_photons.dt <= dt_window[1])
        ]
    # Calibrate peak arrival time from a subset of photons
    peak_arrival_time = fitting.calibrate_peak_arrival(calib_photons)
    arrival_time["start"] = peak_arrival_time
    print(f"Peak arrival time: {peak_arrival_time}")
    if verbose and dt_window:
        print(f"Considering photons with dt in {dt_window}")

    # Area of the circular position-fitting ROI
    fit_area = (pos_diameter / 2) ** 2 * np.pi

    # Initialize event index
    idx = 0

    # Iterate over events by group, with a progress bar over all events
    progress = tqdm(total=total_events, desc="Analyzing events", unit="event")
    try:
        for group_value, events_group in events.groupby("group"):
            if verbose:
                print(f"__________Analyzing group {int(group_value + 1)}__________")
                print(f"{len(events_group)} events in current group.")
            pick_photons = get_photons.get_pick_photons(
                events_group,
                photons,
                drift,
                offset,
                diameter=(crop_diameter),
                int_time=int_time,
            )
            if verbose:
                print(f"Number of picked photons: {len(pick_photons)}")

            # Apply dt window filter if provided
            if dt_window:
                # inclusive bounds, matching the calibration-subset filter above
                pick_photons = pick_photons[
                    (pick_photons.dt >= dt_window[0])
                    & (pick_photons.dt <= dt_window[1])
                ]

            # Iterate over each event in the current group
            for i, my_event in events_group.iterrows():

                # Crop the relevant photons for this event
                cylinder_photons = get_photons.crop_event(
                    my_event,
                    pick_photons,
                    crop_diameter,
                    more_ms=more_ms,
                    verbose=verbose,
                )

                # Analyze the event using a helper function
                result = fit_event(
                    cylinder_photons,
                    x_start=my_event.x,
                    y_start=my_event.y,
                    sigma=(my_event.sx + my_event.sy) / 2,
                    dt_peak=peak_arrival_time,
                    pos_diameter=pos_diameter,
                    lt_diameter=lt_diameter,
                )

                x_position[idx] = result.x_fit
                y_position[idx] = result.y_fit
                sdx[idx] = result.sdx_fit
                sdy[idx] = result.sdy_fit
                lifetime[idx] = result.lifetime
                lft_sem[idx] = result.lft_sem
                total_photons_arr[idx] = result.num_photons
                start_ms_new[idx] = result.start_ms
                end_ms_new[idx] = result.end_ms
                duration_ms_arr[idx] = result.duration_ms
                delta_x[idx] = my_event.x - result.x_fit
                delta_y[idx] = my_event.y - result.y_fit
                idx += 1
                progress.update(1)
    finally:
        progress.close()

    # Update events DataFrame with computed values
    bg_picasso = events["bg"].to_numpy()

    # Subtract background photons: bg rate x (event duration / frame time) x ROI area
    photons_arr = total_photons_arr - (
        bg_picasso * duration_ms_arr / int_time * fit_area
    )

    events["x"] = x_position
    events["y"] = y_position
    events["photons"] = photons_arr.astype(np.float32)
    events["bg"] = bg_picasso
    events["lpx"] = sdx  
    events["lpy"] = sdy
    events["lifetime_10ps"] = lifetime.astype(np.float32)
    events["lt_sem"] = lft_sem
    events["duration_ms"] = duration_ms_arr.astype(np.float32)
    events["brightness_phot_ms"] = (photons_arr / duration_ms_arr).astype(np.float32)
    events["photons_COM"] = total_photons_arr.astype(np.float32)
    events["start_ms"] = start_ms_new.astype(np.int32)
    events["end_ms"] = end_ms_new.astype(np.int32)
    events["delta_x"] = delta_x.astype(np.float32)
    events["delta_y"] = delta_y.astype(np.float32)
    events.drop(columns=["start_ms_fr", "end_ms_fr"], inplace=True)

    # Save to picasso file if event_file is provided as a string
    if isinstance(event_file, str):
        helper.dataframe_to_picasso(events, event_file, "eve_lt_avgPos")

    print(f"_______________FINISHED: {len(events)} events analysed!_______________")
    return events
