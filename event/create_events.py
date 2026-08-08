"""
Create events from linked localizations. One event has the key coordinates
start_ms_fr, end_ms_fr, x, and y.
"""

from typing import Union

import numpy as np
import pandas as pd

from event import link_locs
from utilities import helper


def locs_to_events(
    localizations_file: Union[str, pd.DataFrame],
    offset: float,
    int_time: float,
    max_dark_frames: int = 1,
    proximity: float = 2,
    filter_single: bool = True,
) -> pd.DataFrame:
    """
    Link localizations into events and compute event-level summary metrics.

    Parameters
    ----------
    localizations_file : str or pd.DataFrame
        Path to a localizations file. Must contain the columns 
        {'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy'}.
    offset : float
        Scaling factor applied to the 'frame' column when converting frames to
        time.
    int_time : float
        Integration time per frame (ms).
    max_dark_frames : int, optional
        Number of consecutive frames without a localization that may be skipped
        when linking. Default 1.
    proximity : float, optional
        Maximum distance (in units of lpx + lpy) between adjacent localizations
        for them to belong to the same event. Default 2.
    filter_single : bool, optional
        If True, exclude single localizations that cannot be linked into an
        event. Default True.

    Returns
    -------
    pd.DataFrame
        Events with columns:
        ['event', 'frame', 'x', 'y', 'photons', 'bg', 'lpx', 'lpy', 'sx', 'sy',
         'start_frame', 'end_frame', 'start_ms_fr', 'end_ms_fr', 'num_frames',
         'net_gradient', 'ellipticity', 'group'].
    """
    # --- STEP 1: Load and validate the input DataFrame ---
    localizations = helper.process_input(localizations_file, dataset="locs")
    required_cols = {"frame", "x", "y", "photons", "bg", "lpx", "lpy"}
    missing = required_cols - set(localizations.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # --- STEP 2: Link localizations into events ---
    # link_locs_by_group adds an 'event' column holding the event ID
    localizations_eve = link_locs.link_locs_by_group(
        localizations,
        max_dark_frames=max_dark_frames,
        proximity=proximity,
        filter_single=filter_single,
    )

    # --- STEP 3: Build event-level records ---
    event_records = []
    grouped = localizations_eve.groupby("event")

    for event_id, eve_group in grouped:
        # Reset the index so positional access is clean
        eve_group = eve_group.reset_index(drop=True)

        # First and last localization of the event
        first_loc = eve_group.iloc[0]
        last_loc = eve_group.iloc[-1]

        # Peak (brightest) localization
        peak_idx = eve_group["photons"].idxmax()
        peak_loc = eve_group.loc[peak_idx]

        # Photon-weighted means of position and PSF width
        x_weighted = avg_photon_weighted(eve_group, "x")
        y_weighted = avg_photon_weighted(eve_group, "y")
        sx_weighted = avg_photon_weighted(eve_group, "sx")
        sy_weighted = avg_photon_weighted(eve_group, "sy")

        # Convert oversampled frame index to time: /offset undoes the 10x oversampling,
        # *int_time gives ms; end gets +1 frame so the window covers the last frame fully
        start_ms = (first_loc.frame / offset) * int_time
        end_ms = ((last_loc.frame / offset) + 1) * int_time

        # Accumulate event data in a dict
        event_data = {
            "event": np.uint32(first_loc["event"]),
            "frame": np.uint32(peak_loc["frame"]),
            "x": np.float32(x_weighted),
            "y": np.float32(y_weighted),
            "photons": np.float32(peak_loc["photons"]),
            "bg": np.float32(eve_group["bg"].mean()),
            "lpx": np.float32(peak_loc["lpx"]),
            "lpy": np.float32(peak_loc["lpy"]),
            "sx": np.float32(sx_weighted),
            "sy": np.float32(sy_weighted),
            "group": first_loc.get("group", np.nan),
            "num_frames": np.uint32((last_loc["frame"] - first_loc["frame"]) + 1),
            "start_frame": np.uint32(first_loc["frame"]),
            "end_frame": np.uint32(last_loc["frame"]),
            "net_gradient": np.float32(peak_loc.get("net_gradient", np.nan)),
            "ellipticity": np.float32(peak_loc.get("ellipticity", np.nan)),
            # preliminary frame-based bounds (_fr); refined later by change-point detection
            "start_ms_fr": np.float32(start_ms),
            "end_ms_fr": np.float32(end_ms),
        }
        event_records.append(event_data)

    # --- STEP 4: Build the events DataFrame ---
    events = pd.DataFrame(event_records)

    # --- STEP 5: Report and return ---
    print(f"Linked {len(localizations)} locs to {len(events)} events.")
    return events


def avg_photon_weighted(localizations, column):
    """
    Photon-weighted mean of a column over a set of localizations.

    Parameters
    ----------
    localizations : pd.DataFrame
        Localizations containing a 'photons' column and `column`.
    column : str
        Name of the column to average (e.g. 'x', 'y', 'sx').

    Returns
    -------
    float
        sum(value_i * photons_i) / sum(photons_i).
    """
    photons = localizations["photons"].to_numpy()
    values = localizations[column].to_numpy()
    return np.dot(values, photons) / photons.sum()
