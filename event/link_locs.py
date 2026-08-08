import numpy as np


def are_nearby(x1, y1, x2, y2, threshold):
    """
    Test whether two points are within a Euclidean distance threshold.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Coordinates of the two points.
    threshold : float
        Maximum allowed distance.

    Returns
    -------
    bool
        True if the distance between the points is <= threshold, else False.
    """
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) <= (threshold * threshold)


def link_group(group_frames, group_x, group_y, max_distance, max_dark_frames):
    """
    Link localizations into events within a single group.

    Assumes the input arrays are sorted by frame in ascending order; the
    early-exit (break) logic relies on this ordering. Sorting is done by the
    caller (link_locs_by_group).

    Parameters
    ----------
    group_frames : np.ndarray
        1D array of localization frames (int), sorted ascending.
    group_x : np.ndarray
        1D array of localization x-coordinates (float).
    group_y : np.ndarray
        1D array of localization y-coordinates (float).
    max_distance : np.ndarray
        1D array of per-localization distance thresholds (typically
        proximity * (lpx + lpy)).
    max_dark_frames : int
        Maximum number of frames that may be skipped while still treating
        localizations as part of the same event.

    Returns
    -------
    np.ndarray
        1D array of event IDs, one per localization. Localizations linked into
        the same event share the same ID.
    """

    n = len(group_frames)  
    event_ids = np.zeros(n, dtype=np.int32)
    current_event_id = 0

    # Loop over all group localizations
    for i in range(n):

        if event_ids[i] == 0:  # If not yet assigned to an event
            current_event_id += 1
            event_ids[i] = current_event_id

        # Compare with localizations in the next frames within max_dark_frames
        for j in range(i + 1, n):
            # Allowed gap: the next frame (+1) plus up to max_dark_frames skipped
            if group_frames[j] > group_frames[i] + 1 + max_dark_frames:
                break  # Stop if frames are beyond the allowed gap

            if (
                group_frames[i]
                < group_frames[j]
                <= group_frames[i] + 1 + max_dark_frames
            ):
                if are_nearby(
                    group_x[i], group_y[i], group_x[j], group_y[j], max_distance[i]
                ):
                    event_ids[j] = event_ids[i]  # Assign the same event ID
                    break  # Link to the first nearby successor only, then stop

    return event_ids


def link_locs_by_group(
    localizations_dset, filter_single=True, proximity=2, max_dark_frames=1
):
    """
    Link localizations into events group by group.

    Connects localizations in adjacent or nearby frames (up to max_dark_frames),
    iterating over each group separately.

    Parameters
    ----------
    localizations_dset : pd.DataFrame
        Localizations with columns: frame, group, x, y, lpx, lpy, etc.
    filter_single : bool, optional
        If True, remove single localizations that cannot be linked into a
        multi-localization event. Default True.
    proximity : float, optional
        Multiplier for the per-localization distance threshold. Default 2.
    max_dark_frames : int, optional
        Maximum number of frames that may be skipped when linking. Default 1.

    Returns
    -------
    pd.DataFrame
        A copy of the localizations with two added columns:
          - 'event': integer label of the linked event
          - 'count': number of localizations in that event
    """
    localizations = localizations_dset.copy()
    localizations.insert(1, "event", 0)  

    # Group by 'group' and iterate
    grouped = localizations.groupby("group", sort=False)

    # Global event counter
    global_event_id = 0
    for group_id, group_df in grouped:

        frames = group_df["frame"].to_numpy()
        x_coords = group_df["x"].to_numpy()
        y_coords = group_df["y"].to_numpy()
        lpx_p_lpy = (group_df["lpx"] + group_df["lpy"]).to_numpy()

        # Sort by frame to ensure temporal order
        sort_indices = np.argsort(frames)
        frames = frames[sort_indices]
        x_coords = x_coords[sort_indices]
        y_coords = y_coords[sort_indices]
        lpx_p_lpy = lpx_p_lpy[sort_indices]

        group_event_ids = link_group(
            frames, x_coords, y_coords, proximity * lpx_p_lpy, max_dark_frames
        )

        # Map local group event IDs to global event IDs
        unique_local_event_ids = np.unique(group_event_ids)
        local_to_global_map = {
            local_id: global_event_id + idx + 1
            for idx, local_id in enumerate(unique_local_event_ids)
        }
        global_event_id += len(unique_local_event_ids)

        # Assign the global event IDs back to the original DataFrame
        localizations.loc[group_df.index[sort_indices], "event"] = [
            local_to_global_map[e] for e in group_event_ids
        ]

    # Count the number of localizations per event
    localizations.insert(
        2, "count", localizations["event"].map(localizations["event"].value_counts())
    )

    # Optionally filter out single-localization events
    if filter_single:
        locs_before = len(localizations)
        localizations = localizations[localizations["count"] > 1].reset_index(drop=True)
        locs_after = len(localizations)
        print(f"removed {locs_before-locs_after} unlinkable single localizations.")

    return localizations
