import numpy as np

from fitting.illumination.local_map import local_brightness_map


def normalize_brightness(events, frame_size=256):
    """
    Normalize event brightness using a local brightness map (inverse distance
    weighting).

    Parameters
    ----------
    events : pd.DataFrame
        Events with 'x', 'y', and 'brightness_phot_ms' columns (and optionally
        'lifetime_10ps').
    frame_size : int, optional
        Detector frame size in pixels, passed to the brightness map. Default 256.

    Returns
    -------
    pd.DataFrame
        The events with an added 'brightness_norm' column and, if
        'lifetime_10ps' is present, an added 'lt_over_bright' column.
    """
    bg_map, grid_x, grid_y = local_brightness_map(
        events, radius=5, p=1, grid_size=1, frame_size=frame_size
    )
    px_x = np.round(events["x"]).astype(int)
    px_y = np.round(events["y"]).astype(int)
    # Clip indices so events outside the computed map stay in range.
    max_x, max_y = bg_map.shape
    px_x = np.clip(px_x, 0, max_x - 1)
    px_y = np.clip(px_y, 0, max_y - 1)

    # bg_map is indexed [y, x], not [x, y]
    norm_values = bg_map[px_y, px_x]
    # Replace zeros with 1 to avoid division by zero
    norm_values_safe = np.where(norm_values == 0, 1, norm_values)
    # Normalize brightness
    if hasattr(events, "brightness_phot_ms"):
        brightness_arr = events.brightness_phot_ms
        brightness_idx = events.columns.get_loc("brightness_phot_ms")
        events.insert(
            brightness_idx,
            "brightness_norm",
            (events["brightness_phot_ms"] / norm_values_safe).astype(np.float32),
        )
        events.drop(columns="brightness_phot_ms", inplace=True)
        events["brightness_phot_ms"] = brightness_arr.astype(np.float32)
        if hasattr(events, "lifetime_10ps"):
            events["lt_over_bright"] = (
                events["lifetime_10ps"] / events["brightness_norm"]
            ).astype(np.float32)
    return events
