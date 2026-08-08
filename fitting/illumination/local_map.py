import numpy as np
from scipy.spatial import cKDTree


def local_brightness_map(events, radius=10, p=1, grid_size=1, frame_size=256):
    """
    Compute a brightness map from sparse localizations via inverse distance
    weighting (IDW) over a fixed neighborhood radius.

    For each grid pixel (spacing set by grid_size), all events within `radius`
    contribute a weighted average of their brightness, with weight 1 / dist**p.

    Parameters
    ----------
    events : pd.DataFrame
        Localizations with 'x', 'y', and 'brightness_phot_ms' columns.
    radius : float, optional
        Neighborhood radius in pixels. Default 10.
    p : float, optional
        Power parameter for inverse distance weighting. Default 1.
    grid_size : float, optional
        Spacing between grid points (px). Default 1.
    frame_size : int, optional
        Detector frame size in pixels; the grid spans 0 to frame_size in both
        x and y. Default 256.

    Returns
    -------
    brightness_map : np.ndarray
        2D brightness map over the grid.
    grid_x : np.ndarray
        X coordinates of the grid.
    grid_y : np.ndarray
        Y coordinates of the grid.
    """
    # Convert localization data to numpy arrays
    x = np.asarray(events.x, dtype=np.float64)
    y = np.asarray(events.y, dtype=np.float64)
    brightness_array = np.asarray(events.brightness_phot_ms, dtype=np.float64)

    # Grid extents span the detector frame (0 to frame_size in both axes).
    min_x, max_x = 0, frame_size
    min_y, max_y = 0, frame_size

    # Create grid coordinates (grid points at integer coordinates)
    grid_x = np.arange(min_x, max_x + 1, grid_size)
    grid_y = np.arange(min_y, max_y + 1, grid_size)
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")

    # Build a KDTree for fast spatial queries on the localization points
    points = np.column_stack((x, y))
    tree = cKDTree(points)

    # Initialize the brightness map
    brightness_map = np.zeros_like(X, dtype=np.float64)

    # Evaluate every grid pixel from all points within the radius
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            grid_point = np.array([X[i, j], Y[i, j]])
            # Indices of all localizations within the radius
            indices = tree.query_ball_point(grid_point, r=radius)
            if indices:
                # Euclidean distances from the grid pixel to each point
                distances = np.sqrt(
                    (x[indices] - grid_point[0]) ** 2
                    + (y[indices] - grid_point[1]) ** 2
                )
                # If any localization sits exactly on the grid point, use its value directly
                if np.any(distances == 0):
                    brightness_map[i, j] = np.mean(
                        brightness_array[indices][distances == 0]
                    )
                else:
                    # IDW weights: 1 / dist**p
                    weights = 1 / (distances**p)
                    weighted_brightness = np.sum(brightness_array[indices] * weights)
                    sum_weights = np.sum(weights)
                    brightness_map[i, j] = weighted_brightness / sum_weights
            else:
                # No localizations nearby: assign a small floor (avoids zero-division in normalize)
                brightness_map[i, j] = 0.01

    return brightness_map, grid_x, grid_y
