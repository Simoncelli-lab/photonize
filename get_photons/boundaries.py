def min_max_box(localizations, box_side_length=0):
    """
    Bounding box of a set of localizations, expanded by half a box side.

    Parameters
    ----------
    localizations : pd.DataFrame
        Localizations with 'x' and 'y' columns.
    box_side_length : float, optional
        Margin added on each side (the box is expanded by box_side_length / 2
        in every direction). Default 0.

    Returns
    -------
    min_x, max_x, min_y, max_y : float
        Expanded bounding-box limits.
    """
    min_x = min(localizations.x) - (box_side_length / 2)
    max_x = max(localizations.x) + (box_side_length / 2)
    min_y = min(localizations.y) - (box_side_length / 2)
    max_y = max(localizations.y) + (box_side_length / 2)
    return min_x, max_x, min_y, max_y


def spatial_boundaries(event, diameter):
    """
    Square boundaries of side `diameter` centered on an event.

    Parameters
    ----------
    event : pd.Series
        Event with 'x' and 'y' attributes (px).
    diameter : float
        Side length of the square ROI (px).

    Returns
    -------
    x_min, x_max, y_min, y_max : float
        ROI boundaries.
    """
    x_min = event.x - (diameter / 2)
    x_max = x_min + diameter
    y_min = event.y - (diameter / 2)
    y_max = y_min + diameter
    return x_min, x_max, y_min, y_max
