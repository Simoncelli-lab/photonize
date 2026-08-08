from get_photons.boundaries import min_max_box
from get_photons.crop_photons import crop_rectangle
from get_photons.undrift import undrift_photons


def get_pick_photons(locs_group, photons, drift, offset, diameter, int_time):
    """
    Collect drift-corrected photons in the area of a single pick (group).

    Crops photons to the pick's bounding box (expanded by the drift range),
    applies drift correction, then crops again to the final box.

    Parameters
    ----------
    locs_group : pd.DataFrame
        Localizations of this pick (group).
    photons : pd.DataFrame
        Photons with 'x', 'y', 'ms' columns.
    drift : pd.DataFrame
        Drift with 'x' and 'y' columns.
    offset : float
        Frame offset used by the drift correction.
    diameter : float
        Crop diameter in pixels (the bounding box is expanded by diameter + 1).
    int_time : float
        Camera integration time per frame (ms).

    Returns
    -------
    pd.DataFrame
        Drift-corrected photons within the pick area (+/- diameter / 2).
    """
    # expand the crop by the max drift so no in-pick photons are missed before undrifting;
    # -0.46875 aligns to the binned pixel grid (see undrift)
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = min_max_box(locs_group, diameter + 1)
    # crop photons of area of interest plus drift
    phot_cr = crop_rectangle(
        photons,
        (min_x - 0.46875 - dr_x),
        (max_x - 0.46875 + dr_x),
        (min_y - 0.46875 - dr_y),
        (max_y - 0.46875 + dr_y),
    )
    # undrift and align photons
    phot_cr_und = undrift_photons(phot_cr, drift, offset, int_time)
    # crop photons again after drift
    phot_cr_und_cr = crop_rectangle(phot_cr_und, min_x, max_x, min_y, max_y)
    return phot_cr_und_cr
