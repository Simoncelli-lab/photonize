# photonize

Event-level analysis of DNA-PAINT data recorded with a wide-field 
time-correlated single photon counting camera (LINCam, Photonscore). 
The pipeline links single-molecule localizations into binding events, 
retrieves the raw photons of each event, and estimates for every event:

- its **position**, as the center of mass of the photon coordinates, with the
  standard error of the mean reported as the localization precision
  (`lpx`, `lpy`);
- its **fluorescence lifetime**, as the Gaussian-mask-weighted mean photon
  arrival time relative to a calibrated pulse peak, with a per-event
  uncertainty (`lt_sem`);
- its **timing** (start, end, duration) from change-point detection on the
  photon time trace, and its background-corrected **brightness** (optional).


## Requirements

Python ≥ 3.10. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Input files

`evelyze` (the pipeline entry point) takes three inputs:

1. **Localization file** (`.hdf5`, Picasso format): picked and filtered
   localizations. Must contain a `group` column (from picking in Picasso or an
   upstream clustering step) and the standard Picasso columns
   (`frame`, `x`, `y`, `photons`, `bg`, `lpx`, `lpy`, `sx`, `sy`, ...).
   The matching Picasso `.yaml` sidecar must sit next to this file — the
   output `.yaml` is derived from it.
2. **Photon index file** (`.hdf5`): the ordered per-photon stream with columns
   `x`, `y`, `ms`, `dt`, produced by the acquisition preprocessing.
   Coordinates are in the binned-pixel frame of the localization images;
   `dt` is the photon arrival time in TCSPC channels.
3. **Drift file** (`.txt`): per-frame drift, two space-separated columns
   (x, y), as exported by Picasso.

The upstream preprocessing (raw `.photons` → index file + localization TIFF
stack) is a separate MATLAB step and is not part of this repository.

## Usage

See `analysis.ipynb` for a ready-to-edit example. Run it from the repository
root (the packages are imported from the working directory):

```python
import event

event.evelyze(
    locs_file,             # picked localizations (.hdf5, with .yaml sidecar)
    index_file,            # photon index (.hdf5)
    drift_file,            # drift (.txt)
    offset=10,             # temporal oversampling factor of the localization file
    int_time=200,          # camera integration time per frame (ms)
    pos_diameter=4.5,      # ROI diameter (px) for position fitting
    lt_diameter=5.5,       # ROI diameter (px) for lifetime estimation
    frame_size=256,        # detector frame size (px)
    ref_pixel_nm=115,      # binned pixel size (nm)
    max_dark_frames=1,     # max skipped frames when linking locs into an event
    proximity=2,           # link threshold = proximity x (lpx + lpy)
    filter_single=True,    # drop single-localization events
    norm_brightness=True,  # normalize brightness across the field of view
    dt_window=None,        # optional (min, max) arrival-time gate
    more_ms=0,             # extra ms cropped before/after each event
)
```

The result is saved next to the localization file with the suffix `_event`
(e.g. `..._picked_set_event.hdf5` plus the matching `.yaml`), in Picasso
format, with the full parameter set appended to the `.yaml` for provenance.

Note: all ROI diameters and coordinates are in **binned pixels** (115 nm per
pixel for a 256-px frame in our configuration); lifetimes are in **TCSPC
channels** (10 ps per channel in our data), referenced to the calibrated
pulse-peak channel determined once per dataset.

## Output columns (main)

| column | meaning |
| --- | --- |
| `x`, `y` | event position: center of mass of the event's photons (px) |
| `lpx`, `lpy` | localization precision: standard error of the photon center of mass (px) |
| `lifetime_10ps` | event lifetime: weighted mean arrival channel minus the calibrated peak |
| `lt_sem` | standard error of the lifetime (weighted SEM over the effective photon number) |
| `photons` | background-corrected photon count of the event |
| `photons_COM` | raw photon count used for the position fit |
| `start_ms`, `end_ms`, `duration_ms` | event timing from change-point detection |
| `brightness_phot_ms`, `brightness_norm` | brightness and field-flattened brightness |

If brightness normalization is enabled, events whose position fit produced
non-finite coordinates (degenerate events with too few photons in the ROI)
are removed before the normalization; the number of dropped events is
printed.


## Citation

If you use this code, please cite: [].

## License

See `LICENSE`.
