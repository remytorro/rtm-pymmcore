# %% [markdown]
# # Move up — region-percentage stim, Inscoper microscope
#
# The Inscoper variant of `region_percentage_stim.ipynb`. Exercises the **async**
# faro pipeline end to end with the region-percentage stimulator (`StimUp`),
# which illuminates the top *percentage* of every segmented cell — the patterned,
# per-cell job a galvo is actually good at.
#
# * **Imaging:** `470 WF` (nuclear / reporter) + `550 WF` (second reporter)
# * **Stim:** the FRAP galvo through `all lasers`
# * **Ref == a separate channel:** `640 WF` on the last frame
# * **Async run:** `run_experiment` returns a `RunHandle`; the
#   `ExperimentStatusWidget` mirrors and steers it live
#
# Site configuration lives in `../inscoper_site.py`.
#
# **Two differences from the original that change what you must do:**
#
# 1. **No `mic.calibrate_dmd(...)`.** The original calibrates a DMD on a
#    background thread. Here the stim device is a FRAP galvo whose calibration
#    the *firmware* applies per point, so a DMD affine calibrated in this
#    notebook would transform the mask a second time and aim the beam wrong.
#    `InscoperMicroscope.uses_dmd_affine` is `False` and the mask passes through
#    in camera pixels.
# 2. **The stim mask has a price.** `StimUp` scales with cell count and with
#    `STIM_CELL_PERCENTAGE`, and one FRAP fire covers ~36,000 px of scan path
#    (~5 s). Pricing the real mask before the run is the only check that knows
#    whether *this* sample's segmentation produces something firable.

# %% [markdown]
# ## Microscope + napari

# %%
import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

from faro.core.data_structures import (
    Channel,
    PowerChannel,
    RTMSequence,
    SegmentationMethod,
    combine,
    wait,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.controller import Controller
import faro.core.utils as utils

from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.erk_ktr import FE_ErkKtr
from faro.stimulation.percentage_of_cell import StimUp

from faro.widgets import ExperimentStatusWidget

# %%
# 512 px: big enough for a useful number of cells, small enough that a
# per-cell stim mask stays inside the scan budget. See site.FULL_FOV_NOTE for
# what a *whole-field* burn would cost at each frame size — this notebook never
# asks for one, because StimUp is patterned.
mic = site.make_microscope(roi=512)

# %%
viewer, mm_wdg = site.open_napari(mic, title="25 region-percentage stim (Inscoper)")

# %% [markdown]
# ## Channels
#
# The original's `mScarlet3` / `miRFP` / `mCitrine` / `CyanStim` do not exist in
# this configuration. Plain `Channel` is used rather than `PowerChannel`: an
# Inscoper `.cbc` already carries its own wavelength and power, so there is
# nothing left to push. (`site.POWER_PROPERTIES` maps every channel to its
# device and property if you do want `PowerChannel`.)

# %%
# Stimulation — the channel that opens the FRAP light path. Its exposure is
# bookkeeping: the burn's duration is the scan, and its dose the repetitions.
stim_channel = Channel(config=site.FRAP_CHANNEL, exposure=200, group=site.CHANNEL_GROUP)

# Imaging — order matters, channel 0 is segmented (and channel 1 too, below).
imaging_channels = (
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),
)

# Reference / optocheck readout.
ref_channel = Channel(config="640 WF", exposure=60, group=site.CHANNEL_GROUP)

# %% [markdown]
# ## No DMD calibration
#
# The cell the original puts here is `mic.calibrate_dmd(...)`. Deliberately
# omitted — see difference 1 at the top. The galvo's calibration is made once by
# `inscoper_useq/scripts/script_frap_calibration.py` and stored per camera and
# filter cube; `site.designate_frap` (run during the load) checks that a
# calibration matching the current device state exists, which is the check that
# actually matters here.

# %%
assert getattr(mic.mmc, "use_frap_as_slm", False), (
    "expected the FRAP galvo to be the stim device on this configuration"
)
assert not mic.uses_dmd_affine, (
    "uses_dmd_affine is True — masks would be warped by the DMD affine as well "
    "as by the firmware's FRAP calibration, i.e. transformed twice"
)
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"
print("FRAP is the stim device; no DMD calibration wanted or needed.")

# %% [markdown]
# ## Experiment config: baseline -> stim -> recovery -> ref

# %%
## --- timing --------------------------------------------------------------
INTERVAL_S = 30          # seconds between timepoints
TIME_PER_FOV_S = 5.0     # seconds to image one FOV

## --- phases --------------------------------------------------------------
N_BASELINE = 2           # imaging frames before stim
N_STIM = 8               # stim frames (StimUp fires on each)
N_RECOVERY = 2           # imaging frames after stim
N_REF = 1                # final reference frame
N_FRAMES = N_BASELINE + N_STIM + N_RECOVERY + N_REF

STIM_FRAMES = list(range(N_BASELINE, N_BASELINE + N_STIM))
REF_FRAMES = [-1]

# Fraction of each cell to stimulate. Read from event metadata by StimUp;
# it is also the main lever on how expensive the burn is.
STIM_CELL_PERCENTAGE = 0.2

## --- storage -------------------------------------------------------------
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "25_region_percentage_stim_inscoper"
path = os.path.join(base_path, experiment_name)

site.check_fov_timing(INTERVAL_S, TIME_PER_FOV_S)
print(f"{N_FRAMES} frames at {INTERVAL_S}s = {(N_FRAMES - 1) * INTERVAL_S / 60:.1f} min")
print(f"stim on frames {STIM_FRAMES[0]}..{STIM_FRAMES[-1]}, ref on {REF_FRAMES}")
print(f"storage: {path}")

# %% [markdown]
# ## Pipeline: segmentation + tracking + region-percentage stim

# %%
segmentators = [
    SegmentationMethod(
        name="labels",
        # The original uses CellposeV4(gamma=0.3); site.make_segmentator falls
        # back to a threshold segmentator when cellpose is absent and says so.
        # multichannel=True matters: use_channel is a *list*, so the
        # segmentator is handed a (C, Y, X) stack. Cellpose reduces that
        # itself; the fallback does not, and unreduced (C, Y, X) labels fail
        # inside regionprops_table on the first analysed frame -- i.e. after
        # the run has started.
        segmentation_class=site.make_segmentator(gamma=0.3, multichannel=True),
        use_channel=[0, 1],           # segment on both imaging channels
        save_tracked=True,
    )
]
# StimUp reads its fraction from the CONSTRUCTOR, not from event
# metadata -- unlike StimPercentageOfCell, which reads
# metadata['stim_cell_percentage']. The original notebook sets
# stim_cell_percentage in rtm_metadata and instantiates StimUp() with no
# argument, so the metadata is carried into exp_data.parquet while the
# stimulator quietly uses its 0.2 default. Pass it explicitly so the
# recorded value and the fired value are the same number.
stimulator = StimUp(fraction=STIM_CELL_PERCENTAGE)
feature_extractor = FE_ErkKtr("labels")        # per-cell features
tracker = TrackerTrackpy(search_range=30)

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=feature_extractor,
    tracker=tracker,
    stimulator=stimulator,
)

writer = site.make_writer(path)
ctrl = Controller(mic, pipeline, writer=writer)

# %% [markdown]
# ## Build the acquisition
#
# Pick the FOV positions in the napari MDA widget first, then run this cell.

# %%
USE_GUI_FOVS = True
if USE_GUI_FOVS:
    fov_positions = utils.generate_fov_positions(mic, viewer=viewer)
else:
    fov_positions = site.fovs_here(mic, 1)
print(f"{len(fov_positions)} FOV(s)")
site.check_fov_timing(INTERVAL_S, TIME_PER_FOV_S, len(fov_positions))

phase = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": N_FRAMES},
    stage_positions=fov_positions,
    channels=imaging_channels,
    stim_channels=(stim_channel,),
    stim_frames=STIM_FRAMES,
    ref_channels=(ref_channel,),
    ref_frames=REF_FRAMES,
    rtm_metadata={
        "phase_name": "region_percentage_stim",
        "phase_id": 0,          # the pipeline names the tracks parquet from this
        "stim_cell_percentage": STIM_CELL_PERCENTAGE,
        "treatment_name": "move_up_stim",
        "stim_fov": None,
    },
)

events = utils.apply_fov_batching(
    phase, time_per_fov=TIME_PER_FOV_S, offset_min_start_time=True
)
print(f"{len(events)} events")
utils.events_to_dataframe(events).head()

# %% [markdown]
# ## Validate, and price the mask this pipeline will actually fire
#
# Two checks that answer different questions:
#
# * `validate_events` — do the channels exist, are the exposures legal, is
#   `stim_cell_percentage` present on the events (`StimUp` declares it as
#   `required_metadata`)?
# * the pricing below — can the galvo fire what this sample's segmentation
#   produces? Nothing static can answer that, because the mask does not exist
#   until a frame has been segmented.

# %%
assert ctrl.validate_events(events), "validation reported problems — see warnings"

# StimUp returns an EMPTY mask when tracks is None, so a probe that does not
# supply one prices a free burn for a mask that is in fact expensive.
# site.probe_stim_mask synthesises the tracks frame the pipeline would provide.
labels, mask, plan = site.probe_stim_mask(
    mic,
    segmentators[0].segmentation_class,
    stimulator,
    channel=imaging_channels[0].config,
    exposure=imaging_channels[0].exposure,
    metadata={"stim_cell_percentage": STIM_CELL_PERCENTAGE, "stim_fov": True},
)

# %% [markdown]
# ## Dock the status widget
#
# Mirrors and steers the current run (state, event strip, FOV map, lag,
# pause/stop). It re-binds automatically on each `run_experiment`.

# %%
status_wdg = ExperimentStatusWidget(ctrl)
viewer.window.add_dock_widget(status_wdg, name="Experiment status")

# %% [markdown]
# ## Run (async)
#
# Returns a `RunHandle` immediately and drives the MDA on a worker thread.
#
# `stim_mode="previous"` fires frame *t-1*'s mask *before* imaging frame *t* —
# lower latency, one timestep stale, and suppressed at t=0 where there is no
# predecessor. `"current"` images first and fires that frame's own mask, which
# is the right choice when a ~2 s galvo scan inside a 30 s interval is
# affordable. On this rig it is, so `"current"` is the default here.

# %%
handle = ctrl.run_experiment(events, stim_mode="current")
print("running — watch the status dock, or handle.status()")

# %%
handle.status()

# %%
# handle.pause() / handle.resume() / handle.cancel() also work from here.

# %% [markdown]
# ## Finish + post-process

# %%
handle.wait()
ctrl.finish_experiment()      # flush the pipeline, close the writer
utils.generate_exp_data_from_tracks(path)

df_exp = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"{len(df_exp)} cell rows, {df_exp['particle'].nunique()} particles")
df_exp.head()

# %% [markdown]
# ## Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
