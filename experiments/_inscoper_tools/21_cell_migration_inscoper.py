# %% [markdown]
# # Cell Migration RTM Pipeline — Inscoper microscope
#
# The Inscoper variant of `cell_migration.ipynb`. Directed migration by
# stimulating the leading part of each cell, as in the real-time feedback
# microscopy paper — but the stimulation device here is a **FRAP galvo
# scanner**, not the DMD the original drives.
#
# Site configuration (paths, camera, channels, focus, FRAP channel) lives in
# `../inscoper_site.py`.

# %% [markdown]
# ## Why a FRAP galvo changes this notebook
#
# Four things, and every one of them fails **silently** if it is wrong — no
# light, no exception, and a 24 h run that looks successful until analysis.
#
# 1. **The stim channel cannot be picked by name.** Every stock channel called
#    `... WF Quad FRAP` is an *imaging* channel looking through the quad FRAP
#    dichroic, and ships `iLas2::Shutter = 0` — which scans the galvo in the
#    dark. The stimulation channel is `all lasers`, and
#    `site.designate_frap` validates it against the configuration's own
#    `Ilda.frap.ActiveModeChannel`.
#
# 2. **Masks stay in camera pixels — do not calibrate the DMD.** The FRAP
#    calibration maps camera pixels to galvo DAC units and the *firmware*
#    applies it per point. `InscoperMicroscope.prepare_stim_mask` therefore
#    passes the mask through untouched and `uses_dmd_affine` is `False`.
#    Running `mic.calibrate_dmd(...)` here would transform the mask a second
#    time and aim the beam wrong. It is currently harmless only by accident —
#    an *uncalibrated* DMD degrades to an identity resize — so calibrating it
#    is precisely what breaks FRAP.
#
# 3. **A burn is bounded by scan path, not by mask area.** One fire covers
#    ~36,000 px of scan path (~5 s at ~7200 px/s) and this rig hatches at a
#    4 px row pitch. `plan_mask` **refuses** an over-budget region rather than
#    truncating it, so an oversized mask is an exception at the first stim
#    event. The pricing cell below turns that into a number before the run.
#
# 4. **`StimTreatment` does not control the FRAP dose.** `stim_exposure_list`
#    and `stim_power` describe a *lamp* exposure. The FRAP burn takes its laser
#    line and power from the designated channel and its dose from the number of
#    repetitions, so `stim_exposure_list=200` does **not** make the burn 200 ms
#    long. Keep `stim_channel_name` pointing at the FRAP channel so the two
#    agree, and read the exposure fields as bookkeeping.

# %% [markdown]
# ## Imports and microscope

# %%
import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

from faro.core.data_structures import Channel, SegmentationMethod, StimTreatment
import faro.core.utils as utils

# %%
# roi=512 is a deliberate choice, not tidying:
#  * a stim mask over a 2304 px field is refused at this rig's 4 px pitch
#    (site.FULL_FOV_NOTE has the measured table), and
#  * 512x512 readout is what makes a 15 s interval over several FOVs keep up.
# frap_channel defaults to site.FRAP_CHANNEL ("all lasers") and is designated
# and preflighted during the load.
mic = site.make_microscope(roi=512)

# %% [markdown]
# ## FRAP preflight
#
# `site.designate_frap` already ran inside `make_microscope` and printed its
# verdict. This cell re-runs it as an assertion — the whole point is that an
# unopened FRAP path is invisible at run time — and then prices a mask the size
# of the region you actually intend to stimulate.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

# What a plausible single-region stim costs, in scan time. Keep one pass well
# under ~5 s. (The original notebook wrote this as a `for ... else`, which
# always prints "clear" — a for/else runs its else clause unless the loop
# breaks — so it reported success even when the preflight had complaints.)
plan = site.price_mask(mic, site.sample_disc(mic, radius_fraction=0.125))
assert plan is not None, "the sample region is already over budget — shrink it"

# And confirm the pipeline's own stimulator cannot produce something refused.
# StimPercentageOfCell illuminates a fraction of every segmented cell, so the
# worst case is "many cells, large fraction". Price that, not the best case.
worst_case = np.zeros((mic.image_height, mic.image_width), bool)
yy, xx = np.ogrid[: mic.image_height, : mic.image_width]
rng = np.random.default_rng(0)
for _ in range(25):
    cy, cx = rng.integers(40, min(mic.image_height, mic.image_width) - 40, 2)
    worst_case |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= 22**2
print("\nworst case (25 cells, r=22 px):")
site.price_mask(mic, worst_case)

# %% [markdown]
# ## Experiment configuration
#
# Same knobs as the original. The channel names are the difference: this
# configuration has no `mScarlet3` / `mCitrine` / `CyanStim`, so the reporter,
# the optocheck and the stimulation channel are re-mapped onto
# `site.IMAGING_CHANNELS` and `site.FRAP_CHANNEL`.

# %%
## --- timing -------------------------------------------------------------
N_FRAMES = 30 * 4                    # number of timesteps
TIME_BETWEEN_TIMESTEPS = 15          # seconds between timesteps
TIME_PER_FOV = 3.75                  # seconds to image one FOV
SLEEP_BEFORE_EXPERIMENT_START_in_H = 0

ADD_STIM_EXPOSURE_GROUP = False
REGULAR_SPACING_BETWEEN_STIMULATIONS = False

# generate_df_acquire divides by `time_between_timesteps // time_per_fov`,
# so an interval shorter than the per-FOV cost raises a bare
# ZeroDivisionError from inside faro. Fail here with a sentence instead.
site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV)

## --- storage ------------------------------------------------------------
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "21_cell_migration_inscoper"
path = os.path.join(base_path, experiment_name)

## --- channels -----------------------------------------------------------
# Order matters: channel 0 is what the segmentator sees.
channels = [Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP)]

# Optocheck: the optogenetic-marker readout. On this rig it can be the same
# light path as imaging; a longer exposure is the usual reason to separate it.
channel_optocheck = Channel(config="550 WF", exposure=50, group=site.CHANNEL_GROUP)
optocheck_timepoints = (N_FRAMES - 1,)

## --- conditions ---------------------------------------------------------
condition = ["optoTIAM_single"]
n_fovs_per_well = None

## --- stimulation --------------------------------------------------------
# stim_channel_name MUST be the designated FRAP channel: the burn's laser line
# and power come from that channel's .cbc, and pointing this elsewhere means
# the bookkeeping in exp_data.parquet describes a different light path from the
# one that actually fired.
stim_phase = [
    StimTreatment(
        treatment_name="15min_stim",
        stim_timestep=tuple(range(1, N_FRAMES, 1)),
        stim_exposure_list=200,      # bookkeeping — see note 4 above
        stim_power=100,              # ditto; the .cbc sets the real laser power
        stim_channel_name=site.FRAP_CHANNEL,
        stim_channel_group=site.CHANNEL_GROUP,
        stim_channel_device_name="iLas2",
        stim_channel_power_property_name="BluePower",
        auto_repeat_stim_exposure=True,
    )
]

# site.make_microscope already assigned site.POWER_PROPERTIES, which is what
# resolve_power() consults. The stim_channel_device_name /
# stim_channel_power_property_name fields above are carried for bookkeeping
# only — without a POWER_PROPERTIES entry, resolve_power() raises on the first
# stim event rather than dropping the requested power silently.
print("power mapping for the stim channel:",
      mic.POWER_PROPERTIES.get(site.FRAP_CHANNEL))

# %% [markdown]
# ## Pipeline
#
# `StimPercentageOfCell` is the stimulator that makes this a *directed
# migration* experiment: it illuminates the leading `stim_cell_percentage` of
# each tracked cell, which is exactly the patterned, per-cell job a galvo is
# for — and it is far cheaper in scan path than a full field.

# %%
from faro.stimulation.percentage_of_cell import StimPercentageOfCell
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.simple import SimpleFE
from faro.feature_extraction.ref import RefFE
from faro.core.pipeline import ImageProcessingPipeline

segmentators = [
    SegmentationMethod(
        name="labels",
        # The original segments with a remote StarDist server at
        # izbniesen.izb.unibe.ch, which is not reachable from here.
        # site.make_segmentator() gives CellposeV4 when installed and a
        # threshold segmentator otherwise, and says which.
        segmentation_class=site.make_segmentator(),
        use_channel=0,
        save_tracked=True,
    )
]

stimulator = StimPercentageOfCell()
feature_extractor = SimpleFE("labels")
tracker = TrackerTrackpy(search_range=25)
optocheck = RefFE(used_mask="labels")

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=feature_extractor,
    tracker=tracker,
    stimulator=stimulator,
    feature_extractor_ref=optocheck,
)
print(f"storage: {path}")

# %% [markdown]
# ## GUI — napari + napari-micromanager
#
# Pick the FOV positions in the **MDA** dock widget after this cell. Click the
# MDA button at least once, or `generate_fov_positions` raises a `KeyError`
# saying the widget is not registered.

# %%
viewer, mm_wdg = site.open_napari(mic, title="21 cell migration (Inscoper)")

# %% [markdown]
# ## No DMD calibration
#
# Deliberately not `mic.calibrate_dmd(...)`. See note 2 at the top: the galvo
# has its own firmware-applied calibration (made by
# `inscoper_useq/scripts/script_frap_calibration.py`), and a DMD affine
# calibrated here would warp the mask a second time.

# %%
assert getattr(mic.mmc, "use_frap_as_slm", False), (
    "This notebook assumes the FRAP galvo is the stimulation device. If this "
    "system really has an SLM, restore the calibrate_dmd call from the "
    "original notebook."
)
assert not mic.uses_dmd_affine, (
    "uses_dmd_affine is True, so stim masks would go through the DMD affine "
    "as well as the firmware's FRAP calibration — transformed twice."
)
print("FRAP is the stimulation device; masks stay in camera pixels.")

# %% [markdown]
# ## Map the experiment onto FOVs
#
# `utils.generate_fov_positions(mic, viewer=viewer)` reads the MDA widget.
# `site.fovs_here(mic, n)` is the rehearsal alternative — *n* copies of the
# field the stage is on now. Do not use `fake_fovs=n`: those positions are all
# `(0, 0)`, i.e. the machine origin.

# %%
USE_GUI_FOVS = True

if USE_GUI_FOVS:
    fovs = utils.generate_fov_positions(mic, viewer=viewer)
else:
    fovs = site.fovs_here(mic, 1)
print(f"{len(fovs)} FOV(s): {fovs}")
site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV, len(fovs))

df_acquire = utils.generate_df_acquire(
    fovs,
    n_frames=N_FRAMES,
    time_between_timesteps=TIME_BETWEEN_TIMESTEPS,
    time_per_fov=TIME_PER_FOV,
    channels=channels,
    condition=condition,
    channel_optocheck=channel_optocheck,
    optocheck_timepoints=optocheck_timepoints,
)
df_acquire = utils.apply_stim_treatments_to_df_acquire(
    df_acquire,
    stim_phase,
    condition,
    n_fovs_per_well=n_fovs_per_well,
    add_stim_exposure_group=ADD_STIM_EXPOSURE_GROUP,
    regular_spacing_between_stimulations=REGULAR_SPACING_BETWEEN_STIMULATIONS,
)
df_acquire

# %% [markdown]
# `df_acquire` carries the whole experiment, and the stimulation module reads
# its metadata — so the fraction of each cell to stimulate can vary per
# timestep. Here the second timestep stimulates 40 % instead of 30 %.

# %%
df_acquire["stim_cell_percentage"] = 0.3
df_acquire.loc[1, "stim_cell_percentage"] = 0.4
df_acquire

# %% [markdown]
# ## Validate before firing
#
# Two independent checks, and both matter on this microscope:
#
# * `ctrl.validate_events` — channel configs exist, exposures are inside the
#   camera's limits, and the pipeline's `required_metadata` (here
#   `stim_cell_percentage`) is present on the events.
# * pricing the **first real stim mask** — the only check that knows whether
#   the galvo can actually fire what this pipeline will produce on *this*
#   sample. `validate_events` cannot know: the mask does not exist until the
#   pipeline has segmented a frame.

# %%
from faro.core.controller import Controller
from faro.core.conversion import df_to_events

events = df_to_events(df_acquire)
print(f"{len(events)} events")

writer = site.make_writer(path)
ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

# Segment one live frame and price the mask the stimulator makes from it --
# the closest thing to a dry run of the burn. site.probe_stim_mask also
# synthesises the `tracks` frame the pipeline would supply, because several
# stimulators return an empty mask without one and would otherwise price a
# free burn for a mask that is in fact expensive.
labels, mask, plan = site.probe_stim_mask(
    mic,
    segmentators[0].segmentation_class,
    stimulator,
    channel=channels[0].config,
    exposure=channels[0].exposure,
    metadata={"stim_cell_percentage": 0.4},   # the largest value used below
)

# %% [markdown]
# ## Run
#
# `stim_mode="current"`: image frame *t*, run the pipeline on it, then fire the
# mask that frame produced. `"previous"` fires frame *t-1*'s mask before
# imaging, which is lower-latency but one timestep stale.

# %%
from faro.widgets import ExperimentStatusWidget

for _ in range(int(SLEEP_BEFORE_EXPERIMENT_START_in_H * 3600)):
    time.sleep(1)

viewer.window.add_dock_widget(
    ExperimentStatusWidget(ctrl), name="experiment status", area="right"
)
handle = ctrl.run_experiment(events, stim_mode="current")
print("running — watch the status dock, or handle.status()")

# %%
handle.status()

# %% [markdown]
# ## Finish and post-process

# %%
handle.wait()
ctrl.finish_experiment()      # drain the pipeline so all tracks are written
utils.generate_exp_data_from_tracks(path)

df_exp = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"{len(df_exp)} cell rows, {df_exp['particle'].nunique()} particles")
df_exp.head()

# %% [markdown]
# ## Release the hardware
#
# `post_experiment()` is a full teardown on this adapter — it closes the bridge
# — so a second run in the same kernel needs a fresh `site.make_microscope()`.
# Clear the ROI first so the next user does not inherit the 512 px crop.

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
