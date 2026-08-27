# %% [markdown]
# # ERK-KTR Full FOV Stimulation (RTMSequence) — Inscoper microscope
#
# The Inscoper variant of `stim_rtmsequence.ipynb`. Multi-phase optogenetic
# experiment with **whole-field** stimulation, built from `RTMSequence` phases.
#
# Site configuration lives in `../inscoper_site.py`.
#
# **Workflow**
# 1. Load the microscope, crop it to a stimulable field, preflight FRAP
# 2. Configure the pipeline (segmentation, tracking, ERK-KTR features)
# 3. Pick FOVs in napari, build the phases, chain them with `combine`
# 4. Validate — including pricing the actual whole-field burn — and run
# 5. Merge tracks into `exp_data.parquet`

# %% [markdown]
# ## The one thing that does not port: "full FOV"
#
# The original stimulates the whole field with a lamp — `StimWholeFOV` returns
# `True`, the engine flashes the stim channel, done. Here `StimWholeFOV`'s
# `True` becomes an all-on **camera-space mask** which the bridge hands to the
# FRAP galvo, and a galvo does not flood a field: it rasters it, one hatched
# row at a time, inside a fixed budget of ~36,000 px of scan path (~5 s).
#
# On the native 2304x2304 frame that budget is exceeded at every usable row
# pitch, so `plan_mask` **refuses** — an exception at the first stim event, not
# a half-burnt field you discover in analysis. `site.FULL_FOV_NOTE` carries the
# measured table; the short version, at this rig's 4 px pitch:
#
# | field | whole-field burn |
# |-------|------------------|
# | 256 px | 2.3 s — fits comfortably |
# | 352 px | 4.3 s — fits, close to the limit |
# | 384 px and up | REFUSED |
#
# So this notebook **crops the camera to 256 px** and means it: the field it
# images is the field it can stimulate. That is a real change in what the
# experiment measures — fewer cells per FOV — and the honest alternative is to
# stop asking for a flood and stimulate patterned regions instead, which is what
# `21_cell_migration_inscoper.ipynb` and `25_region_percentage_stim_inscoper.ipynb`
# do.
#
# The crop pays for itself twice over: a 256 px frame reads out fast enough
# that many FOVs actually fit inside the interval.

# %% [markdown]
# ## 1. Microscope

# %%
import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

from faro.core.data_structures import (
    Channel,        # basic imaging channel (config, exposure, group)
    PowerChannel,   # adds light-source power; see the note below
    RTMSequence,    # one phase: time plan, channels, stim frames, ref frames
    SegmentationMethod,
    combine,        # chain phases along t (or run them in parallel along p)
)
import faro.core.utils as utils

# %%
# 256 so the whole field is stimulable in one fire — see the table above.
# 352 also fits (4.3 s) if you need more cells per FOV and can afford the
# scan time inside your interval.
STIM_FIELD_PX = 256

mic = site.make_microscope(roi=STIM_FIELD_PX)

# %% [markdown]
# ## 2. Channels
#
# The original's `miRFP` / `mScarlet3` / `mCitrine` / `CyanStim` do not exist in
# this configuration — `utils.validate_hardware` reports them as missing — so
# they are re-mapped onto `site.IMAGING_CHANNELS` and `site.FRAP_CHANNEL`.
#
# **`Channel` rather than `PowerChannel`.** An Inscoper `.cbc` already selects
# its wavelength *and* its power (`470 WF` ships
# `Lumencor_EpiFluorescence-470 Intensity (%) = 25`), so a plain `Channel` is
# fully specified. `PowerChannel` still works — `site.POWER_PROPERTIES` maps
# every channel here to its device and property, and `make_microscope` has
# already assigned it to `mic.POWER_PROPERTIES` — but note that this bridge does
# not enumerate device properties, so `validate_events` cannot range-check the
# power it is handed. A wrong value fails at the hardware, not at validation.

# %%
## Stimulation channel — the one that opens the FRAP light path.
# Its exposure is bookkeeping on this rig: the burn's duration comes from the
# scan (~2.3 s for a 256 px field) and its dose from the repetition count, not
# from this number.
stim_channel = Channel(
    config=site.FRAP_CHANNEL, exposure=100, group=site.CHANNEL_GROUP
)

## Imaging channels — every timepoint. Channel 0 is what gets segmented.
imaging_channels = (
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),   # nuclear marker
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),   # ERK-KTR reporter
)

## Optocheck / reference channel — acquired only on ref_frames.
optocheck_channel = Channel(
    config="640 WF", exposure=80, group=site.CHANNEL_GROUP
)

## Storage
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "11_stim_rtmsequence_inscoper"
path = os.path.join(base_path, experiment_name)

SLEEP_BEFORE_EXPERIMENT_START_in_H = 0
print(f"storage: {path}")

# %% [markdown]
# ## 3. FRAP preflight, and pricing the burn this notebook will actually fire
#
# `site.price_full_field` prices exactly what `StimWholeFOV` produces — an
# all-on mask the size of the current frame. If it refuses, the crop above is
# too generous and nothing downstream can work.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

plan = site.price_full_field(mic)
assert plan is not None, (
    f"a whole-field burn on a {mic.image_width}x{mic.image_height} frame is "
    "over budget. Lower STIM_FIELD_PX (256 is known to fit) or switch to a "
    "patterned stimulator."
)

# %% [markdown]
# ## 4. Pipeline
#
# Same components as the original, with two substitutions the environment
# forces: `site.make_segmentator()` (cellpose is not installed here) and
# `site.make_writer()`, which picks `OmeZarrWriter` when `ome_writers` is
# importable and `TiffWriter` otherwise. That choice is made up front because
# `OmeZarrWriter` constructs fine without the package and fails *inside*
# `init_stream` — after the run has started — which is the difference between
# a printed line and a dead acquisition.

# %%
from faro.stimulation.base import StimWholeFOV
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.erk_ktr import FE_ErkKtr
from faro.feature_extraction.optocheck import OptoCheckFE
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.controller import Controller

segmentators = [
    SegmentationMethod(
        name="labels",
        segmentation_class=site.make_segmentator(),
        use_channel=0,              # segment the nuclear marker
        save_tracked=True,
    )
]

stimulator = StimWholeFOV()                   # all-on mask -> full-field raster
feature_extractor = FE_ErkKtr("labels")       # cytoplasmic/nuclear ratio
tracker = TrackerTrackpy(search_range=25)     # 25 px at 256 px frame
optocheck = OptoCheckFE(used_mask="labels")

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=feature_extractor,
    tracker=tracker,
    stimulator=stimulator,
    feature_extractor_ref=optocheck,          # runs only on ref_frames
)

writer = site.make_writer(path)

# %% [markdown]
# ## 5. GUI
#
# Pick the FOV positions in the **MDA** dock widget. Click the MDA button at
# least once, or `generate_fov_positions` raises a `KeyError` saying the widget
# is not registered.

# %%
viewer, mm_wdg = site.open_napari(mic, title="11 full-FOV stim (Inscoper)")

# %% [markdown]
# ## 6. Build the acquisition
#
# `stim_frames` and `ref_frames` are **per-phase** and 0-indexed; `combine`
# offsets timepoints and `min_start_time` when it chains them. Negative indices
# count from the end, so `ref_frames=[-1]` is "last frame of this phase".
#
# `apply_fov_batching` keeps the timing honest when the FOVs cannot all be
# imaged inside one interval: they are split into sequential batches, which
# means each FOV is revisited every *N* intervals rather than every one.

# %%
INTERVAL_S = 20.0
TIME_PER_FOV_S = 3.0

USE_GUI_FOVS = True
if USE_GUI_FOVS:
    fov_positions = utils.generate_fov_positions(mic, viewer=viewer)
else:
    # Rehearsal without the GUI. NOT utils.generate_fov_positions(fake_fovs=n),
    # whose positions are all (0, 0) — the machine origin.
    fov_positions = site.fovs_here(mic, 1)
print(f"{len(fov_positions)} FOV(s)")
site.check_fov_timing(INTERVAL_S, TIME_PER_FOV_S, len(fov_positions))

# Phase 1: baseline imaging with one stim pulse.
phase_1 = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": 4},
    stage_positions=fov_positions,
    channels=imaging_channels,
    stim_channels=(stim_channel,),
    stim_frames=range(1, 2),          # frame 1 only, within this phase
    rtm_metadata={
        "phase_name": "PreDrug",
        "phase_id": 0,
        "treatment_name": "Priming Phase 1 pre Drug",
    },
)

# Phase 2: sustained stim, plus the optocheck on the last frame.
phase_2 = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": 4},
    stage_positions=fov_positions,
    channels=imaging_channels,
    stim_channels=(stim_channel,),
    stim_frames=range(1, 3),          # frames 1 and 2
    ref_channels=(optocheck_channel,),
    ref_frames=[-1],
    rtm_metadata={
        "phase_name": "PostDrug",
        "phase_id": 1,
        "treatment_name": "Sustained Phase 2 post Drug",
    },
)

events = combine(phase_1, phase_2, axis="t")
events = utils.apply_fov_batching(events, time_per_fov=TIME_PER_FOV_S)

print(f"Total events: {len(events)}")
utils.events_to_dataframe(events).sort_values("timestep")

# %% [markdown]
# ## 7. Validate
#
# `validate_events` checks the channel configs exist, the exposures are inside
# the camera's limits, and the pipeline's `required_metadata` is present. It
# does *not* know whether the galvo can fire the mask — that is what the
# pricing cell above is for, and on this microscope the two together are the
# pre-flight.

# %%
ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

from faro.widgets import ExperimentStatusWidget

viewer.window.add_dock_widget(
    ExperimentStatusWidget(ctrl), name="experiment status", area="right"
)

# %% [markdown]
# ## 8. Run
#
# Non-blocking: returns a `RunHandle` immediately. `stim_mode="current"` images
# frame *t*, runs the pipeline on it, then fires. napari-micromanager keeps
# routing frames into its preview layer during the run, so the live link does
# not need tearing down first.

# %%
for _ in range(int(SLEEP_BEFORE_EXPERIMENT_START_in_H * 3600)):
    time.sleep(1)

handle = ctrl.run_experiment(events, stim_mode="current")
print("running — watch the status dock, or handle.status()")

# %%
handle.status()

# %% [markdown]
# ## 9. Post-processing

# %%
handle.wait()
ctrl.finish_experiment()                 # flush the pipeline queue, close the writer
utils.generate_exp_data_from_tracks(path)

df_exp = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"{len(df_exp)} cell rows across {df_exp['timestep'].nunique()} timesteps")
df_exp.head()

# %% [markdown]
# ## 10. Release the hardware
#
# Clear the crop so the next user does not inherit a 256 px frame, then close
# the bridge. `post_experiment` is a full teardown here, so another run in this
# kernel needs a fresh `site.make_microscope()`.

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
