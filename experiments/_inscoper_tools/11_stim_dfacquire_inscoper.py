# %% [markdown]
# # ERK-KTR Full FOV Stimulation (df_acquire API) — Inscoper microscope
#
# The Inscoper variant of `stim_dfacquire.ipynb`. Functionally the same
# experiment as `11_stim_rtmsequence_inscoper.ipynb`, built through the legacy
# **`df_acquire` DataFrame API** instead of `RTMSequence` phases.
#
# **When to use which**
#
# * **`RTMSequence`** (recommended) — phases are declared as objects and chained
#   with `combine`. Stim frames, ref frames and channels are per-phase. See
#   `11_stim_rtmsequence_inscoper.ipynb`.
# * **`df_acquire`** (this notebook) — full control over the acquisition
#   DataFrame. Worth it for per-FOV condition assignment, wellplate layouts, or
#   custom stim schedules that do not map cleanly onto phases. Every row is one
#   (FOV, timestep), and you can edit it with pandas before it becomes events.
#
# Site configuration lives in `../inscoper_site.py`.
#
# **The Inscoper-specific constraint is the same one as in the RTMSequence
# variant, and it is not about the API:** whole-field stimulation goes to a FRAP
# galvo, which rasters rather than floods, inside a ~36,000 px scan budget
# (~5 s). That is exceeded at every usable row pitch on the native 2304x2304
# frame, so the camera is cropped to a field the galvo can actually cover.
# `site.FULL_FOV_NOTE` has the measured table.

# %% [markdown]
# ## 1. Imports and microscope

# %%
import os
import sys
import time
from pprint import pprint

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

from faro.core.data_structures import (
    Channel,           # basic imaging channel (config, exposure, group)
    StimTreatment,     # df_acquire-specific: the stim schedule for a condition
    SegmentationMethod,
)
import faro.core.utils as utils

# %%
# 256 px so the whole field is stimulable in one fire. 352 also fits (4.3 s).
STIM_FIELD_PX = 256
mic = site.make_microscope(roi=STIM_FIELD_PX)

# %% [markdown]
# ## 2. Configuration
#
# > **RTMSequence equivalent:** `time_plan={"interval": 60.0, "loops": 100}`
# > replaces `N_FRAMES_PHASE_*` and `TIME_BETWEEN_TIMESTEPS`, and
# > `utils.apply_fov_batching(events, time_per_fov=...)` replaces `TIME_PER_FOV`.

# %%
N_FRAMES_PHASE_1 = 8       # timesteps in phase 1 (pre-drug)
N_FRAMES_PHASE_2 = 10      # timesteps in phase 2 (post-drug)

SLEEP_BEFORE_EXPERIMENT_START_in_H = 0

TIME_BETWEEN_TIMESTEPS = 30      # seconds between timesteps
TIME_PER_FOV = 4.0               # seconds per FOV

# df_acquire-specific bookkeeping options; RTMSequence needs neither.
ADD_STIM_EXPOSURE_GROUP = False
REGULAR_SPACING_BETWEEN_STIMULATIONS = False

site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV)

## Storage
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "11_stim_dfacquire_inscoper"
path = os.path.join(base_path, experiment_name)

## Imaging channels — order matters, channel 0 is segmented.
# Plain Channel, not PowerChannel: an Inscoper .cbc already carries its own
# wavelength and power. site.POWER_PROPERTIES maps every channel here to its
# (device, property) if you do want PowerChannel — but note this bridge does not
# enumerate device properties, so validate_events cannot range-check a power.
channels = [
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),   # nuclear marker
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),   # ERK-KTR reporter
]

## Optocheck channel — the optogenetic-tool readout.
# RTMSequence equivalent: ref_channels=(...), ref_frames=[-1]
channel_optocheck = Channel(config="640 WF", exposure=80, group=site.CHANNEL_GROUP)
optocheck_timepoints = (N_FRAMES_PHASE_2 - 1,)   # absolute index within phase 2

## Conditions assigned to FOVs
condition = ["Drug"]
n_fovs_per_well = None           # an int for wellplate experiments

print(f"storage: {path}")

# %% [markdown]
# ## 3. Stimulation treatments
#
# `StimTreatment` carries the whole schedule: which timesteps, what exposure,
# what power, which channel.
#
# > **RTMSequence equivalent:** `stim_channels=(PowerChannel(...),)` plus
# > `stim_frames=range(10, 100)`, which are per-phase and auto-offset on
# > `combine`. `stim_exposure` may be a scalar or a list, so
# > `auto_repeat_stim_exposure` is unnecessary there.
#
# **On this microscope `stim_exposure_list` and `stim_power` do not set the
# dose.** They describe a lamp pulse. The FRAP burn takes its laser line and
# power from the designated channel's `.cbc` and its dose from the repetition
# count, so its duration is the scan time priced below — not `100 ms`. Keep
# `stim_channel_name` pointing at the FRAP channel so the recorded light path
# and the fired one are the same.

# %%
stim_phase_1 = [
    StimTreatment(
        treatment_name="Priming Phase 1 pre Drug",
        stim_timestep=tuple(range(2, N_FRAMES_PHASE_1, 1)),
        stim_exposure_list=100,          # bookkeeping — see above
        auto_repeat_stim_exposure=True,
        stim_power=100,                  # ditto
        stim_channel_name=site.FRAP_CHANNEL,
        stim_channel_group=site.CHANNEL_GROUP,
        stim_channel_device_name="iLas2",
        stim_channel_power_property_name="BluePower",
    )
]

stim_phase_2 = [
    StimTreatment(
        treatment_name="Sustained Phase 2 post Drug",
        stim_timestep=tuple(range(2, N_FRAMES_PHASE_2, 1)),
        stim_exposure_list=100,
        auto_repeat_stim_exposure=True,
        stim_power=100,
        stim_channel_name=site.FRAP_CHANNEL,
        stim_channel_group=site.CHANNEL_GROUP,
        stim_channel_device_name="iLas2",
        stim_channel_power_property_name="BluePower",
    )
]

for phase in (stim_phase_1, stim_phase_2):
    utils.print_stim_exposures_timesteps(phase)

# %% [markdown]
# ## 4. FRAP preflight and the whole-field price
#
# The one check that has no `df_acquire` or `RTMSequence` equivalent: can the
# galvo fire the mask this pipeline produces? `StimWholeFOV` returns an all-on
# mask the size of the frame, so `site.price_full_field` prices exactly it.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

plan = site.price_full_field(mic)
assert plan is not None, (
    f"a whole-field burn on a {mic.image_width}x{mic.image_height} frame is over "
    "budget. Lower STIM_FIELD_PX (256 is known to fit) or use a patterned "
    "stimulator instead."
)

# %% [markdown]
# ## 5. Pipeline
#
# Shared between both APIs. Note `feature_extractor_ref` — it was renamed from
# `feature_extractor_optocheck` in an earlier version of faro, and the old name
# is silently ignored rather than rejected.

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
        # The original uses a remote StarDist server at izbniesen.izb.unibe.ch,
        # which is not reachable from here.
        segmentation_class=site.make_segmentator(),
        use_channel=0,
        save_tracked=True,
    )
]

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=FE_ErkKtr("labels"),
    tracker=TrackerTrackpy(search_range=25),
    stimulator=StimWholeFOV(),
    feature_extractor_ref=OptoCheckFE(used_mask="labels"),
)

writer = site.make_writer(path)

# %% [markdown]
# ## 6. GUI
#
# Pick the FOV positions in the **MDA** dock widget.

# %%
viewer, mm_wdg = site.open_napari(mic, title="11 stim df_acquire (Inscoper)")

# %% [markdown]
# ## 7. Build the acquisition DataFrames
#
# One DataFrame per phase, then `df_to_events()`.
#
# * `generate_fov_positions(mic, viewer=viewer)` reads the MDA widget
#   (`generate_fov_objects` is a backwards-compat alias)
# * `generate_df_acquire(fovs, ...)` — one row per (FOV, timestep)
# * `apply_stim_treatments_to_df_acquire(df, stim, ...)` — maps the
#   `StimTreatment` objects onto those rows
#
# > **RTMSequence equivalent:** all three are replaced by declaring
# > `RTMSequence(time_plan=..., stage_positions=..., channels=...,
# > stim_channels=..., stim_frames=...)` and chaining with `combine`.

# %%
USE_GUI_FOVS = True
if USE_GUI_FOVS:
    fovs = utils.generate_fov_positions(mic, viewer=viewer)
else:
    # Rehearsal. NOT fake_fovs=n, whose positions are the machine origin.
    fovs = site.fovs_here(mic, 1)
print(f"{len(fovs)} FOV(s)")
site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV, len(fovs))

# Phase 1 — pre-drug
df_acquire = utils.generate_df_acquire(
    fovs,
    n_frames=N_FRAMES_PHASE_1,
    time_between_timesteps=TIME_BETWEEN_TIMESTEPS,
    time_per_fov=TIME_PER_FOV,
    channels=channels,
    phase_name="PreDrug",
    phase_id=0,
    condition=condition,
)
df_acquire = utils.apply_stim_treatments_to_df_acquire(
    df_acquire,
    stim_phase_1,
    condition,
    n_fovs_per_well=n_fovs_per_well,
    add_stim_exposure_group=ADD_STIM_EXPOSURE_GROUP,
    regular_spacing_between_stimulations=REGULAR_SPACING_BETWEEN_STIMULATIONS,
)
print(f"phase 1: {len(df_acquire)} rows, {int(df_acquire['stim'].sum())} stim")
df_acquire.head()

# %%
# Phase 2 — post-drug, with the optocheck channel on its own timepoints.
# df_acquire takes the optocheck as parameters here; note optocheck_timepoints
# are ABSOLUTE indices within this phase, not per-phase offsets as
# RTMSequence's ref_frames are.
df_acquire_2 = utils.generate_df_acquire(
    fovs,
    n_frames=N_FRAMES_PHASE_2,
    time_between_timesteps=TIME_BETWEEN_TIMESTEPS,
    time_per_fov=TIME_PER_FOV,
    channels=channels,
    channel_optocheck=channel_optocheck,
    optocheck_timepoints=optocheck_timepoints,
    phase_name="PostDrug",
    phase_id=1,
    condition=condition,
)
df_acquire_2 = utils.apply_stim_treatments_to_df_acquire(
    df_acquire_2,
    stim_phase_2,
    condition,
    n_fovs_per_well=n_fovs_per_well,
    add_stim_exposure_group=ADD_STIM_EXPOSURE_GROUP,
    regular_spacing_between_stimulations=REGULAR_SPACING_BETWEEN_STIMULATIONS,
)
print(f"phase 2: {len(df_acquire_2)} rows, {int(df_acquire_2['stim'].sum())} stim")
df_acquire_2.head()

# %% [markdown]
# ## 8. Run phase 1
#
# The original creates a second `Controller` for phase 2. Use
# `ctrl.continue_experiment(...)` instead: it reuses the Analyzer, so tracking
# state, timestep counters and filenames carry across the phase boundary. A
# fresh `Controller` restarts all three, which is what makes phase 2's tracks
# unlinkable to phase 1's.

# %%
from faro.core.conversion import df_to_events
from faro.widgets import ExperimentStatusWidget

for _ in range(int(SLEEP_BEFORE_EXPERIMENT_START_in_H * 3600)):
    time.sleep(1)

events = df_to_events(df_acquire)
ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "phase 1 validation reported problems"

viewer.window.add_dock_widget(
    ExperimentStatusWidget(ctrl), name="experiment status", area="right"
)
handle = ctrl.run_experiment(events, stim_mode="current")
print(f"phase 1 running: {len(events)} events")

# %% [markdown]
# ## 9. Pipette the drug, then run phase 2
#
# Phase 1 must finish before phase 2 starts on the same microscope, so this
# blocks first. Add the drug while it is blocked.

# %%
handle.wait()
print("phase 1 done — add the drug now, then run the next cell")

# %%
events_2 = df_to_events(df_acquire_2)
assert ctrl.validate_events(events_2), "phase 2 validation reported problems"
handle2 = ctrl.continue_experiment(events_2, stim_mode="current")
print(f"phase 2 running: {len(events_2)} events")

# %% [markdown]
# ## 10. Post-processing

# %%
handle2.wait()
ctrl.finish_experiment()          # drain the pipeline so all tracks are written
utils.generate_exp_data_from_tracks(path)

df_exp = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"{len(df_exp)} cell rows")

# The two APIs name the phase column differently, and neither validates it:
# generate_df_acquire(phase_name=...) writes a column called "phase", while
# RTMSequence(rtm_metadata={"phase_name": ...}) writes "phase_name". Grouping on
# the wrong one is a KeyError at analysis time, long after the run.
phase_col = next((c for c in ("phase", "phase_name", "phase_id") if c in df_exp.columns), None)
if phase_col:
    print()
    print(f"per-phase summary (column {phase_col!r}):")
    print(df_exp.groupby(phase_col)["timestep"].agg(["min", "max", "count"]))
else:
    print(f"no phase column; columns are {sorted(df_exp.columns)}")
df_exp.head()

# %% [markdown]
# ## 11. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
