# %% [markdown]
# # Random per-cell 14 px-patch stimulation — Inscoper microscope
#
# The Inscoper variant of `random_stim_per_cell_14px_patches.ipynb`.
# Single-cell optogenetic stimulation with `RandomStimPerCell14pxPatches`: each
# FOV is tiled into a regular grid of 14 px patches, only patches entirely
# inside a cell mask are kept, and one per cell is chosen at random to receive a
# 10 px-diameter dot. The chosen patch is recorded so post-processing can
# re-extract features for that exact patch and correlate them with the local
# response.
#
# Site configuration lives in `../inscoper_site.py`.
#
# **This is the experiment the FRAP galvo is best at.** A handful of small dots
# is exactly the sparse, patterned target a point-scanning device fires cheaply
# — far cheaper than the whole-field burns in `11_*`, which need a cropped
# camera to fit at all. The pricing cell below shows the difference.
#
# **Three things the original does that must not be done here:**
#
# 1. **No `mic.calibrate_dmd(...)`.** The galvo's calibration is applied by the
#    firmware per point, so a DMD affine calibrated in the notebook would
#    transform the mask a second time and aim every dot wrong.
# 2. **No DMD focus checkerboard.** The original arms a checkerboard with
#    `setSLMExposure` / `OverlapMode` to refocus the DMD. There is no SLM here
#    (`getSLMDevice()` returns `""`); `mic.dmd` is a shim whose dimensions
#    mirror the camera FOV.
# 3. **No `OverlapMode` property.** This bridge exposes no device properties at
#    all beyond `Core` and the camera's `Exposure`, so `getProperty(...)` on the
#    stim device raises.

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
    Channel,
    PowerChannel,
    RTMSequence,
    SegmentationMethod,
    combine,
)
import faro.core.utils as utils

# %%
# 512 px. The dots themselves are cheap at any frame size; the crop is here to
# keep the readout fast enough for a 30 s interval across many FOVs.
mic = site.make_microscope(roi=512)

# %% [markdown]
# ## 2. Configuration

# %%
SLEEP_BEFORE_EXPERIMENT_START_in_H = 0

INTERVAL_S = 30          # seconds between frames
TIME_PER_FOV_S = 5.0     # seconds to image one FOV

# Frame counts per phase (0-indexed within the single phase)
N_BASELINE = 10          # imaging frames before the stim block
N_STIM = 5               # consecutive stim frames (mask stays fixed across these)
N_RECOVERY = 20          # imaging frames after the stim block
N_FRAMES = N_BASELINE + N_STIM + N_RECOVERY

STIM_FRAMES = list(range(N_BASELINE, N_BASELINE + N_STIM))

## Storage
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "23_random_stim_per_cell_14px_patches_inscoper"
path = os.path.join(base_path, experiment_name)

## Channels — the original's mScarlet3 / miRFP / mCitrine do not exist here.
# Plain Channel, not PowerChannel: an Inscoper .cbc already carries its own
# wavelength and power, so there is nothing left to push.
stim_channel = Channel(config=site.FRAP_CHANNEL, exposure=300, group=site.CHANNEL_GROUP)

imaging_channels = (
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),   # reporter
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),   # second marker
)

optocheck_channel = Channel(config="640 WF", exposure=80, group=site.CHANNEL_GROUP)

site.check_fov_timing(INTERVAL_S, TIME_PER_FOV_S)
print(f"Timing: {N_FRAMES} frames @ {INTERVAL_S}s = {(N_FRAMES - 1) * INTERVAL_S / 60:.1f} min")
print(f"Baseline : frames 0..{N_BASELINE - 1}")
print(f"Stim     : frames {STIM_FRAMES[0]}..{STIM_FRAMES[-1]}")
print(f"Recovery : frames {STIM_FRAMES[-1] + 1}..{N_FRAMES - 1}")
print(f"Optocheck: last frame ({N_FRAMES - 1})")
print(f"storage  : {path}")

# %% [markdown]
# ## 3. Pipeline
#
# `RandomStimPerCell14pxPatches(seed=0)` seeds its choice by
# `(fov, fov_timestep)`, so re-running post-processing reproduces the exact
# patches that were illuminated live. Patch coordinates accumulate in memory
# during the run and are merged into `exp_data.parquet` afterwards.

# %%
from faro.stimulation.random_stim_per_cell_14px_patches import (
    RandomStimPerCell14pxPatches,
)
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.erk_ktr import FE_ErkKtr
from faro.feature_extraction.optocheck import OptoCheckFE
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.controller import Controller

segmentators = [
    SegmentationMethod(
        name="labels",
        segmentation_class=site.make_segmentator(),
        use_channel=0,
        save_tracked=True,
    )
]

stimulator = RandomStimPerCell14pxPatches(seed=0)
feature_extractor = FE_ErkKtr("labels")
tracker = TrackerTrackpy(search_range=30)
optocheck = OptoCheckFE(used_mask="labels")

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=feature_extractor,
    tracker=tracker,
    stimulator=stimulator,
    feature_extractor_ref=optocheck,
)

writer = site.make_writer(path)

# %% [markdown]
# ## 4. FRAP preflight, and what a dot pattern costs
#
# Worth doing side by side with the whole-field number: the same rig that
# refuses a 512 px flood fires a few dozen dots in a fraction of a second.
# That asymmetry is the reason this experiment ports to Inscoper cleanly and the
# `11_*` full-FOV ones need a cropped camera.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

print("--- what a whole-field burn would cost on this frame ---")
site.price_full_field(mic, verbose=True)

print("\n--- what this notebook actually fires ---")
labels, mask, plan = site.probe_stim_mask(
    mic,
    segmentators[0].segmentation_class,
    stimulator,
    channel=imaging_channels[0].config,
    exposure=imaging_channels[0].exposure,
    metadata={"fov": 0, "fov_timestep": 0},
)
assert plan is not None or mask is None, (
    "the dot pattern is over budget, which should not happen — check that the "
    "segmentation is not producing thousands of objects"
)

# %% [markdown]
# ## 5. GUI
#
# Pick the FOV positions in the **MDA** dock widget.

# %%
viewer, mm_wdg = site.open_napari(mic, title="23 random per-cell patches (Inscoper)")

# %% [markdown]
# ## 6. No DMD calibration, no DMD focus checkerboard
#
# The original has three cells here: `calibrate_dmd`, arm-checkerboard, and
# disarm-checkerboard. All three are wrong on this microscope — see notes 1–3 at
# the top. The assertions below are what is left of them: they confirm the
# assumption those cells would have violated.

# %%
assert getattr(mic.mmc, "use_frap_as_slm", False), (
    "expected the FRAP galvo to be the stim device"
)
assert not mic.uses_dmd_affine, (
    "uses_dmd_affine is True — the dots would be warped twice"
)
print(f"stim device: {mic.dmd.name} (FRAP galvo), "
      f"mask space {mic.dmd.width}x{mic.dmd.height} = camera FOV")

# %% [markdown]
# ## 7. Build the acquisition

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
    ref_channels=(optocheck_channel,),
    ref_frames=[-1],
    rtm_metadata={
        "phase_name": "RandomPatchStim",
        "phase_id": 0,
        "treatment_name": "Random per-cell 14px-patch stimulation",
    },
)

# combine() flattens the RTMSequence into per-event records; axis="t" is a
# no-op for a single phase but is what apply_fov_batching consumes.
events = combine(phase, axis="t")
events = utils.apply_fov_batching(events, time_per_fov=TIME_PER_FOV_S)

print(f"Total events: {len(events)}")
utils.events_to_dataframe(events).sort_values("timestep").head()

# %% [markdown]
# ## 8. Validate

# %%
ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

from faro.widgets import ExperimentStatusWidget

viewer.window.add_dock_widget(
    ExperimentStatusWidget(ctrl), name="experiment status", area="right"
)

# %% [markdown]
# ## 9. Run

# %%
for _ in range(int(SLEEP_BEFORE_EXPERIMENT_START_in_H * 3600)):
    time.sleep(1)

handle = ctrl.run_experiment(events, stim_mode="current")
print("running — watch the status dock, or handle.status()")

# %%
handle.status()

# %% [markdown]
# ## 10. Post-processing
#
# The patch coordinates the stimulator accumulated are merged into
# `exp_data.parquet` on `(fov, fov_timestep, label)`. Stim-frame rows for
# stimulated cells gain `patch_y_min`, `patch_x_min`, `patch_y_max`,
# `patch_x_max`, `patch_dot_y`, `patch_dot_x`; every other row gets NaN.

# %%
handle.wait()
ctrl.finish_experiment()
utils.generate_exp_data_from_tracks(path)

exp_data_path = os.path.join(path, "exp_data.parquet")
df_exp = pd.read_parquet(exp_data_path)
df_patches = stimulator.to_dataframe()
if not df_patches.empty:
    df_patches = df_patches.drop_duplicates(subset=["fov", "fov_timestep", "label"])
    df_exp = df_exp.merge(df_patches, on=["fov", "fov_timestep", "label"], how="left")
    df_exp.to_parquet(exp_data_path)
print(f"Merged {len(df_patches)} patch rows into {exp_data_path}")
df_exp.head()

# %% [markdown]
# ## 11. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
