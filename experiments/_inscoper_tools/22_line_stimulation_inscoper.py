# %% [markdown]
# # Dynamic line pattern — Inscoper microscope
#
# The Inscoper variant of `line_stimulation.ipynb`. A stripe sweeps across the
# field, one step per timestep, projected as a stimulation pattern. There is no
# segmentation and no tracking: `StimLine` is a plain `Stim`, so its mask is a
# function of the timestep alone.
#
# Site configuration lives in `../inscoper_site.py`.
#
# **What changes on a FRAP galvo, and it changes the geometry:**
#
# A stripe is the *worst-case* shape for a point-scanning device. A galvo rasters
# a filled region row by row inside a fixed budget of ~36,000 px of scan path
# (~5 s), and a stripe spanning the full width of the frame is close to a
# whole-field burn: `stripe_width * frame_width` pixels, hatched at this rig's
# 4 px row pitch. The original's `stripe_width=278` on a 2304 px frame is far
# over budget and `plan_mask` refuses it.
#
# So this notebook picks the frame and the stripe together, and the cell below
# prices **every timestep's stripe** before the run rather than discovering the
# refusal at the first stim event. That sweep is the whole pre-flight here — a
# dynamic pattern has a different mask every frame, so pricing one of them
# proves nothing about the rest.

# %% [markdown]
# ## 1. Imports and microscope

# %%
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

from faro.core.data_structures import Channel, SegmentationMethod
import faro.core.utils as utils

# %%
# 256 px. A stripe is nearly a whole-field burn, so the frame has to be one
# whose whole field is stimulable — see site.FULL_FOV_NOTE.
FRAME_PX = 256
mic = site.make_microscope(roi=FRAME_PX)

# %% [markdown]
# ## 2. Configuration

# %%
FIRST_FRAME_STIMULATION = 1
N_FRAMES = 30

SLEEP_BEFORE_EXPERIMENT_START_in_H = 0

TIME_BETWEEN_TIMESTEPS = 15     # seconds between frames
TIME_PER_FOV = 5                # seconds per FOV

MIX_STIM_EXPOSURE = False
ADD_STIM_EXPOSURE_GROUP = False
REGULAR_SPACING_BETWEEN_STIMULATIONS = False

site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV)

## Storage
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "22_line_stimulation_inscoper"
path = os.path.join(base_path, experiment_name)

## Channels — the original's miRFP / mRuby2 / mTurquoise / CyanStim do not
## exist in this configuration.
channels = [
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),
]

channel_optocheck = Channel(config="640 WF", exposure=60, group=site.CHANNEL_GROUP)
optocheck_timepoints = (N_FRAMES - 1,)

condition = ["test"]
n_fovs_per_well = None

print(f"storage: {path}")

# %% [markdown]
# ## 3. The stripe, sized for the galvo
#
# `StimLine` sweeps a stripe of `stripe_width` pixels through a full loop every
# `frames_for_1_loop` timesteps. On a lamp-plus-DMD rig the width is a purely
# biological choice; here it is also the single biggest term in the scan cost, so
# it is chosen against the budget.
#
# `STRIPE_WIDTH` below is deliberately a fraction of the frame rather than the
# original's fixed 278 px: the original number is meaningless once the frame is
# cropped, and expressing it as a fraction keeps the *pattern* the same when you
# change `FRAME_PX`.

# %%
from faro.stimulation.moving_line_20x import StimLine

STRIPE_FRACTION = 0.25                              # of the frame width
STRIPE_WIDTH = max(4, int(FRAME_PX * STRIPE_FRACTION))
FRAMES_FOR_1_LOOP = 20

stimulator = StimLine(
    first_stim_frame=FIRST_FRAME_STIMULATION,
    frames_for_1_loop=FRAMES_FOR_1_LOOP,
    stripe_width=STRIPE_WIDTH,
    n_frames_total=N_FRAMES,
)
print(f"stripe {STRIPE_WIDTH} px on a {FRAME_PX} px frame, "
      f"one loop every {FRAMES_FOR_1_LOOP} frames")

# %% [markdown]
# ## 4. FRAP preflight, and price *every* stripe
#
# `StimLine` is a plain `Stim`: `get_stim_mask(metadata)` takes no labels, no
# image and no tracks, so the mask for every timestep can be computed here,
# offline, before anything moves. A dynamic pattern has a different mask each
# frame — the widest one is the one that has to fit.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

img_shape = (mic.image_height, mic.image_width)


def stripe_at(t):
    """The mask StimLine produces at timestep *t*.

    `stim: True` is not optional. StimLine reads `metadata["stim"]` and returns
    an all-zero mask when it is falsy, so a pricing loop that omits it prices
    30 empty masks and reports that everything fits.
    """
    mask, _ = stimulator.get_stim_mask(
        {"timestep": t, "img_shape": img_shape, "stim": True}
    )
    return np.asarray(mask)


worst_px, worst_t, refused = -1, None, []
for t in range(N_FRAMES):
    lit = int(np.count_nonzero(stripe_at(t)))
    if lit > worst_px:
        worst_px, worst_t = lit, t
    if lit and site.price_mask(mic, stripe_at(t), verbose=False) is None:
        refused.append((t, lit))

assert worst_px > 0, (
    "every stripe is empty — StimLine produced nothing for any timestep, so "
    "nothing would be projected"
)
print(f"widest stripe: {worst_px} px lit at timestep {worst_t} "
      f"({worst_px / (img_shape[0] * img_shape[1]) * 100:.0f}% of the frame)")
site.price_mask(mic, stripe_at(worst_t))

if refused:
    raise AssertionError(
        f"{len(refused)} of {N_FRAMES} timesteps are over the scan budget "
        f"(first: t={refused[0][0]}, {refused[0][1]} px lit). Lower "
        "STRIPE_FRACTION or FRAME_PX — plan_mask raises at the first such "
        "stim event, so a run started now dies partway through."
    )
print(f"all {N_FRAMES} stripes fit the scan budget.")

# %% [markdown]
# ## 5. Pipeline
#
# No segmentator, no tracker, no feature extractor: the stripe does not depend
# on the image. `StimLine` needs only `timestep` and `img_shape`, both of which
# the pipeline supplies as metadata.

# %%
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.controller import Controller

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=None,
    feature_extractor=None,
    tracker=None,
    stimulator=stimulator,
    feature_extractor_ref=None,
)
writer = site.make_writer(path)

# %% [markdown]
# ## 6. Stimulation schedule
#
# The original builds `stim_exposures_timesteps` dicts by hand and then
# reimplements most of `generate_df_acquire` and
# `apply_stim_treatments_to_df_acquire` inline — roughly 130 lines that assign
# treatments to FOVs, expand exposures per timestep and derive the `stim`
# column. Those two helpers now do all of it, so this uses them.
#
# On the exposure fields: `stim_exposure` describes a *lamp* pulse. The FRAP
# burn's duration is its scan (priced above) and its dose is the repetition
# count, so these numbers are bookkeeping — carried into `exp_data.parquet` so
# the record says which light path fired, not used to time the burn.

# %%
from faro.core.data_structures import StimTreatment

stim_phase = [
    StimTreatment(
        treatment_name="LinePattern",
        stim_timestep=tuple(range(FIRST_FRAME_STIMULATION, N_FRAMES, 1)),
        stim_exposure_list=200,          # bookkeeping — see above
        auto_repeat_stim_exposure=True,
        stim_power=100,                  # ditto; the .cbc sets the laser power
        stim_channel_name=site.FRAP_CHANNEL,
        stim_channel_group=site.CHANNEL_GROUP,
        stim_channel_device_name="iLas2",
        stim_channel_power_property_name="BluePower",
    )
]

if MIX_STIM_EXPOSURE:
    # The original shuffles the exposure list. Kept for parity, though on this
    # rig the exposure is not what sets the dose.
    exposures = list(stim_phase[0].stim_exposure_list or ())
    random.shuffle(exposures)
    print(f"shuffled exposures: {exposures[:8]}...")

utils.print_stim_exposures_timesteps(stim_phase)

# %% [markdown]
# ## 7. GUI

# %%
viewer, mm_wdg = site.open_napari(mic, title="22 line stimulation (Inscoper)")

# %% [markdown]
# ## 8. Build the acquisition
#
# Pick the FOV positions in the MDA widget, or load them from a `fovs.json`
# saved earlier — `utils.generate_fov_positions(mic, filename=...)` reads the
# same format the napari-MM MDA widget writes.

# %%
USE_GUI_FOVS = True
FOVS_JSON = None            # e.g. os.path.join(path, "fovs.json")

if FOVS_JSON:
    fovs = utils.generate_fov_positions(mic, filename=FOVS_JSON)
elif USE_GUI_FOVS:
    fovs = utils.generate_fov_positions(mic, viewer=viewer)
else:
    fovs = site.fovs_here(mic, 1)
print(f"{len(fovs)} FOV(s)")
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

pd.set_option("display.max_columns", None)
print(f"total experiment time: {df_acquire['time'].max() / 3600:.2f} h")
df_acquire.head()

# %% [markdown]
# ## 9. Run

# %%
from faro.core.conversion import df_to_events
from faro.widgets import ExperimentStatusWidget

for _ in range(int(SLEEP_BEFORE_EXPERIMENT_START_in_H * 3600)):
    time.sleep(1)

events = df_to_events(df_acquire)
ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

viewer.window.add_dock_widget(
    ExperimentStatusWidget(ctrl), name="experiment status", area="right"
)
handle = ctrl.run_experiment(events, stim_mode="current")
print(f"running {len(events)} events")

# %%
handle.status()

# %% [markdown]
# ## 10. Finish
#
# Two things about the output are specific to a pipeline with **no segmentator**,
# and both are easy to misread as "the stimulation did not happen":
#
# * **`tracks/` is empty**, so `utils.generate_exp_data_from_tracks(path)` raises
#   `ValueError: No objects to concatenate`. There is nothing to merge — this
#   experiment measures a projected pattern, not cells. It is not called here.
# * **`stim_mask/` is not written.** Masks are stored by the pipeline's analysis
#   worker, which only runs when there is something to segment. With
#   `segmentators=None` that worker never runs, so no mask ever reaches disk —
#   even though the mask *was* computed and fired. `stim/` holds the camera
#   frames captured during each burn, and those are the record of what happened.

# %%
import glob

import tifffile

handle.wait()
ctrl.finish_experiment()

for folder in ("raw", "stim", "stim_mask", "labels", "tracks"):
    files = glob.glob(os.path.join(path, folder, "*"))
    print(f"{folder:10s}: {len(files)} file(s)")

stim_frames = sorted(glob.glob(os.path.join(path, "stim", "*.tif*")))
print()
print(f"{len(stim_frames)} stim readout frame(s) — one per fired burn")
assert stim_frames, (
    "no stim readout frames, so no burn was dispatched. Check that "
    "df_acquire['stim'] is True on the stim timesteps and that the FRAP "
    "preflight was clear."
)

# %% [markdown]
# ## 11. Check the stripe actually swept
#
# The point of the experiment is that the pattern *moved*, and there are two
# independent ways to see it here.
#
# **Offline, from the stimulator.** `StimLine` is a pure function of the
# timestep, so the intended sweep can be recomputed exactly — no hardware, no
# sample. If this is flat, the stimulator is misconfigured.
#
# **From the camera, out of `stim/`.** Each of those frames was exposed while the
# galvo was scanning, so a bright band should track the intended stripe. This is
# the one that proves *light landed where the mask said* — the offline curve
# cannot, and neither can `stim_mask/` even when it is written. It needs a
# fluorescent sample to show anything, so treat a flat measured curve as
# inconclusive rather than as a failure.

# %%
import matplotlib.pyplot as plt

intended = []
for t in range(N_FRAMES):
    m = stripe_at(t) > 0
    intended.append(np.nan if not m.any() else float(np.argwhere(m)[:, 1].mean()))

measured, measured_t = [], []
for f in stim_frames:
    frame = np.atleast_2d(np.squeeze(tifffile.imread(f))).astype(float)
    if frame.ndim > 2:                            # a stack: average the planes
        frame = frame.reshape(-1, *frame.shape[-2:]).mean(axis=0)
    profile = frame.mean(axis=0)                  # collapse rows -> x profile
    # Centre of mass of the brightest part of the profile, above the median.
    weight = np.clip(profile - np.median(profile), 0, None)
    measured_t.append(int(os.path.basename(f).split("_")[-1].split(".")[0]))
    measured.append(
        np.nan if weight.sum() == 0
        else float((weight * np.arange(weight.size)).sum() / weight.sum())
    )

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(range(N_FRAMES), intended, "o-", ms=4, label="intended (StimLine)")
ax.plot(measured_t, measured, "s--", ms=4, label="measured (stim readout)")
ax.set_xlabel("frame")
ax.set_ylabel("stripe centre, x (px)")
ax.set_title(f"stripe sweep — one loop every {FRAMES_FOR_1_LOOP} frames")
ax.legend(fontsize=8)
plt.tight_layout()

spread = np.nanmax(intended) - np.nanmin(intended)
print(f"intended sweep spans {spread:.0f} px of a {FRAME_PX} px frame")
assert spread > FRAME_PX * 0.1, (
    "the intended stripe barely moved — check FRAMES_FOR_1_LOOP against N_FRAMES"
)

# %% [markdown]
# ## 12. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
