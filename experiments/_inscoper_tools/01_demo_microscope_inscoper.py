# %% [markdown]
# # faro Demo — Inscoper microscope
#
# The Inscoper variant of `demo_microscope.ipynb`. Same six steps, but against
# the real microscope through `InscoperMicroscope` instead of the Micro-Manager
# demo devices.
#
# Everything that describes *this installation* — config folder, channel names,
# camera, focus, FRAP channel — lives in `../inscoper_site.py`, which is the
# `faro` counterpart of `inscoper_useq/scripts/frap_config.py`. Read that file
# once; the rest of the notebooks in this folder set import it too.
#
# **Four differences from the demo-microscope original**, each of which fails
# quietly rather than loudly if you get it wrong:
#
# 1. **The licence is resolved against the working directory.** `inscoper.lic`
#    must sit next to this notebook. Without it `loadSystemConfiguration()`
#    still returns, but no devices come up: the frame reads 0x0 and every snap
#    is empty. `site.make_microscope` refuses instead.
# 2. **Stage coordinates are absolute.** The original's
#    `stage_positions=[{"x": 0, "y": 0, "z": 0}]` would drive this stage to the
#    machine origin — a full-travel move off your sample. `site.at(mic)` reads
#    the stage and stays there.
# 3. **The environment is not the one `pyproject.toml` describes.**
#    `ome_writers` was declared but missing until it was installed on
#    2026-08-26; `cellpose` still is. That matters because `OmeZarrWriter` does
#    not fail at construction — it fails inside `init_stream`, *after* the run
#    has already started, as the run's `fatal_error`. `site.make_writer()` and
#    `site.make_segmentator()` decide up front and print what they chose.
# 4. **Frames arrive as `int64`** and `getPixelSizeUm()` answers `0.0`, so
#    anything deriving a field of view from the pixel size gets zero.

# %% [markdown]
# ## 1. Connect to the microscope

# %%
import os
import sys

# inscoper_site.py lives one level up, next to the experiment folders.
sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

# A 512x512 centred crop. Not required for imaging, but it keeps the frame
# small enough that the 1 s time plan below is actually achievable, and it is
# the crop the stimulation notebooks need (see site.FULL_FOV_NOTE).
mic = site.make_microscope(roi=512, frap_channel=None)

# %%
import faro.core.utils as utils

# The Inscoper configuration exposes exactly one config group, "Channel", and
# its members are the .cbc files in site.CHANNELS_DIR. Note that none of the
# pertzlab channel names the original notebooks use ("DAPI", "miRFP",
# "mScarlet3", "phase-contrast") exist here — that is why every *_inscoper
# notebook re-maps its channels onto the list this prints.
utils.print_configs(mic.mmc)

# %% [markdown]
# ## 2. Set up the processing pipeline

# %%
from faro.core.data_structures import Channel, SegmentationMethod
from faro.segmentation.base import OtsuSegmentator
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.simple import SimpleFE
from faro.core.pipeline import ImageProcessingPipeline

path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps", "01_demo_inscoper")

segmentators = [
    SegmentationMethod(
        name="labels",
        segmentation_class=OtsuSegmentator(),
        use_channel=0,
        save_tracked=True,
    )
]

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=SimpleFE("labels"),
    tracker=TrackerTrackpy(),
)
print(f"storage: {path}")

# %% [markdown]
# ## 3. Preview segmentation
#
# Snap one frame through a real channel and check that Otsu finds something. If
# the labels image is empty the field is blank or out of focus, and there is no
# point running the time-lapse: `site.here(mic)` reports the focus the drive is
# at, and `site.FOCUS_UM` is the value recorded from the GUI for the 20x.

# %%
import matplotlib.pyplot as plt

CHANNEL = "470 WF"   # site.IMAGING_CHANNELS lists the alternatives
EXPOSURE_MS = 30

mic.mmc.setConfig(site.CHANNEL_GROUP, CHANNEL)
mic.mmc.setExposure(EXPOSURE_MS)
mic.mmc.snapImage()
test_img = mic.mmc.getImage()

seg = OtsuSegmentator()
labels = seg.segment(test_img)

print(f"frame  : {test_img.shape} {test_img.dtype}")
print(f"signal : min {test_img.min()}  mean {test_img.mean():.1f}  max {test_img.max()}")
print(f"labels : {labels.max()} object(s)")
if labels.max() == 0:
    print(
        "No objects. Check focus (site.here(mic) vs site.FOCUS_UM = "
        f"{site.FOCUS_UM}) and that the stage is on a field with cells "
        f"(site.GOOD_XY_UM = {site.GOOD_XY_UM}; site.goto_good_field(mic) "
        "drives there)."
    )

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(test_img, cmap="gray")
axes[0].set_title(f"Raw image — {CHANNEL}")
axes[1].imshow(labels, cmap="nipy_spectral")
axes[1].set_title("Segmentation")
for ax in axes:
    ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ## 4. Define the experiment with RTMSequence
#
# `site.at(mic)` is the whole difference from the original here. The demo
# notebook writes `{"x": 0.0, "y": 0.0, "z": 0.0}`, which on a demo device
# means nothing and on this one means *drive to the machine origin* — a
# millimetres-scale move away from the sample, and on the focus axis a move
# towards the objective. Positions are always expressed relative to where the
# stage already is.

# %%
from faro.core.data_structures import RTMSequence

N_FRAMES = 10
INTERVAL_S = 1.0

seq = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": N_FRAMES},
    stage_positions=[site.at(mic)],
    channels=[Channel(config=CHANNEL, exposure=EXPOSURE_MS, group=site.CHANNEL_GROUP)],
    rtm_metadata={"phase_name": "demo", "phase_id": 0, "treatment_name": "none"},
)

events = list(seq)
print(f"{len(events)} events")
utils.events_to_dataframe(events).head()

# %% [markdown]
# ## 5. Run the experiment

# %%
from faro.core.controller import Controller

writer = site.make_writer(path)
ctrl = Controller(mic, pipeline, writer=writer)

# Checks the channel exists in a config group and the exposure is inside the
# camera's limits (0.001–999999 ms on this camera). It cannot range-check a
# PowerChannel's power: this bridge does not enumerate device properties.
assert ctrl.validate_events(events), "validation reported problems — see warnings above"

ctrl.run_experiment(events).wait()
ctrl.finish_experiment()
print("run complete")

# %% [markdown]
# ## 6. Inspect results

# %%
import pandas as pd

utils.generate_exp_data_from_tracks(path)
tracks = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(
    f"Tracked {tracks['particle'].nunique()} cells across "
    f"{tracks['timestep'].nunique()} frames"
)
tracks.head()

# %%
# The original reads an OME-Zarr store here. With TiffWriter the same frames
# are one TIFF per (fov, timestep) under raw/ and labels/, so read whichever
# the writer actually produced rather than assuming.
import glob

import numpy as np
import tifffile

if site.has_ome_writers():
    import zarr

    store = zarr.open(os.path.join(path, "acquisition.ome.zarr"), mode="r")
    raw_first, raw_last = store["0"][0], store["0"][N_FRAMES - 1]
    lbl_first = store["labels/labels/0"][0]
else:
    raw_files = sorted(glob.glob(os.path.join(path, "raw", "*.tif*")))
    lbl_files = sorted(glob.glob(os.path.join(path, "labels", "*.tif*")))
    raw_first = tifffile.imread(raw_files[0])
    raw_last = tifffile.imread(raw_files[-1])
    lbl_first = tifffile.imread(lbl_files[0])
    print(f"{len(raw_files)} raw frames, {len(lbl_files)} label frames")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, img, title, cmap in (
    (axes[0], raw_first, "Raw (t=0)", "gray"),
    (axes[1], raw_last, f"Raw (t={N_FRAMES - 1})", "gray"),
    (axes[2], lbl_first, "Labels (t=0)", "nipy_spectral"),
):
    ax.imshow(np.squeeze(img), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ## 7. Release the hardware
#
# `post_experiment()` waits for the acquisition to drain and then closes the
# bridge. It is a full teardown on this adapter, so a second run in the same
# kernel needs a fresh `site.make_microscope()` — and the ROI goes back to the
# full frame first, so the next user does not inherit a 512 px crop.

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
