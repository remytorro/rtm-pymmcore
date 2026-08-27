# %% [markdown]
# # TiffWriter test — Inscoper microscope
#
# The Inscoper variant of `tiff_writer.ipynb`. Exercises the `TiffWriter`
# backend against a real acquisition: raw images, segmentation masks and tracked
# labels, one TIFF per `(fov, timestep)`, plus tracks as parquet.
#
# Site configuration lives in `../inscoper_site.py`.
#
# **Why `TiffWriter` still matters here.** `OmeZarrWriter` needs the
# `ome_writers` package, which `pyproject.toml` declares but the `py313`
# environment did not have until it was installed on 2026-08-26 — see
# `ome_zarr_writer_inscoper.ipynb`, which now runs. The failure mode is the
# reason `TiffWriter` remains the safe default on a fresh machine:
# `OmeZarrWriter` constructs fine without the package and fails *inside*
# `init_stream`, after `run_experiment` has already started, with a bare
# `ModuleNotFoundError` surfacing as the run's `fatal_error`. Nothing is
# acquired, and the traceback names neither the writer nor the package.
# `site.make_writer()` decides up front; this notebook asks for `TiffWriter`
# explicitly.

# %% [markdown]
# ## 1. Connect to the microscope

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

import faro.core.utils as utils

# Imaging only, so no FRAP channel to designate.
mic = site.make_microscope(roi=512, frap_channel=None)
utils.print_configs(mic.mmc)

# %%
mic.mmc.setConfig(site.CHANNEL_GROUP, "470 WF")
mic.mmc.setExposure(30)
mic.mmc.snapImage()
test_img = mic.mmc.getImage()
print(f"Camera: {test_img.shape[1]}x{test_img.shape[0]}, dtype={test_img.dtype}")

# Frames arrive as int64 from this bridge -- 8 bytes per pixel for 16-bit
# camera data. TiffWriter preserves that dtype (the raw files below really are
# int64), so the files are four times larger than they need to be; OmeZarrWriter
# instead casts to its `dtype=` argument, which defaults to uint16 and would
# wrap silently on a value above 65535. Worth checking the range once for a
# given exposure rather than assuming either behaviour.
print(f"Range : {test_img.min()} .. {test_img.max()}  "
      f"({'fits uint16' if test_img.max() <= 65535 else 'WOULD OVERFLOW uint16'})")

# %% [markdown]
# ## 2. Set up the pipeline

# %%
from faro.core.data_structures import Channel, SegmentationMethod
from faro.segmentation.base import OtsuSegmentator
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.simple import SimpleFE
from faro.core.pipeline import ImageProcessingPipeline

base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
path = os.path.join(base_path, "03_tiff_writer_inscoper")

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
    tracker=TrackerTrackpy(search_range=20),
)
print(f"storage: {path}")

# %% [markdown]
# ## 3. Create the writer

# %%
from faro.core.writers import TiffWriter

writer = TiffWriter(storage_path=path)
print(f"writer: {type(writer).__name__}")
print(f"ome_writers available (so OmeZarrWriter usable): {site.has_ome_writers()}")

# %% [markdown]
# ## 4. Define the experiment
#
# Two channels and two positions, so the output exercises both the channel and
# the position axis — a single-channel single-position run cannot show whether
# the naming keeps them apart.
#
# `site.at(mic, dx=…)` rather than the original's `{"x": 0.0, "y": 0.0,
# "z": 0.0}`: on this stage a literal zero is the **machine origin**, a
# full-travel move off the sample.

# %%
from faro.core.data_structures import RTMSequence

N_FRAMES = 6
INTERVAL_S = 4.0
FOV_STEP_UM = 150.0

channels = [
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),
]

seq = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": N_FRAMES},
    stage_positions=[site.at(mic), site.at(mic, dx=FOV_STEP_UM)],
    channels=channels,
    rtm_metadata={"phase_name": "tiff-writer-test", "phase_id": 0,
                  "treatment_name": "none"},
)

events = list(seq)
print(f"{len(events)} events "
      f"({N_FRAMES} timepoints x 2 FOVs x {len(channels)} channels)")
utils.events_to_dataframe(events).head()

# %% [markdown]
# ## 5. Run

# %%
from faro.core.controller import Controller

ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

ctrl.run_experiment(events).wait()
ctrl.finish_experiment()
print("Experiment complete.")

# %% [markdown]
# ## 6. Inspect what was written
#
# `TiffWriter` lays the output out as one folder per data kind, and one file per
# `(fov, timestep)` named `{fov:03d}_{timestep:05d}.tiff`. The **channels are
# stacked inside each raw file**, which is what makes a 2-channel run look like
# the same file count as a 1-channel one.

# %%
import glob

import tifffile

print(f"{'folder':14s} {'files':>6s}  first file")
for folder in sorted(
    d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
):
    files = sorted(glob.glob(os.path.join(path, folder, "*")))
    first = os.path.basename(files[0]) if files else "-"
    print(f"{folder:14s} {len(files):6d}  {first}")

raw_files = sorted(glob.glob(os.path.join(path, "raw", "*.tif*")))
lbl_files = sorted(glob.glob(os.path.join(path, "labels", "*.tif*")))

raw0 = tifffile.imread(raw_files[0])
lbl0 = tifffile.imread(lbl_files[0])
print()
print(f"raw file    : {raw0.shape} {raw0.dtype}   "
      f"(channel axis first when >2D)")
print(f"labels file : {lbl0.shape} {lbl0.dtype}")

expected = N_FRAMES * 2          # timepoints x FOVs; channels are inside the file
print(f"raw files   : {len(raw_files)} (expected {expected})")
assert len(raw_files) == expected, (
    f"{len(raw_files)} raw files for {expected} (fov, timestep) pairs — the "
    "naming scheme or the channel stacking is not what this cell assumes"
)

# %% [markdown]
# ## 7. Tracks

# %%
utils.generate_exp_data_from_tracks(path)
tracks = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"Tracked {tracks['particle'].nunique()} cells across "
      f"{tracks['timestep'].nunique()} timesteps in "
      f"{tracks['fov'].nunique()} FOVs")
tracks.head()

# %% [markdown]
# ## 8. Look at it
#
# First and last raw frame of FOV 0 with its labels, to confirm the files line
# up with the events that asked for them rather than merely existing.

# %%
import matplotlib.pyplot as plt


def channel_of(path_tiff, channel=0):
    arr = np.squeeze(tifffile.imread(path_tiff))
    return arr[channel] if arr.ndim == 3 else arr


fov0_raw = [f for f in raw_files if os.path.basename(f).startswith("000_")]
fov0_lbl = [f for f in lbl_files if os.path.basename(f).startswith("000_")]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, img, title, cmap in (
    (axes[0], channel_of(fov0_raw[0]), f"FOV 0 raw, ch0, t=0", "gray"),
    (axes[1], channel_of(fov0_raw[-1]), f"FOV 0 raw, ch0, t={len(fov0_raw) - 1}", "gray"),
    (axes[2], channel_of(fov0_lbl[0]), "FOV 0 labels, t=0", "nipy_spectral"),
):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
