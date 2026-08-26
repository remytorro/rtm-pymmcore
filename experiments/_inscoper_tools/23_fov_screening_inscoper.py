# %% [markdown]
# # FOV screening — pick positions before a stim run, Inscoper microscope
#
# The Inscoper variant of `fov_screening.ipynb`. Interpolates a grid of FOVs
# across a region you bound by hand, acquires one frame at each in both imaging
# channels, runs the same segmentation and feature extraction as the stim
# notebooks, then filters cells by area and per-channel expression and exports
# the FOV positions that still contain a match — ready to paste into
# `23_random_stim_per_cell_14px_patches_inscoper.ipynb`.
#
# Site configuration lives in `../inscoper_site.py`.
#
# **No stimulation happens here**, so this is the one notebook in the set that
# needs no FRAP preflight and no scan-budget arithmetic. It does, however, need
# the two things that bite hardest on this microscope:
#
# * **the corner positions must be real stage coordinates.** The original reads
#   a `corner-positions.json` saved from the napari-MM MDA widget. If you have
#   no such file, `site.here(mic)` plus a span is the safe way to make one — a
#   grid built around `(0, 0)` would sweep the machine origin.
# * **`time_per_fov` must be honest.** A 100-FOV screen at the native
#   2304x2304 frame is a lot of readout; the crop below is what makes it quick.

# %% [markdown]
# ## 1. Microscope

# %%
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from faro.core.data_structures import (
    Channel,
    PowerChannel,
    RTMSequence,
    SegmentationMethod,
    combine,
)
import faro.core.utils as utils

# %%
# Imaging only — no stim channel to designate, so frap_channel=None.
mic = site.make_microscope(roi=512, frap_channel=None)

# %% [markdown]
# ## 2. Configuration
#
# `GRID_N` squared is the FOV count, so it grows fast: 10 -> 100, 25 -> 625.
# Start small and widen once the timing is known.

# %%
GRID_N = 5                      # GRID_N x GRID_N FOVs
TIME_PER_FOV_S = 2.0

## Where the corners come from. Point CORNER_JSON_PATH at a file saved from the
## napari-MM MDA widget, or leave it None to build a grid centred on wherever
## the stage is now.
CORNER_JSON_PATH = None
SPAN_UM = 400.0                 # used only when CORNER_JSON_PATH is None

base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "23_fov_screening_inscoper"
path = os.path.join(base_path, experiment_name)

# The same two channels the stim notebooks image with. Channel 0 is segmented.
imaging_channels = (
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),
)

print(f"output : {path}")
print(f"corners: {CORNER_JSON_PATH or 'from the current stage position'}")

# %% [markdown]
# ## 3. Build the FOV grid
#
# An axis-aligned grid across the bounding box of the corners. Rotation in the
# hand-set region is ignored — fine for a first-pass screen.
#
# The `else` branch is the Inscoper-specific part: a grid built from
# `site.here(mic)` is centred on the field the operator focused on, whereas the
# original's fallback of "no corners" has no meaning and a literal `(0, 0)`
# would drive to the machine origin.

# %%
if CORNER_JSON_PATH:
    with open(CORNER_JSON_PATH, "r") as fh:
        corners = json.load(fh)
    xs = [c["x"] for c in corners]
    ys = [c["y"] for c in corners]
    z0 = corners[0].get("z")
else:
    cx, cy, z0 = site.here(mic)
    xs = [cx - SPAN_UM / 2, cx + SPAN_UM / 2]
    ys = [cy - SPAN_UM / 2, cy + SPAN_UM / 2]
    print(f"grid centred on the current field: x={cx:.0f} y={cy:.0f} z={z0:.0f} um")

x_grid = np.linspace(min(xs), max(xs), GRID_N)
y_grid = np.linspace(min(ys), max(ys), GRID_N)

# Row-major: FOV 0 is (min y, min x), FOV GRID_N starts the second row.
# z is carried explicitly rather than left as None — this stage has no PFS
# holding focus, so a position without z acquires wherever the drive happens
# to be.
fov_positions = [
    utils.FovPosition(x=float(x), y=float(y), z=float(z0), name=f"r{j:02d}c{i:02d}")
    for j, y in enumerate(y_grid)
    for i, x in enumerate(x_grid)
]

print(f"{len(fov_positions)} FOVs covering "
      f"x=[{min(xs):.0f}..{max(xs):.0f}], y=[{min(ys):.0f}..{max(ys):.0f}] um")

# %% [markdown]
# ## 4. Pipeline
#
# Same components as the stim notebooks, minus the stimulator and the ref
# channel: this is a one-shot screen. The tracker runs on a single frame, so it
# just initialises and exits.

# %%
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.erk_ktr import FE_ErkKtr
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

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=FE_ErkKtr("labels"),
    tracker=TrackerTrackpy(search_range=30),
    stimulator=None,
    feature_extractor_ref=None,
)

writer = site.make_writer(path)

# %% [markdown]
# ## 5. Build and run
#
# A single timepoint, two channels, no stim, no ref. `time_per_fov` is set
# generously so the whole grid falls in one batch; `apply_fov_batching` handles
# overflow correctly either way.

# %%
phase = RTMSequence(
    time_plan={"interval": 1, "loops": 1},        # single timepoint
    stage_positions=fov_positions,
    channels=imaging_channels,
    stim_channels=(),
    stim_frames=[],
    ref_channels=(),
    ref_frames=[],
    rtm_metadata={
        "phase_name": "FovScreening",
        "phase_id": 0,
        "treatment_name": "FOV screening pre-flight",
    },
)

events = combine(phase, axis="t")
events = utils.apply_fov_batching(events, time_per_fov=TIME_PER_FOV_S)
print(f"{len(events)} events / {len(fov_positions)} FOVs")

ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

ctrl.run_experiment(events, stim_mode="current").wait()
ctrl.finish_experiment()

utils.generate_exp_data_from_tracks(path)
exp_data_path = os.path.join(path, "exp_data.parquet")
df_exp = pd.read_parquet(exp_data_path)
print(f"acquired {df_exp['fov'].nunique()} FOVs, {len(df_exp)} cell rows")
df_exp.head()

# %% [markdown]
# ## 6. Overview: the grid, both channels
#
# The original reads a `(t, p, c, y, x)` OME-Zarr array. With `TiffWriter` the
# same frames are one TIFF per `(fov, timestep)` under `raw/`, with the channels
# stacked on the first axis — so the collage is built from whichever the writer
# actually produced.

# %%
import glob

import tifffile

THUMB = 64
raw_files = sorted(glob.glob(os.path.join(path, "raw", "*.tif*")))
print(f"{len(raw_files)} raw files")


def frame_for(fov_index, channel):
    """One channel of one FOV's frame, from zarr or from TIFF."""
    if site.has_ome_writers():
        import zarr

        arr = zarr.open(os.path.join(path, "acquisition.ome.zarr"), mode="r")["0"]
        return np.asarray(arr[0, fov_index, channel])
    stack = np.squeeze(tifffile.imread(raw_files[fov_index]))
    return stack[channel] if stack.ndim == 3 else stack


n_show = min(len(imaging_channels), 2)
fig, axs = plt.subplots(1, n_show, figsize=(7 * n_show, 7), dpi=110)
axs = np.atleast_1d(axs)

for ch in range(n_show):
    collage = np.zeros((GRID_N * THUMB, GRID_N * THUMB), dtype=np.float32)
    for p in range(min(len(raw_files), GRID_N * GRID_N)):
        j, i = divmod(p, GRID_N)
        frame = frame_for(p, ch)
        sy = max(1, frame.shape[0] // THUMB)
        sx = max(1, frame.shape[1] // THUMB)
        thumb = frame[::sy, ::sx][:THUMB, :THUMB]
        ty, tx = thumb.shape
        collage[j * THUMB : j * THUMB + ty, i * THUMB : i * THUMB + tx] = thumb
    axs[ch].imshow(collage, cmap="gray_r")
    axs[ch].set_title(imaging_channels[ch].config)
    axs[ch].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Filter cells by area and expression
#
# `df_exp` columns of interest, as `FE_ErkKtr` names them:
#
# * `area_nuc` — cell area in pixels (`FE_ErkKtr` renames skimage's `area`)
# * `mean_intensity_C0_nuc` — nuclear mean in channel 0
# * `mean_intensity_C1_nuc` — nuclear mean in channel 1
#
# The positional columns `x` / `y` carry image **row** and **column** indices
# respectively — a legacy rename of skimage's `centroid-0` / `centroid-1` — which
# is why the crop below indexes rows with `x`.
#
# The thresholds are sample-specific and the originals' values were tuned for a
# different scope, camera and magnification. This cell prints the actual
# distributions first so the numbers are picked from data rather than inherited.

# %%
for col in ("area_nuc", "mean_intensity_C0_nuc", "mean_intensity_C1_nuc"):
    if col in df_exp.columns:
        q = df_exp[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
        print(f"{col:26s} min {q[0]:9.1f}  q25 {q[0.25]:9.1f}  "
              f"med {q[0.5]:9.1f}  q75 {q[0.75]:9.1f}  max {q[1.0]:9.1f}")
    else:
        print(f"{col:26s} MISSING — columns are: {sorted(df_exp.columns)[:12]}...")

# %%
# --- Filter ranges (edit from the quantiles above and re-run) ---
AREA_MIN, AREA_MAX = float(df_exp["area_nuc"].quantile(0.5)), float(df_exp["area_nuc"].max())
C0_MEAN_MIN, C0_MEAN_MAX = float(df_exp["mean_intensity_C0_nuc"].quantile(0.5)), np.inf
C1_MEAN_MIN, C1_MEAN_MAX = float(df_exp["mean_intensity_C1_nuc"].quantile(0.5)), np.inf

mask = (
    df_exp["area_nuc"].between(AREA_MIN, AREA_MAX)
    & df_exp["mean_intensity_C0_nuc"].between(C0_MEAN_MIN, C0_MEAN_MAX)
    & df_exp["mean_intensity_C1_nuc"].between(C1_MEAN_MIN, C1_MEAN_MAX)
)
df_match = df_exp[mask].copy()
print(f"matching cells: {len(df_match)}  /  FOVs hit: {df_match['fov'].nunique()}"
      f" of {df_exp['fov'].nunique()}")

# --- Crop preview ---
CROP = 96
N_PREVIEW = min(24, len(df_match))
if N_PREVIEW:
    cols = 6
    rows = int(np.ceil(N_PREVIEW / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), dpi=110)
    axs = np.atleast_1d(axs).reshape(rows, cols)
    sample = df_match.sample(N_PREVIEW, random_state=0).reset_index(drop=True)
    for k, row in sample.iterrows():
        r, c = divmod(k, cols)
        frame = frame_for(int(row["fov"]), 0)
        # df["x"] = image row, df["y"] = image col (pipeline rename quirk).
        row_idx, col_idx = int(row["x"]), int(row["y"])
        y0, y1 = max(0, row_idx - CROP // 2), min(frame.shape[0], row_idx + CROP // 2)
        x0, x1 = max(0, col_idx - CROP // 2), min(frame.shape[1], col_idx + CROP // 2)
        axs[r, c].imshow(frame[y0:y1, x0:x1], cmap="gray_r")
        axs[r, c].set_title(f"fov{int(row['fov'])} L{int(row['label'])}", fontsize=7)
        axs[r, c].axis("off")
    for k in range(N_PREVIEW, rows * cols):
        r, c = divmod(k, cols)
        axs[r, c].axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("(no preview — loosen the filters)")

# %% [markdown]
# ## 8. Save the filtered FOV positions
#
# FOVs with at least one matching cell are written as `filtered_positions.json`
# in the same `x` / `y` / `z` / `name` shape the napari-MM MDA widget uses, so
# they load straight back with
# `utils.generate_fov_positions(mic, filename=...)` — which is how the stim
# notebooks in this folder consume them.

# %%
hit_fov_idx = sorted(int(p) for p in df_match["fov"].unique())
hit_positions = [
    {
        "x": fov_positions[p].x,
        "y": fov_positions[p].y,
        "z": fov_positions[p].z,
        "name": fov_positions[p].name,
    }
    for p in hit_fov_idx
    if p < len(fov_positions)
]

out_json = os.path.join(path, "filtered_positions.json")
with open(out_json, "w") as fh:
    json.dump(hit_positions, fh, indent=2)

print(f"kept {len(hit_positions)}/{len(fov_positions)} FOVs")
print(f"wrote {out_json}")
print("\nLoad these in a stim notebook with:")
print(f"    fovs = utils.generate_fov_positions(mic, filename=r\"{out_json}\")")

# %% [markdown]
# ## 9. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
