# %% [markdown]
# # OmeZarrWriter test — Inscoper microscope
#
# The Inscoper variant of `ome_zarr_writer.ipynb`. Streams raw images,
# segmentation masks, tracked labels and stim readouts into a single OME-Zarr
# v0.5 store instead of individual TIFFs.
#
# Site configuration lives in `../inscoper_site.py`.
#
# > **This notebook needs `ome_writers`, which was missing until 2026-08-26.**
# > `pyproject.toml` declares the package, but the `py313` environment did not
# > have it, and the absence did not present as an import error you could act
# > on: `OmeZarrWriter` constructs fine without it and fails inside
# > `init_stream`, i.e. **after `run_experiment` has already started**, with a
# > bare `ModuleNotFoundError` surfacing as the run's `fatal_error`. Nothing is
# > acquired, and the traceback names neither the writer nor the package.
# >
# > It is installed now and this notebook runs. Cell 0 still checks, because the
# > next machine will not have it either — and `tiff_writer_inscoper.ipynb`
# > covers the same ground with no extra dependency.

# %% [markdown]
# ## 0. Is the OME-Zarr path available at all?

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

HAVE_OME_WRITERS = site.has_ome_writers()
print(f"ome_writers importable: {HAVE_OME_WRITERS}")
if not HAVE_OME_WRITERS:
    print()
    print("OmeZarrWriter cannot be used in this environment. Install it with:")
    print("    conda activate py313 && pip install ome-writers")
    print()
    print("Until then this notebook stops at the run cell, and")
    print("03_tiff_writer_inscoper.ipynb covers the same ground with TiffWriter.")

# %% [markdown]
# ## 1. Connect to the microscope

# %%
import numpy as np
import pandas as pd

import faro.core.utils as utils

# A stim channel is designated: this notebook exercises store_stim_images, which
# writes the stim readout as an extra channel in the raw array. That only has
# something to write if stimulation actually fires.
mic = site.make_microscope(roi=256)
utils.print_configs(mic.mmc)

# %%
mic.mmc.setConfig(site.CHANNEL_GROUP, "470 WF")
mic.mmc.setExposure(30)
mic.mmc.snapImage()
test_img = mic.mmc.getImage()
print(f"Camera: {test_img.shape[1]}x{test_img.shape[0]}, dtype={test_img.dtype}")
print(f"Range : {test_img.min()} .. {test_img.max()}")

# OmeZarrWriter stores its `dtype` argument, which defaults to uint16, and
# coerces every incoming frame to it -- clipping, so a value above 65535
# saturates instead of wrapping. The bridge now hands out uint16 directly
# (it used to hand out int64, which the store refused). The range above is
# still worth a look before committing a long run.
assert test_img.max() <= 65535, (
    f"peak {test_img.max()} exceeds uint16 — pass dtype='uint32' to "
    "OmeZarrWriter or shorten the exposure"
)

# %% [markdown]
# ## 2. Pipeline
#
# A stimulator is included so `store_stim_images=True` has something to store.
# `CenterCircle` is the cheapest possible patterned target for the FRAP galvo —
# a filled disc in the middle of the field, well inside the scan budget.

# %%
from faro.core.data_structures import Channel, SegmentationMethod
from faro.segmentation.base import OtsuSegmentator
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.simple import SimpleFE
from faro.stimulation.center_circle import CenterCircle
from faro.core.pipeline import ImageProcessingPipeline

base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
path = os.path.join(base_path, "03_ome_zarr_writer_inscoper")

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
    stimulator=CenterCircle(),
)
print(f"storage: {path}")

# %% [markdown]
# ## 3. FRAP preflight and the burn price

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

plan = site.price_mask(mic, site.sample_disc(mic, radius_fraction=0.125))
assert plan is not None, "the centre disc is over the scan budget — crop further"

# %% [markdown]
# ## 4. Create the writer
#
# The chunking arguments are the point of this notebook, and they are the
# tradeoff every long run makes:
#
# * `raw_chunk_t=1` — one timepoint per chunk, so a viewer can seek to any frame
#   without reading its neighbours. Larger values compress better and stream
#   worse.
# * `label_chunk_t=1` with `label_shard_t=50` — random access per frame, but the
#   chunks are grouped 50 to a shard file, so a 24 h run does not produce a
#   directory with a hundred thousand entries in it. On SMB that difference is
#   the difference between usable and not.
# * `store_stim_images=True` — the stim readout becomes an extra channel in the
#   raw array (zeros on non-stim timepoints) instead of falling back to TIFF, so
#   one store holds the whole experiment.

# %%
if HAVE_OME_WRITERS:
    from faro.core.writers import OmeZarrWriter

    writer = OmeZarrWriter(
        storage_path=path,
        raw_chunk_t=1,
        label_chunk_t=1,
        label_shard_t=50,
        store_stim_images=True,
    )
    print(f"Zarr store: {writer._zarr_path}")
else:
    writer = None
    print("skipped — ome_writers is not installed (see cell 0)")

# %% [markdown]
# ## 5. Define the experiment
#
# Two channels, two positions, a stim block and a ref frame — so the store
# exercises every axis it has: `t`, `p`, `c`, plus the label groups and the stim
# channel.
#
# `site.at(mic, dx=…)` rather than the original's literal `(0, 0, 0)` and
# `(1, 0, 0)`: on this stage zero is the machine origin, and a 1 µm step is not
# a different field.

# %%
from faro.core.data_structures import RTMSequence

N_FRAMES = 6
INTERVAL_S = 6.0
FOV_STEP_UM = 150.0
STIM_FRAMES = [2, 3]

channels = [
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),
]
stim_channel = Channel(config=site.FRAP_CHANNEL, exposure=100,
                       group=site.CHANNEL_GROUP)
ref_channel = Channel(config="640 WF", exposure=60, group=site.CHANNEL_GROUP)

seq = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": N_FRAMES},
    stage_positions=[site.at(mic), site.at(mic, dx=FOV_STEP_UM)],
    channels=channels,
    stim_channels=[stim_channel],
    stim_frames=STIM_FRAMES,
    ref_channels=[ref_channel],
    ref_frames=[-1],
    rtm_metadata={"phase_name": "ome-zarr-writer-test", "phase_id": 0,
                  "treatment_name": "centre-disc-stim"},
)

events = list(seq)
print(f"{len(events)} events")
utils.events_to_dataframe(events).head()

# %% [markdown]
# ## 6. Run

# %%
from faro.core.controller import Controller

if writer is None:
    raise RuntimeError(
        "ome_writers is not installed, so OmeZarrWriter cannot be built. "
        "Install it (pip install ome-writers) and re-run from cell 0, or use "
        "03_tiff_writer_inscoper.ipynb."
    )

ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

ctrl.run_experiment(events, stim_mode="current").wait()
ctrl.finish_experiment()
print("Experiment complete.")

# %% [markdown]
# ## 7. Inspect the store

# %%
import zarr

store = zarr.open(writer._zarr_path, mode="r")
print("Zarr store tree:")
print(store.tree())

# %%
raw = store["0"]
print(f"Raw array: shape={raw.shape}, dtype={raw.dtype}, chunks={raw.chunks}")
print(f"  axes are (t, p, c, y, x); c includes the stim readout channel when "
      f"store_stim_images=True")
print("Root attributes:")
for key, value in store.attrs.asdict().items():
    print(f"  {key}: {value}")

# %%
labels_grp = store["labels"]
names = list(labels_grp.attrs.get("labels", []))
print(f"Label groups: {names}")
for name in names:
    arr = store[f"labels/{name}/0"]
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}")

# %% [markdown]
# ## 8. Tracks
#
# Tracking data stays parquet, outside the zarr store.

# %%
utils.generate_exp_data_from_tracks(path)
tracks = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"Tracked {tracks['particle'].nunique()} cells across "
      f"{tracks['timestep'].nunique()} timesteps in {tracks['fov'].nunique()} FOVs")
tracks.head()

# %% [markdown]
# ## 9. Validate the NGFF metadata
#
# A store that opens is not the same as a store a viewer will read: the
# `multiscales` block on the image group and the `image-label` block on each
# label group are what napari-ome-zarr and the Inscoper viewer key on.

# %%
print("Image group attributes:")
for key, val in store["0"].attrs.items():
    print(f"  {key}: {val}")

for name in names:
    print(f"Label '{name}' attributes:")
    for key, val in store[f"labels/{name}"].attrs.items():
        print(f"  {key}: {val}")

# %% [markdown]
# ## 10. Look at it
#
# First and last raw frame of FOV 0, its labels, and the stim mask that fired —
# which is the one panel that proves the stim channel really was stored rather
# than merely allocated.

# %%
import matplotlib.pyplot as plt

lbl = store["labels/labels/0"]
stim = store["labels/stim_mask/0"] if "stim_mask" in names else None

panels = [
    (np.asarray(raw[0, 0, 0]), "Raw ch0 (t=0, FOV 0)", "gray"),
    (np.asarray(raw[N_FRAMES - 1, 0, 0]), f"Raw ch0 (t={N_FRAMES - 1})", "gray"),
    (np.asarray(lbl[0, 0]), "Labels (t=0)", "nipy_spectral"),
]
if stim is not None:
    panels.append(
        (np.asarray(stim[STIM_FRAMES[0], 0]), f"Stim mask (t={STIM_FRAMES[0]})", "Reds")
    )

fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4))
for ax, (img, title, cmap) in zip(np.atleast_1d(axes), panels):
    ax.imshow(np.squeeze(img), cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

if stim is not None:
    fired = int((np.asarray(stim[STIM_FRAMES[0], 0]) > 0).sum())
    print(f"stim mask at t={STIM_FRAMES[0]}: {fired} px lit")
    assert fired > 0, (
        "the stored stim mask is empty, so either nothing fired or "
        "store_stim_images did not capture it"
    )

# %% [markdown]
# ## 11. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
