# %% [markdown]
# # OME-Zarr writer, plate layout — Inscoper microscope
#
# The Inscoper variant of `ome_zarr_writer_plate.ipynb`. Compare with
# `03_ome_zarr_writer_inscoper.ipynb`, which uses `OmeZarrWriter`.
#
# `OmeZarrWriterPlate` stores each FOV as a **well** in an OME-Zarr plate, so
# napari-ome-zarr tiles the positions spatially as a mosaic instead of putting
# them behind a slider.
#
# | | `OmeZarrWriter` | `OmeZarrWriterPlate` |
# |--|--|--|
# | Positions in napari | **slider** (`p` dimension) | **spatial mosaic** |
# | Array layout | one 5D `(t, p, c, y, x)` | per-well `(t, c, y, x)` |
# | Labels in napari | yes | **no** (reader limitation) |
#
# > **Labels are still written.** Segmentation masks, tracked labels and stim
# > masks go into each well's zarr group and load fine programmatically — the
# > cells below read them. `napari-ome-zarr`'s plate/well reader simply does not
# > descend into wells to find label sub-groups, so it shows only the raw
# > mosaic. That is a limitation of `ome-zarr-py`'s `Well` spec, not of the data.
#
# Site configuration lives in `../inscoper_site.py`.

# %% [markdown]
# ## 0. Is the OME-Zarr path available?
#
# `OmeZarrWriterPlate` inherits from `OmeZarrWriter` and needs the same
# `ome_writers` package — installed here on 2026-08-26 — and fails the same way
# without it: not at construction, but inside `init_stream`, *after*
# `run_experiment` has started, as the run's `fatal_error`.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

HAVE_OME_WRITERS = site.has_ome_writers()
print(f"ome_writers importable: {HAVE_OME_WRITERS}")
if not HAVE_OME_WRITERS:
    print()
    print("Install with:  conda activate py313 && pip install ome-writers")
    print("Or use 03_tiff_writer_inscoper.ipynb, which needs nothing extra.")

# %% [markdown]
# ## 1. Connect to the microscope
#
# The plate layout only means anything with several positions, so this notebook
# uses four — stepped by a real distance rather than the original's 1 µm, which
# on this stage would be four copies of the same field.

# %%
import numpy as np
import pandas as pd

import faro.core.utils as utils

mic = site.make_microscope(roi=256, frap_channel=None)
utils.print_configs(mic.mmc)

# %%
mic.mmc.setConfig(site.CHANNEL_GROUP, "470 WF")
mic.mmc.setExposure(30)
mic.mmc.snapImage()
test_img = mic.mmc.getImage()
print(f"Camera: {test_img.shape[1]}x{test_img.shape[0]}, dtype={test_img.dtype}")
print(f"Range : {test_img.min()} .. {test_img.max()}")
assert test_img.max() <= 65535, (
    f"peak {test_img.max()} exceeds uint16 — pass dtype='uint32' to the writer"
)

# %% [markdown]
# ## 2. Pipeline

# %%
from faro.core.data_structures import Channel, SegmentationMethod
from faro.segmentation.base import OtsuSegmentator
from faro.tracking.trackpy import TrackerTrackpy
from faro.feature_extraction.simple import SimpleFE
from faro.core.pipeline import ImageProcessingPipeline

base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
path = os.path.join(base_path, "03_ome_zarr_writer_plate_inscoper")

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
# ## 3. Create the plate writer
#
# One well per position, in a single-row plate: `A/1`, `A/2`, `A/3`, ... The
# `overwrite=True` default matters here — a plate whose well count does not match
# the events would otherwise be reused with the wrong layout.

# %%
if HAVE_OME_WRITERS:
    from faro.core.writers import OmeZarrWriterPlate

    writer = OmeZarrWriterPlate(
        storage_path=path,
        raw_chunk_t=1,
        label_chunk_t=1,
        label_shard_t=50,
    )
    print(f"Zarr store: {writer._zarr_path}")
else:
    writer = None
    print("skipped — ome_writers is not installed (see cell 0)")

# %% [markdown]
# ## 4. Define the experiment
#
# Four positions, two channels. `site.at(mic, dx=…, dy=…)` lays them out as a
# 2x2 block around the current field, so the mosaic in napari corresponds to
# something real on the sample.

# %%
from faro.core.data_structures import RTMSequence

N_FRAMES = 4
INTERVAL_S = 5.0
FOV_STEP_UM = 150.0

channels = [
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),
]

positions = [
    site.at(mic, dx=0.0, dy=0.0),
    site.at(mic, dx=FOV_STEP_UM, dy=0.0),
    site.at(mic, dx=0.0, dy=FOV_STEP_UM),
    site.at(mic, dx=FOV_STEP_UM, dy=FOV_STEP_UM),
]

seq = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": N_FRAMES},
    stage_positions=positions,
    channels=channels,
    rtm_metadata={"phase_name": "plate-writer-test", "phase_id": 0,
                  "treatment_name": "none"},
)

events = list(seq)
print(f"{len(events)} events over {len(positions)} positions "
      f"-> {len(positions)} wells")
utils.events_to_dataframe(events).head()

# %% [markdown]
# ## 5. Run

# %%
from faro.core.controller import Controller

if writer is None:
    raise RuntimeError(
        "ome_writers is not installed, so OmeZarrWriterPlate cannot be built. "
        "Install it (pip install ome-writers) and re-run from cell 0."
    )

ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

ctrl.run_experiment(events).wait()
ctrl.finish_experiment()
print("Experiment complete.")

# %% [markdown]
# ## 6. Inspect the plate
#
# The structure is the difference from `OmeZarrWriter`: instead of one 5D array,
# there is a plate group whose wells each hold their own `(t, c, y, x)` image and
# their own labels.

# %%
import zarr

store = zarr.open(writer._zarr_path, mode="r")
print("Store tree:")
print(store.tree())

print()
print("Plate metadata:")
for key, value in store.attrs.asdict().items():
    print(f"  {key}: {value}")

# %%
# Walk the wells and report what each holds. The layout is
# <row>/<column>/<field>/0 for the raw array, with labels alongside.
plate = store.attrs.asdict().get("plate") or store.attrs.asdict().get("ome", {}).get("plate", {})
wells = plate.get("wells", []) if isinstance(plate, dict) else []
print(f"{len(wells)} well(s) declared in the plate metadata")

for well in wells:
    wpath = well.get("path")
    grp = store[f"{wpath}/0"]
    raw = grp["0"]
    print(f"  well {wpath}: raw {raw.shape} {raw.dtype} chunks={raw.chunks}")
    # `"labels" in grp` answers False on a zarr v3 group even when the
    # sub-group is there, so ask by indexing and catch the miss instead.
    try:
        names = list(grp["labels"].attrs.get("labels", []))
    except KeyError:
        names = []
    for name in names:
        arr = grp[f"labels/{name}/0"]
        print(f"      labels/{name}: {arr.shape} {arr.dtype}")

assert len(wells) == len(positions), (
    f"{len(wells)} wells for {len(positions)} positions — the plate layout does "
    "not match the events"
)

# %% [markdown]
# ## 7. Labels really are there
#
# The claim from the top of the notebook, checked rather than asserted: napari
# will not show these, but they are in the store and readable.

# %%
first_well = wells[0]["path"]
grp = store[f"{first_well}/0"]
label_names = list(grp["labels"].attrs.get("labels", []))
print(f"well {first_well} label groups: {label_names}")
assert "labels" in label_names, (
    "no segmentation labels in the well group — save_tracked/segmentators did "
    "not reach the writer"
)

lbl = np.asarray(grp["labels/labels/0"][0])
print(f"labels at t=0: {lbl.shape} {lbl.dtype}, {int(lbl.max())} object(s)")

# %% [markdown]
# ## 8. The mosaic, as napari would tile it

# %%
import matplotlib.pyplot as plt

n = len(wells)
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = np.atleast_1d(axes).reshape(rows, cols)

for k, well in enumerate(wells):
    r, c = divmod(k, cols)
    raw = store[f"{well['path']}/0/0"]
    axes[r, c].imshow(np.squeeze(np.asarray(raw[0, 0])), cmap="gray")
    axes[r, c].set_title(f"well {well['path']} (t=0, ch0)", fontsize=9)
    axes[r, c].axis("off")
for k in range(n, rows * cols):
    r, c = divmod(k, cols)
    axes[r, c].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Tracks

# %%
utils.generate_exp_data_from_tracks(path)
tracks = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"Tracked {tracks['particle'].nunique()} cells across "
      f"{tracks['timestep'].nunique()} timesteps in {tracks['fov'].nunique()} FOVs")
tracks.head()

# %% [markdown]
# ## 10. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
