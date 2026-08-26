# %% [markdown]
# # Optogenetic feedback: watershed + trackpy + edge stimulation — Inscoper
#
# The Inscoper variant of `demo_sim_optogenetic.ipynb`. The original runs the
# whole faro pipeline against the `virtual_microscope` optogenetic backend; this
# one runs it against the real microscope, with the FRAP galvo as the
# stimulation device.
#
# * **Segmentation:** Gaussian blur -> Otsu -> distance transform -> watershed,
#   defined in this notebook (the original's `WatershedSegmentator`)
# * **Tracking:** trackpy
# * **Stimulation:** illuminate the leading edge of each cell — either always
#   the top (`StimUp`) or up/down by particle parity (`StimUpDown`)
#
# Site configuration lives in `../inscoper_site.py`.
#
# Two things the original relies on that do not exist here: the
# `virtual_microscope` package (this notebook needs no extra), and the
# `phase-contrast` channel (re-mapped onto `site.IMAGING_CHANNELS`). The
# substantive difference is the third one — see the next cell.

# %% [markdown]
# ## 1. Connect to the microscope
#
# The simulated backend has an SLM in the camera's own pixel grid, so
# `UniMMCoreSimulation` treats the camera->SLM transform as the identity and any
# mask is projectable. Here the stimulation device is a FRAP galvo with a fixed
# scan budget — one fire covers ~36,000 px of scan path (~5 s) — so a mask has a
# **price**, and `plan_mask` refuses one that is too expensive rather than
# truncating it. The stimulator classes below are unchanged from the original;
# what is new is that this notebook prices their output before running.

# %%
import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import faro.core.utils as utils

# 512 px: enough cells for tracking to be interesting, small enough that a
# leading-edge mask stays inside the scan budget.
mic = site.make_microscope(roi=512)

# The original calls utils.print_configs(core) here. Same thing:
utils.print_configs(mic.mmc)

# %% [markdown]
# ## 2. Segmentation: watershed
#
# A real experiment would use cellpose or stardist. This is the original's
# dependency-free thresholding approach, kept verbatim — it is fast, which
# matters when the pipeline has to finish inside one interval.

# %%
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy import ndimage
from faro.segmentation.base import Segmentator


class WatershedSegmentator(Segmentator):
    """Gaussian blur -> Otsu threshold -> distance transform -> watershed."""

    def __init__(self, sigma=3, min_distance=20):
        self.sigma = sigma
        self.min_distance = min_distance

    def segment(self, image: np.ndarray) -> np.ndarray:
        blurred = gaussian(image, sigma=self.sigma)
        thresh = threshold_otsu(blurred)
        binary = blurred > thresh
        distance = ndimage.distance_transform_edt(binary)
        coords = peak_local_max(distance, min_distance=self.min_distance, labels=binary)
        markers = np.zeros(distance.shape, dtype=bool)
        markers[tuple(coords.T)] = True
        markers = label(markers)
        return watershed(-distance, markers, mask=binary)


segmentator = WatershedSegmentator()

# %%
from faro.core.data_structures import SegmentationMethod

segmentators = [
    SegmentationMethod(
        name="labels",
        segmentation_class=segmentator,
        use_channel=0,
        save_tracked=True,
    )
]

# %% [markdown]
# ## 3. Tracking: trackpy
#
# The parameter to tune is the search range: the furthest a cell's centroid may
# move between frames. It is in **pixels**, so it has to be re-picked whenever
# the frame is cropped or the magnification changes — 15 px on the original's
# simulated frame is not 15 px here.

# %%
from faro.tracking.trackpy import TrackerTrackpy

tracker = TrackerTrackpy(search_range=20)

# %% [markdown]
# ## 4. Feature extraction
#
# Where all the per-cell analysis you want during the run goes. `regionprops` is
# the starting point; a real experiment would compute biosensor activity here
# (an ERK-KTR nuclear/cytosolic ratio, a FRET ratio, ...).

# %%
import skimage.measure
from faro.feature_extraction.base import FeatureExtractor


class SimpleFE(FeatureExtractor):
    def __init__(self, used_mask):
        self.used_mask = used_mask
        super().__init__()

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table = skimage.measure.regionprops_table(
            labels[self.used_mask], properties=["label", "area"]
        )
        return pd.DataFrame.from_dict(table), None


feature_extractor = SimpleFE("labels")

# %% [markdown]
# ## 5. Stimulation A: leading-edge illumination
#
# The `Stim` hierarchy is where the feedback logic lives — given the image, the
# segmentation and the tracks, where should light go? Three base classes:
#
# * **`Stim`** — metadata only (e.g. timestep, img_shape)
# * **`StimWithImage`** — raw image + metadata
# * **`StimWithPipeline`** — labels + metadata + image + tracks
#
# `StimUp` inherits from `StimWithPipeline` and lights the top of each cell's
# y-extent, which is what drives directed migration with an actuator like
# optoTIAM (Rac) or optoFGFR.

# %%
from faro.stimulation.base import StimWithPipeline
from skimage.morphology import disk, dilation


class StimUp(StimWithPipeline):
    """Illuminate the top *fraction* of each cell's y-extent, dilated by disk(3)."""

    def __init__(self, fraction):
        self.fraction = fraction

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)
        selem = disk(3)

        for prop in regionprops(labels):
            minr, minc, maxr, maxc = prop.bbox
            y_cutoff = minr + self.fraction * (maxr - minr)

            cell_mask = labels == prop.label
            rows, cols = np.where(cell_mask)
            top_pixels = rows < y_cutoff
            if not top_pixels.any():
                continue

            local = np.zeros_like(labels, dtype=np.uint8)
            local[rows[top_pixels], cols[top_pixels]] = 1
            local = dilation(local, footprint=selem)
            stim_mask = np.maximum(stim_mask, local)

        return stim_mask, None


stimulator_up = StimUp(fraction=0.2)

# %% [markdown]
# ## 6. Stimulation B: track-aware
#
# Now using the `tracks` DataFrame the pipeline passes in, to steer
# even-numbered particles up and odd-numbered ones down — half the cells should
# migrate each way. This is the trackpy vocabulary:
#
# * `label` — the integer of a cell's mask in this frame's segmentation
# * `particle` — the track id, stable across timepoints
#
# Note it returns an **empty mask when `tracks` is empty**, i.e. on the first
# frame. That is correct behaviour and also a trap for any pre-flight that
# forgets to supply tracks — `site.probe_stim_mask` supplies them.

# %%
class StimUpDown(StimWithPipeline):
    """Steer cells by particle id: even -> up, odd -> down."""

    def __init__(self, fraction=0.2):
        self.fraction = fraction

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)
        selem = disk(3)

        if tracks is None or tracks.empty:
            return stim_mask, None

        current = tracks[tracks["timestep"] == tracks["timestep"].max()]
        label_to_particle = dict(zip(current["label"], current["particle"]))

        for prop in regionprops(labels):
            minr, minc, maxr, maxc = prop.bbox
            pid = label_to_particle.get(prop.label, 0)

            if pid % 2 == 0:
                y_cutoff = minr + self.fraction * (maxr - minr)
                select = lambda rows, c=y_cutoff: rows < c
            else:
                y_cutoff = maxr - self.fraction * (maxr - minr)
                select = lambda rows, c=y_cutoff: rows > c

            cell_mask = labels == prop.label
            rows, cols = np.where(cell_mask)
            edge_pixels = select(rows)
            if not edge_pixels.any():
                continue

            local = np.zeros_like(labels, dtype=np.uint8)
            local[rows[edge_pixels], cols[edge_pixels]] = 1
            local = dilation(local, footprint=selem)
            stim_mask = np.maximum(stim_mask, local)

        return stim_mask, None


stimulator_up_down = StimUpDown(fraction=0.2)

# %% [markdown]
# ## 7. Channels and the pipeline

# %%
from faro.core.data_structures import Channel
from faro.core.pipeline import ImageProcessingPipeline

IMAGING_CHANNEL = Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP)
STIM_CHANNEL = Channel(config=site.FRAP_CHANNEL, exposure=100, group=site.CHANNEL_GROUP)

base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
path = os.path.join(base_path, "02_demo_optogenetic_inscoper")

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=feature_extractor,
    tracker=tracker,
    stimulator=stimulator_up,
)
print(f"storage: {path}")

# %% [markdown]
# ## 8. Preview segmentation and stimulation, and price the burn
#
# The original's preview shows raw / labels / mask. This one adds the number
# that has no analogue on a simulated SLM: what the mask costs in scan path, and
# whether the galvo will fire it at all.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

mic.mmc.setConfig(site.CHANNEL_GROUP, IMAGING_CHANNEL.config)
mic.mmc.setExposure(IMAGING_CHANNEL.exposure)
mic.mmc.snapImage()
test_img = mic.mmc.getImage()

labels_preview = segmentator.segment(test_img)
mask_preview, _ = stimulator_up.get_stim_mask(
    label_images={"labels": labels_preview}, metadata={}
)

print(f"segmented {labels_preview.max()} cell(s); "
      f"mask lights {int(np.count_nonzero(mask_preview))} px")
plan = site.price_mask(mic, mask_preview)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_img, cmap="gray")
axes[0].set_title(f"Raw — {IMAGING_CHANNEL.config}")
axes[1].imshow(labels_preview, cmap="nipy_spectral")
axes[1].set_title(f"Segmentation ({labels_preview.max()} cells)")
axes[2].imshow(mask_preview, cmap="Reds")
axes[2].set_title("Stim mask (top 20% of each cell)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

if plan is None:
    print(
        "This mask cannot be fired. Lower StimUp's fraction, crop further with "
        "site.set_roi(mic, 256), or raise the watershed's min_distance so fewer "
        "cells are found."
    )

# %% [markdown]
# ## 9. Set up the experiment
#
# `RTMSequence` is an `MDASequence` subclass that yields `RTMEvent`s, adding
# stimulation frames and channels plus arbitrary metadata for the pipeline.
# Chain phases with `combine(..., axis="t")`.
#
# The only Inscoper-specific line is `stage_positions`: `site.at(mic)` instead
# of the original's literal `(0.0, 0.0, 0.0)`, which on this stage means the
# machine origin rather than "here".

# %%
from faro.core.data_structures import RTMSequence, combine
from faro.core.utils import events_to_dataframe

N_FRAMES = 20
INTERVAL_S = 8.0
STIM_START, STIM_END = 2, 18

acq = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": N_FRAMES},
    stage_positions=[site.at(mic)],
    channels=[IMAGING_CHANNEL],
    stim_channels=[STIM_CHANNEL],
    stim_frames=range(STIM_START, STIM_END),
    rtm_metadata={
        "phase_name": "top-edge-stim",
        "phase_id": 0,
        "treatment_name": "top-edge-stim",
    },
)

events = list(acq)
print(f"{len(events)} RTMEvents")
events_to_dataframe(events).head()

# %% [markdown]
# ## 10. Validate before running
#
# `ctrl.validate_events(events)` runs both pre-flights in one call:
#
# 1. **Pipeline** — method signatures match their base class, and the events
#    carry the `required_metadata` keys each component declares.
# 2. **Hardware** — channel configs exist, exposures are inside the camera's
#    limits, device property values are in range.
#
# It also runs automatically at the start of `run_experiment` (pass
# `validate=False` to skip). On this microscope there is a **third** pre-flight
# it cannot do — whether the galvo can fire the mask — which is the pricing
# above, and `site.probe_stim_mask` below with the tracks a stimulator may need.
#
# To declare required metadata on your own component:
#
# ```python
# class MyStimulator(Stim):
#     required_metadata: set[str] = {"stim_fraction", "stim_target"}
# ```

# %%
from faro.core.controller import Controller

writer = site.make_writer(path)
ctrl = Controller(mic, pipeline, writer=writer)
assert ctrl.validate_events(events), "validation reported problems — see warnings"

# StimUpDown returns an empty mask without tracks, so price it the way the
# pipeline will actually call it.
print()
print("--- StimUpDown, priced with the tracks the pipeline would supply ---")
site.probe_stim_mask(
    mic,
    segmentator,
    stimulator_up_down,
    channel=IMAGING_CHANNEL.config,
    exposure=IMAGING_CHANNEL.exposure,
)

# %% [markdown]
# ## 11. Run
#
# A live progress bar, plus a timing check: with hardware in the loop the
# question "did the pipeline keep up with the interval?" has a real answer, and
# a stimulated frame now costs a galvo scan on top of imaging and segmentation.

# %%
import io
import logging
from tqdm.auto import tqdm
from faro.core.data_structures import ImgType

n_total = len(events)
pbar = tqdm(total=n_total, desc="Experiment", unit="frame")
_state = {"frames": 0, "t0": None, "max_delay": 0.0, "late": 0}
LATE_THRESHOLD_S = 1.0


def _on_frame(img, event):
    md = event.metadata or {}
    if md.get("img_type") != ImgType.IMG_RAW:
        return

    _state["frames"] += 1
    pbar.update(1)

    now = time.time()
    if _state["t0"] is None:
        _state["t0"] = now
    delay = (now - _state["t0"]) - md.get("time", 0)
    _state["max_delay"] = max(_state["max_delay"], delay)
    late = ""
    if delay > LATE_THRESHOLD_S:
        _state["late"] += 1
        late = f"  LATE +{delay:.1f}s"

    fov = md.get("fov", 0)
    n_cells = ctrl._analyzer.get_fov_state(fov).n_cells_latest
    pbar.set_postfix_str(
        f"t {md.get('timestep', 0) + 1}/{n_total}, cells={n_cells}, "
        f"delay={delay:+.2f}s{late}"
    )


mic.connect_frame(_on_frame)

# Quiet the per-frame pipeline chatter, but keep it retrievable.
_stdout, sys.stdout = sys.stdout, io.StringIO()
_logger = logging.getLogger("pymmcore-plus")
_prev_level = _logger.level
_logger.setLevel(logging.WARNING)
try:
    ctrl.run_experiment(events, stim_mode="current").wait()
    ctrl.finish_experiment()
finally:
    _captured, sys.stdout = sys.stdout.getvalue(), _stdout
    _logger.setLevel(_prev_level)
    try:
        mic.disconnect_frame(_on_frame)
    except Exception:
        pass
    pbar.close()

print("Timing summary:")
print(f"  frames received : {_state['frames']}/{n_total}")
print(f"  max delay       : {_state['max_delay']:.2f}s")
print(f"  late (>{LATE_THRESHOLD_S}s)   : {_state['late']}/{n_total}")
if _state["late"]:
    print(
        "  Some frames could not keep up. On this microscope the usual cause is\n"
        "  the galvo scan on stimulated frames (priced above) landing on top of\n"
        "  imaging and segmentation. Raise the interval, shrink the mask, or\n"
        "  stimulate on fewer frames."
    )

# %% [markdown]
# ## 12. Visualise the result
#
# Tracks over the last frame with the stim mask overlaid, and the same tracks
# normalised to their own starting point — which is where directed migration
# shows up as a cloud shifted off the origin.

# %%
import glob

import tifffile

utils.generate_exp_data_from_tracks(path)
tracks = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
particles = tracks["particle"].unique()
print(f"{len(particles)} particles across {tracks['timestep'].nunique()} timesteps")

raw_files = sorted(glob.glob(os.path.join(path, "raw", "*.tif*")))
stim_files = sorted(glob.glob(os.path.join(path, "stim_mask", "*.tif*")))
last_img = np.squeeze(tifffile.imread(raw_files[-1]))
if last_img.ndim == 3:
    last_img = last_img[0]

stim_mask_img = None
for sf in reversed(stim_files):
    candidate = np.squeeze(tifffile.imread(sf))
    if candidate.any():
        stim_mask_img = candidate
        break

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=130)
cmap = plt.cm.tab20

ax = axes[0]
ax.imshow(last_img, cmap="gray_r")
if stim_mask_img is not None:
    ax.imshow(
        np.ma.masked_where(stim_mask_img == 0, stim_mask_img),
        cmap="Blues", alpha=0.7, vmin=0, vmax=1,
    )
for i, pid in enumerate(particles):
    t = tracks[tracks["particle"] == pid].sort_values("timestep")
    # The pipeline's x/y are image row/col respectively, so plot y against x.
    ax.plot(t["y"], t["x"], color=cmap(i % 20), lw=1.4, alpha=0.85)
ax.set_title("Tracks over the last frame, stim mask in blue")
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1]
for i, pid in enumerate(particles):
    t = tracks[tracks["particle"] == pid].sort_values("timestep")
    if t.empty:
        continue
    x0, y0 = t.iloc[0]["x"], t.iloc[0]["y"]
    ax.plot(t["y"] - y0, t["x"] - x0, color=cmap(i % 20), alpha=0.7)
ax.set_xlabel("dx (px)")
ax.set_ylabel("dy (px)")
ax.set_title("Normalised displacement")
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Did the cells move towards the light?
#
# The claim the experiment makes, reduced to one number. `StimUp` lights the
# **top** of each cell, so a working response is a **negative** median step in
# the image-row direction (the pipeline's `x`), and it should be more negative
# inside the stim window than outside it.
#
# On a short run this will not be significant — it is here so the analysis is
# in place, and so the sign convention is written down rather than rediscovered.

# %%
tracks = tracks.sort_values(["particle", "timestep"])
tracks["dx"] = tracks.groupby("particle")["x"].diff()      # image row = "up/down"

step = tracks.groupby("timestep")["dx"].agg(["median", "sem", "count"]).reset_index()
in_stim = tracks["timestep"].between(STIM_START, STIM_END - 1)
print(f"median row-step, in stim window : {tracks.loc[in_stim, 'dx'].median():+.3f} px")
print(f"median row-step, outside        : {tracks.loc[~in_stim, 'dx'].median():+.3f} px")
print("(negative = towards the top of the frame, i.e. towards the light)")

fig, ax = plt.subplots(figsize=(9, 3.5), dpi=130)
ax.fill_betweenx([-5, 5], STIM_START, STIM_END - 1, color="cyan", alpha=0.12,
                 label="stim window")
ax.axhline(0, color="k", lw=0.5)
ax.plot(step["timestep"], step["median"], "o-", ms=3, lw=1.2, color="tab:blue")
ax.fill_between(
    step["timestep"],
    step["median"] - step["sem"],
    step["median"] + step["sem"],
    alpha=0.25, color="tab:blue",
)
ax.set_xlabel("frame")
ax.set_ylabel("median row-step (px/frame)")
ax.set_title("Movement per frame (negative = towards the stimulated edge)")
ax.set_ylim(-5, 5)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Export a stimulation-overlay movie

# %%
import imageio.v3 as iio

frames = []
for raw_f, stim_f in zip(raw_files, stim_files):
    raw = np.squeeze(tifffile.imread(raw_f))
    if raw.ndim == 3:
        raw = raw[0]
    stim = np.squeeze(tifffile.imread(stim_f))

    lo, hi = float(raw.min()), float(raw.max())
    raw_u8 = (255 - (raw - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    rgb = np.stack([raw_u8] * 3, axis=-1)

    mask = stim > 0
    if mask.any():
        blue = np.array([50, 120, 255], dtype=np.float32)
        rgb[mask] = (rgb[mask] * 0.5 + blue * 0.5).astype(np.uint8)
    frames.append(rgb)

# The original writes an .mp4. That needs an ffmpeg backend, and neither
# imageio-ffmpeg nor PyAV is installed in this environment -- without one,
# imageio silently dispatches .mp4 to the *tifffile* plugin and the call fails
# with "TiffWriter.write() got an unexpected keyword argument 'fps'", which
# names neither the format nor the missing package. GIF goes through Pillow and
# always works, so it is the fallback.
if not frames:
    print("no frames to export - raw/ and stim_mask/ did not line up")
else:
    stack = np.stack(frames)
    for name, kwargs in (
        ("stimulation_overlay.mp4", {"fps": 6, "codec": "libx264"}),
        ("stimulation_overlay.gif", {"fps": 6, "loop": 0}),
    ):
        out = os.path.join(path, name)
        try:
            iio.imwrite(out, stack, **kwargs)
        except Exception as exc:
            print(f"{name}: not written ({type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:70]})")
            continue
        print(f"Saved {len(frames)} frames to {out}")
        break
    else:
        print("Could not write a movie in any format. "
              "For .mp4 install one of: pip install imageio-ffmpeg")

# %% [markdown]
# ## 15. Multi-phase: baseline -> stim -> recovery
#
# `combine(..., axis="t")` chains `RTMSequence` phases into one event list,
# offsetting timepoints and `min_start_time` so they run back to back. This is
# the shape a real optogenetics experiment takes: measure, perturb, watch it
# relax.
#
# A fresh `Controller` is created here because the phases are one experiment
# with its own storage. Within one experiment, use `continue_experiment` so
# tracking state carries across.

# %%
n_baseline, n_stim, n_recovery = 3, 8, 3
stim_fraction = 0.2

baseline = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": n_baseline},
    stage_positions=[site.at(mic)],
    channels=[IMAGING_CHANNEL],
    rtm_metadata={"phase_name": "baseline", "phase_id": 0,
                  "treatment_name": "baseline"},
)

stim_phase = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": n_stim},
    stage_positions=[site.at(mic)],
    channels=[IMAGING_CHANNEL],
    stim_channels=[STIM_CHANNEL],
    stim_frames=range(n_stim),
    rtm_metadata={"phase_name": "stimulation", "phase_id": 1,
                  "treatment_name": "top-edge-stim"},
)

recovery = RTMSequence(
    time_plan={"interval": INTERVAL_S, "loops": n_recovery},
    stage_positions=[site.at(mic)],
    channels=[IMAGING_CHANNEL],
    rtm_metadata={"phase_name": "recovery", "phase_id": 2,
                  "treatment_name": "recovery"},
)

mp_events = combine(baseline, stim_phase, recovery, axis="t")
df_mp = events_to_dataframe(mp_events)
print(f"{len(mp_events)} events: {n_baseline} baseline + {n_stim} stim "
      f"+ {n_recovery} recovery")
print(f"stim frames: {df_mp[df_mp['stim']]['timestep'].min()}"
      f"–{df_mp[df_mp['stim']]['timestep'].max()}")
df_mp.head()

# %%
path_mp = os.path.join(base_path, "02_demo_optogenetic_inscoper_multiphase")

pipeline_mp = ImageProcessingPipeline(
    storage_path=path_mp,
    segmentators=segmentators,
    feature_extractor=SimpleFE("labels"),
    tracker=TrackerTrackpy(search_range=20),
    stimulator=StimUp(fraction=stim_fraction),
)
ctrl_mp = Controller(mic, pipeline_mp, writer=site.make_writer(path_mp))
assert ctrl_mp.validate_events(mp_events), "multi-phase validation reported problems"

pbar = tqdm(total=len(mp_events), desc="Multi-phase", unit="frame")


def _on_frame_mp(img, event):
    if (event.metadata or {}).get("img_type") == ImgType.IMG_RAW:
        pbar.update(1)


mic.connect_frame(_on_frame_mp)
_stdout, sys.stdout = sys.stdout, io.StringIO()
_logger.setLevel(logging.WARNING)
try:
    ctrl_mp.run_experiment(mp_events, stim_mode="current").wait()
    ctrl_mp.finish_experiment()
finally:
    sys.stdout = _stdout
    _logger.setLevel(_prev_level)
    try:
        mic.disconnect_frame(_on_frame_mp)
    except Exception:
        pass
    pbar.close()
print("multi-phase run complete")

# %%
utils.generate_exp_data_from_tracks(path_mp)
tracks_mp = pd.read_parquet(os.path.join(path_mp, "exp_data.parquet"))
tracks_mp = tracks_mp.sort_values(["particle", "timestep"])
tracks_mp["dx"] = tracks_mp.groupby("particle")["x"].diff()

speed = tracks_mp.groupby("timestep")["dx"].agg(["median", "sem"]).reset_index()
stim_t = df_mp[df_mp["stim"]]["timestep"]
stim_start_t, stim_end_t = int(stim_t.min()), int(stim_t.max())

fig, ax = plt.subplots(figsize=(10, 4), dpi=130)
ax.fill_betweenx([-5, 5], stim_start_t, stim_end_t, color="cyan", alpha=0.12,
                 label="stim window")
ax.axvline(stim_start_t, ls="--", color="teal", lw=1)
ax.axvline(stim_end_t, ls="--", color="teal", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.plot(speed["timestep"], speed["median"], "o-", ms=3, lw=1.2, color="tab:blue")
ax.fill_between(
    speed["timestep"],
    speed["median"] - speed["sem"],
    speed["median"] + speed["sem"],
    alpha=0.25, color="tab:blue",
)
ax.set_xlabel("frame")
ax.set_ylabel("median row-step (px/frame)")
ax.set_title("Baseline -> stim -> recovery (negative = towards the light)")
ax.set_ylim(-5, 5)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
