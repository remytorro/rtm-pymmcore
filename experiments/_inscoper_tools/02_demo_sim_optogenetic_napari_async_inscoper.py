# %% [markdown]
# # Async optogenetics with napari + ExperimentStatusWidget — Inscoper
#
# The Inscoper variant of `demo_sim_optogenetic_napari_async.ipynb`. Same
# pipeline as `demo_sim_optogenetic_inscoper.ipynb`, but arranged around the
# **asynchronous** run API:
#
# * a live **napari** viewer with the **napari-micromanager** GUI, so frames
#   stream in as they are acquired;
# * the **`ExperimentStatusWidget`** docked in it, mirroring the controller's
#   current `RunHandle` — state, FOV, event index, frames received, lag, and a
#   Stop button;
# * `ctrl.run_experiment(...)` returns a `RunHandle` **immediately**, so the
#   kernel stays free to poll status, cancel, or queue a `continue_experiment`.
#
# Site configuration lives in `../inscoper_site.py`.
#
# Run the cells one at a time in a kernel with Qt running — start Jupyter from a
# terminal so the napari window can appear.

# %% [markdown]
# ## 1. Start the microscope
#
# The original starts `virtual_microscope`'s optogenetic backend. Here it is the
# real bridge. Two things follow that matter for an *async* notebook in
# particular:
#
# * **The load is the slow part**, not the run. Give it a few seconds.
# * **`post_experiment()` closes the bridge**, so it belongs at the very end —
#   not after the first `handle.wait()`. The original calls it before
#   `ctrl.finish_experiment()`; do the reverse here, or the pipeline is drained
#   against a closed bridge.

# %%
import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))
import inscoper_site as site

import numpy as np
import pandas as pd

import faro.core.utils as utils

mic = site.make_microscope(roi=512)
utils.print_configs(mic.mmc)

# %% [markdown]
# ## 2. Open napari + napari-micromanager
#
# `site.open_napari` binds the widget to the Inscoper bridge. Note the
# assignment form: `InscoperMicroscope.init_scope` has already registered the
# bridge as the `CMMCorePlus` singleton, and `mm_wdg._mmc = mic.mmc` is what is
# known to work against it.
#
# Click **MDA** once if you want to pick FOV positions from the widget later.

# %%
viewer, mm_wdg = site.open_napari(mic, title="02 async optogenetics (Inscoper)")

# %% [markdown]
# ## 3. Build the pipeline
#
# Same components as the original: watershed segmentation, trackpy, a
# region-props feature extractor, and the `StimUpDown` stimulator that pushes
# even-particle cells up and odd-particle cells down.

# %%
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation
from scipy import ndimage
import skimage.measure

from faro.segmentation.base import Segmentator
from faro.feature_extraction.base import FeatureExtractor
from faro.stimulation.base import StimWithPipeline
from faro.tracking.trackpy import TrackerTrackpy
from faro.core.data_structures import Channel, SegmentationMethod


class WatershedSegmentator(Segmentator):
    """Gaussian blur -> Otsu -> distance transform -> watershed."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        blurred = gaussian(image, sigma=3)
        thresh = threshold_otsu(blurred)
        binary = blurred > thresh
        distance = ndimage.distance_transform_edt(binary)
        coords = peak_local_max(distance, min_distance=20, labels=binary)
        markers = np.zeros(distance.shape, dtype=bool)
        markers[tuple(coords.T)] = True
        markers = label(markers)
        return watershed(-distance, markers, mask=binary)


class SimpleFE(FeatureExtractor):
    def __init__(self, used_mask):
        self.used_mask = used_mask
        super().__init__()

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table = skimage.measure.regionprops_table(
            labels[self.used_mask], properties=["label", "area"]
        )
        return pd.DataFrame.from_dict(table), None


class StimUpDown(StimWithPipeline):
    """Even particles -> illuminate the top edge; odd -> the bottom edge."""

    def __init__(self, fraction=0.2):
        self.fraction = fraction

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)
        selem = disk(3)

        # No tracks means no particle ids, so nothing to steer. This is why a
        # pre-flight has to supply tracks -- see site.probe_stim_mask.
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


segmentator = WatershedSegmentator()
segmentators = [
    SegmentationMethod(
        name="labels",
        segmentation_class=segmentator,
        use_channel=0,
        save_tracked=True,
    )
]
tracker = TrackerTrackpy(search_range=20)
feature_extractor = SimpleFE("labels")
stimulator = StimUpDown(fraction=0.2)

IMAGING_CHANNEL = Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP)
STIM_CHANNEL = Channel(config=site.FRAP_CHANNEL, exposure=50, group=site.CHANNEL_GROUP)

# %% [markdown]
# ## 4. Assemble the pipeline and controller
#
# The original writes to a fresh `tempfile.mkdtemp()` and then `rmtree`s it —
# which deletes the directory it just made, so `TiffWriter` recreates it. Kept
# simple here: a named folder under the usual experiment root, so the data
# survives the kernel.

# %%
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.controller import Controller

base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
path = os.path.join(base_path, "02_async_optogenetic_inscoper")

pipeline = ImageProcessingPipeline(
    storage_path=path,
    segmentators=segmentators,
    feature_extractor=feature_extractor,
    tracker=tracker,
    stimulator=stimulator,
)

writer = site.make_writer(path)
ctrl = Controller(mic, pipeline, writer=writer)
print(f"storage: {path}")

# %% [markdown]
# ## 5. FRAP preflight, priced with the tracks `StimUpDown` needs
#
# `StimUpDown` returns an **empty** mask without tracks, so pricing it naively
# reports a free burn for a mask that is in fact expensive — a pre-flight that
# always passes. `site.probe_stim_mask` synthesises the tracks frame the
# pipeline would supply.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

labels, mask, plan = site.probe_stim_mask(
    mic,
    segmentator,
    stimulator,
    channel=IMAGING_CHANNEL.config,
    exposure=IMAGING_CHANNEL.exposure,
)
assert plan is not None or mask is None, (
    "the up/down mask is over the scan budget — lower StimUpDown's fraction or "
    "crop further with site.set_roi(mic, 256)"
)

# %% [markdown]
# ## 6. Dock the `ExperimentStatusWidget`
#
# It subscribes to `ctrl.runStarted`, so it re-binds automatically whenever a
# new run begins — including a `continue_experiment`. **Stop** calls
# `handle.cancel()` on whichever run is currently bound.

# %%
from faro.widgets import ExperimentStatusWidget

status_widget = ExperimentStatusWidget(ctrl)
viewer.window.add_dock_widget(status_widget, name="experiment status", area="right")

# %% [markdown]
# ## 7. Build a multi-phase sequence
#
# Baseline -> stimulation -> recovery, joined with `combine(..., axis="t")`,
# which offsets each phase's `t` indices and `min_start_time` past the previous
# one. `wait(seconds)` inserts a settle pause between phases.
#
# Two Inscoper-specific choices:
#
# * `stage_positions=[site.at(mic), site.at(mic, dx=…)]` — the original writes
#   `[(0, 0, 0), (20, 20, 0)]`, and `(0, 0, 0)` on this stage is the machine
#   origin, not "here". The second position is a real displacement from the
#   first, so the two are genuinely different fields.
# * a slower interval than the original's 1 s. A stimulated frame here costs a
#   galvo scan (priced above) on top of imaging and segmentation, and the status
#   widget's *lag* readout is the honest measure of whether the interval holds.

# %%
from faro.core.data_structures import RTMSequence, combine, wait
from faro.core.utils import events_to_dataframe

n_baseline = 3
n_stim = 5
n_recovery = 3
interval_s = 8          # raise if the status widget shows growing lag
wait_s = 5              # settle before baseline and after stim

# ~1 camera field apart, so the two positions are different fields. The FOV
# cannot be computed from the pixel size on this bridge (getPixelSizeUm()
# returns 0.0), so this is a plain micrometre step.
FOV_STEP_UM = 150.0
positions = [site.at(mic), site.at(mic, dx=FOV_STEP_UM)]

baseline = RTMSequence(
    time_plan={"interval": interval_s, "loops": n_baseline},
    stage_positions=positions,
    channels=[IMAGING_CHANNEL],
    rtm_metadata={"phase_name": "baseline", "phase_id": 0,
                  "treatment_name": "baseline"},
)

stim_phase = RTMSequence(
    time_plan={"interval": interval_s, "loops": n_stim},
    stage_positions=positions,
    channels=[IMAGING_CHANNEL],
    stim_channels=[STIM_CHANNEL],
    stim_frames=range(n_stim),
    rtm_metadata={"phase_name": "stimulation", "phase_id": 1,
                  "treatment_name": "up-down-stim"},
)

recovery = RTMSequence(
    time_plan={"interval": interval_s, "loops": n_recovery},
    stage_positions=positions,
    channels=[IMAGING_CHANNEL],
    rtm_metadata={"phase_name": "recovery", "phase_id": 2,
                  "treatment_name": "recovery"},
)

events = combine(wait(wait_s), baseline, stim_phase, wait(wait_s), recovery, axis="t")
df_events = events_to_dataframe(events)
print(f"{len(events)} events: {wait_s}s wait + {n_baseline} baseline + "
      f"{n_stim} stim + {wait_s}s wait + {n_recovery} recovery")
print(f"across {len(positions)} FOVs")
df_events.head()

# %% [markdown]
# ## 8. Launch (non-blocking)
#
# The feed loop runs on a worker thread and the kernel returns at once. Watch
# the napari viewer for frames and the status dock for state / FOV / event index
# / lag. Clicking **Stop** exits at the next event boundary.
#
# `stim_mode="previous"` fires frame *t-1*'s mask before imaging frame *t*: one
# timestep stale but the burn does not delay the frame it belongs to. It is
# suppressed at `t=0`, where there is no predecessor — firing a blank mask would
# still arm the device. `"current"` images first and fires that frame's own
# mask; prefer it when the scan fits comfortably inside the interval.

# %%
assert ctrl.validate_events(events), "validation reported problems — see warnings"

handle = ctrl.run_experiment(events, stim_mode="previous")
print(f"run started, handle.is_running()={handle.is_running()}")
print(f"current state: {handle.status().state}")

# %% [markdown]
# ### Poll the status from the kernel
#
# Re-run as often as you like while the experiment runs.

# %%
s = handle.status()
print(f"state           : {s.state}")
print(f"events consumed : {s.n_events_consumed} / {s.n_events_total}")
print(f"frames received : {s.n_frames_received}")
print(f"current FOV     : {s.current_fov}")
print(f"current event   : {s.current_event_index}")
print(f"lag (ms)        : {s.lag_ms}")
print(f"bg errors       : {len(s.background_errors)}")
print(f"fatal error     : {s.fatal_error!r}")

# %% [markdown]
# ### Subscribe to live updates
#
# In *addition* to the dock widget, which already listens. Useful for piping
# status into your own logging.

# %%
def _print_status(status):
    print(
        f"[notify] state={status.state} "
        f"consumed={status.n_events_consumed}/{status.n_events_total} "
        f"frames={status.n_frames_received} lag_ms={status.lag_ms}"
    )


handle.statusChanged.connect(_print_status)

# %% [markdown]
# ### Cancel from the kernel (the Stop button's equivalent)

# %%
# handle.cancel()   # uncomment to abort

# %% [markdown]
# ## 9. Block until it finishes
#
# `wait()` re-raises any worker-side `fatal_error`, so a failed run surfaces
# here rather than as a quietly short dataset.

# %%
final_status = handle.wait()
print(f"final state     : {final_status.state}")
print(f"frames received : {final_status.n_frames_received}")
if final_status.fatal_error is not None:
    print(f"fatal           : {final_status.fatal_error!r}")

try:
    handle.statusChanged.disconnect(_print_status)
except Exception:
    pass

# %% [markdown]
# ## 10. Continue the experiment
#
# `ctrl.continue_experiment(...)` reuses the existing Analyzer, so tracking
# state, timestep counters and filenames carry on rather than restarting. Same
# async semantics: a fresh `RunHandle`, and the widget re-binds via
# `runStarted`.
#
# This is the call to reach for whenever an experiment continues after an
# interruption — a drug addition, a refocus, a pause. A second `Controller`
# would restart the counters and make the two halves' tracks unlinkable.

# %%
extra = RTMSequence(
    time_plan={"interval": interval_s, "loops": 3},
    stage_positions=positions,
    channels=[IMAGING_CHANNEL],
    rtm_metadata={"phase_name": "extra-recovery", "phase_id": 3,
                  "treatment_name": "recovery"},
)

handle2 = ctrl.continue_experiment(list(extra), stim_mode="previous")
print(f"continuation started, is_running={handle2.is_running()}")

# %%
print(handle2.wait().state)

# %% [markdown]
# ## 11. Finish and tear down
#
# Order matters: `ctrl.finish_experiment()` **first** — it drains the pipeline
# and closes the writer, and it needs the bridge alive — then
# `mic.post_experiment()`, which on this adapter closes the bridge for good. The
# original does it the other way round, which works on a simulator whose
# `post_experiment` is a no-op.

# %%
ctrl.finish_experiment()
utils.generate_exp_data_from_tracks(path)

df_exp = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"{len(df_exp)} cell rows, {df_exp['particle'].nunique()} particles, "
      f"{df_exp['fov'].nunique()} FOVs")
phase_col = next((c for c in ("phase_name", "phase") if c in df_exp.columns), None)
if phase_col:
    print(df_exp.groupby(phase_col)["timestep"].agg(["min", "max", "count"]))

# %%
site.clear_roi(mic)
mic.post_experiment()
print(f"bridge closed. All data under: {path}")
