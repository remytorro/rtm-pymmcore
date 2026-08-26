# %% [markdown]
# # ERK-KTR stimulation ramp / pre-exposure drug test — Inscoper microscope
#
# The Inscoper variant of `stim_ramp_dfacquire.ipynb`. Several FOV groups get
# different stimulation treatment sets; each group is split at a timestep into a
# pre-drug and a post-drug half, with the drug pipetted by hand in between.
#
# Site configuration lives in `../inscoper_site.py`.

# %% [markdown]
# ## Read this before porting the ramp
#
# The original ramps the **stimulation exposure**: `stim_exposure_list` runs
# `range(0, 900, 10)` so each pulse is longer than the last. On a lamp that is a
# dose ramp — the light is on for longer, so the cells receive more.
#
# **That does not work here, and it fails silently.** The stim device is a FRAP
# galvo. Its burn takes:
#
# * its **laser line and power** from the designated channel's `.cbc`, not from
#   `stim_power`;
# * its **duration** from the scan path of the mask (priced below), not from
#   `stim_exposure`;
# * its **dose** from the number of scan repetitions.
#
# So a `stim_exposure` ramp writes an increasing column into
# `exp_data.parquet` while every burn delivers exactly the same dose. Analysis
# then reports a dose-response against a dose that never varied.
#
# **Repetitions are not reachable from faro today.** The bridge would honour
# `event.metadata["frap"]["repetitions"]`, but faro's stimulation transport is
# `SLMImage(data, device, exposure)` — mask, device, exposure and nothing else
# (`faro/core/_useq_compat.py`, `Controller._build_stim_slm`). There is no field
# for repetitions, so a per-event dose ramp cannot be expressed without a change
# to faro.
#
# **What this notebook ramps instead: frequency.** The dose a cell integrates
# over a phase is (dose per burn) x (number of burns), and the number of burns
# *is* fully expressible — it is just `stim_timestep`. So each treatment here
# stimulates every *n*-th frame, with *n* the ramped variable. The physics is
# different from a pulse-width ramp (same peak, different duty cycle) and the
# notebook says so in the treatment name, so the record cannot be mistaken for
# one.

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
import matplotlib.pyplot as plt

from faro.core.data_structures import (
    Channel,
    StimTreatment,
    SegmentationMethod,
)
import faro.core.utils as utils

# %%
# 256 px so the whole field is stimulable in one fire — see site.FULL_FOV_NOTE.
STIM_FIELD_PX = 256
mic = site.make_microscope(roi=STIM_FIELD_PX)

# %% [markdown]
# ## 2. Configuration

# %%
N_FRAMES = 24                    # timesteps per FOV group, split at SPLIT_AT
SPLIT_AT = 12                    # drug goes in here

TIME_BETWEEN_TIMESTEPS = 30      # seconds between timesteps
TIME_PER_FOV = 4.0               # seconds per FOV

ADD_STIM_EXPOSURE_GROUP = False
REGULAR_SPACING_BETWEEN_STIMULATIONS = False

site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV)

## Storage
base_path = os.path.join(os.path.expanduser("~"), "Desktop", "Remy", "exps")
experiment_name = "11_stim_ramp_dfacquire_inscoper"
path = os.path.join(base_path, experiment_name)

## Imaging channels — order matters, channel 0 is segmented.
channels = [
    Channel(config="470 WF", exposure=30, group=site.CHANNEL_GROUP),   # nuclear marker
    Channel(config="550 WF", exposure=40, group=site.CHANNEL_GROUP),   # ERK-KTR reporter
]

channel_optocheck = Channel(config="640 WF", exposure=80, group=site.CHANNEL_GROUP)
optocheck_timepoints = (N_FRAMES - 1,)

condition = ["EGFR"]
n_fovs_per_well = 3

print(f"storage: {path}")
print(f"{N_FRAMES} frames, drug at timestep {SPLIT_AT}")

# %% [markdown]
# ## 3. Treatment library — a frequency ramp
#
# `build_stim_treatments` takes compact specs and returns `StimTreatment`
# objects, as in the original. The difference is the ramped variable: the
# original's fourth field is an exposure in ms, here it is **`every_n`** — the
# stimulation period in frames. `every_n=1` is every frame (highest dose),
# `every_n=4` is every fourth (a quarter of the dose).
#
# The `prestim` flag still decides whether the pre-drug half is stimulated at
# all, so the timestep set is built from `every_n` and `prestim` together
# rather than looked up in a table.

# %%
STIM_EXPOSURE_MS = 100      # constant, and bookkeeping only — see the note above
STIM_POWER = 100            # ditto; the .cbc sets the real laser power


def stim_timesteps(every_n: int, prestim: bool) -> tuple[int, ...]:
    """Frames to stimulate: every *every_n*-th, pre-drug half optional.

    Frame 0 is never stimulated — it is the baseline the response is measured
    against, and in "current" stim mode it is also the frame whose segmentation
    the first burn is aimed with.
    """
    post = tuple(range(SPLIT_AT, N_FRAMES, every_n))
    if not prestim:
        return post
    return tuple(range(max(1, every_n), SPLIT_AT, every_n)) + post


def build_stim_treatments(specs, *, stim_power=STIM_POWER):
    """StimTreatment objects from ``(drug_nM, prestim, every_n)`` specs.

    The treatment name records *how* the dose was varied, so an analysis of
    this data cannot silently read it as a pulse-width ramp:
    ``"300nM every2f + stimfreq"``.
    """
    treatments = []
    for drug_nM, prestim, every_n in specs:
        if not isinstance(every_n, int) or every_n < 1:
            raise ValueError(f"every_n must be a positive int, got {every_n!r}")
        sign = "+" if prestim else "-"
        timesteps = stim_timesteps(every_n, prestim)
        name = f"{drug_nM}nM every{every_n}f {sign} stimfreq"
        print(f"{name:34s} {len(timesteps):3d} burns at {timesteps[:6]}"
              f"{'...' if len(timesteps) > 6 else ''}")
        treatments.append(
            StimTreatment(
                treatment_name=name,
                stim_timestep=timesteps,
                stim_exposure_list=STIM_EXPOSURE_MS,
                auto_repeat_stim_exposure=True,
                stim_power=stim_power,
                stim_channel_name=site.FRAP_CHANNEL,
                stim_channel_group=site.CHANNEL_GROUP,
                stim_channel_device_name="iLas2",
                stim_channel_power_property_name="BluePower",
            )
        )
    return treatments


def parse_treatment_name(df_acquire):
    """Split treatment_name into columns for analysis.

    Format: ``"{drug}nM every{n}f {+/-} stimfreq"``. Produces
    ``drug_concentration``, ``stim_every_n_frames``, ``prestimulation`` and
    ``stim_ramp_kind`` — the last so a downstream analysis can tell this apart
    from an exposure ramp without parsing prose.
    """
    parts = df_acquire["treatment_name"].str.split(" ", expand=True)
    df_acquire["drug_concentration"] = parts[0].str.replace("nM", "").astype(int)
    df_acquire["stim_every_n_frames"] = (
        parts[1].str.replace("every", "").str.replace("f", "").astype(int)
    )
    df_acquire["prestimulation"] = parts[2] == "+"
    df_acquire["stim_ramp_kind"] = "frequency"
    return df_acquire


def split_df_acquire(df, split_timestep):
    """Split at *split_timestep*; the second half's time restarts at 0.

    That is what lets the drug be added between the two halves without the
    second half trying to catch up to a schedule that ran while you pipetted.
    """
    rows = df.loc[df["timestep"] == split_timestep, "time"]
    if rows.empty:
        raise ValueError(
            f"no rows at timestep {split_timestep}; N_FRAMES is {N_FRAMES}"
        )
    t_offset = rows.iloc[0]
    df_1 = df[df["timestep"] < split_timestep]
    df_2 = df[df["timestep"] >= split_timestep].copy()
    df_2["time"] = df_2["time"] - t_offset
    return df_1, df_2

# %%
# One treatment set per FOV group. Each spec is (drug_nM, prestim, every_n).
stim_specs_group_1 = [
    (0, False, 1),      # control, post-drug only, every frame
    (0, True, 1),       # control, pre + post
    (300, False, 1),
    (300, True, 1),
    (300, False, 2),    # half the burns -> half the integrated dose
    (300, True, 4),     # a quarter
]
stim_phase_1 = build_stim_treatments(stim_specs_group_1)

# %% [markdown]
# ## 4. FRAP preflight and the whole-field price
#
# Every burn in this experiment is the *same* whole-field mask, so one price
# covers the run — which is the flip side of not being able to ramp the dose per
# burn.

# %%
problems = site.designate_frap(mic, site.FRAP_CHANNEL)
assert not problems, f"FRAP path is not open: {problems}"

plan = site.price_full_field(mic)
assert plan is not None, (
    f"a whole-field burn on a {mic.image_width}x{mic.image_height} frame is over "
    "budget. Lower STIM_FIELD_PX (256 is known to fit)."
)

# The burn is not free: a whole-field raster at this frame size takes seconds,
# and every stimulated timestep pays it on top of the imaging. Check it fits.
BURN_S = plan.scan_px / 7200.0 if hasattr(plan, "scan_px") else None
print()
print(f"imaging cost per FOV : ~{TIME_PER_FOV:.1f} s (TIME_PER_FOV)")
print(f"interval             : {TIME_BETWEEN_TIMESTEPS} s")
print("A stimulated timestep costs imaging + the burn above. If the sum exceeds")
print("the interval, frames arrive late — apply_fov_batching only accounts for")
print("TIME_PER_FOV, so raise TIME_PER_FOV to include the burn.")

# %% [markdown]
# ## 5. Pipeline

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
# ## 6. GUI, and a look at the FOV layout

# %%
viewer, mm_wdg = site.open_napari(mic, title="11 stim ramp (Inscoper)")

# %%
USE_GUI_FOVS = True
if USE_GUI_FOVS:
    fovs = utils.generate_fov_positions(mic, viewer=viewer)
else:
    fovs = site.fovs_here(mic, 1)

print(f"{len(fovs)} FOV(s)")
site.check_fov_timing(TIME_BETWEEN_TIMESTEPS, TIME_PER_FOV, len(fovs))

if len(fovs) > 1:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([f.x for f in fovs], [f.y for f in fovs])
    for i, f in enumerate(fovs):
        ax.annotate(str(i), (f.x, f.y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"{len(fovs)} FOV positions")
    ax.axis("equal")
    plt.tight_layout()

# %% [markdown]
# ## 7. Build the DataFrame and split it at the drug
#
# The original builds three groups of ~21 FOVs each and splits every one. This
# keeps the same shape but drives it off `len(fovs)`, so it works with however
# many positions you actually picked rather than silently producing empty groups.

# %%
N_GROUPS = 1 if len(fovs) < 6 else 3
group_size = max(1, len(fovs) // N_GROUPS)
groups = [fovs[i : i + group_size] for i in range(0, len(fovs), group_size)]
print(f"{len(groups)} FOV group(s) of up to {group_size}")

halves = []
for gi, group_fovs in enumerate(groups):
    df = utils.generate_df_acquire(
        group_fovs,
        n_frames=N_FRAMES,
        time_between_timesteps=TIME_BETWEEN_TIMESTEPS,
        time_per_fov=TIME_PER_FOV,
        channels=channels,
        condition=condition,
        channel_optocheck=channel_optocheck,
        optocheck_timepoints=optocheck_timepoints,
        phase_name=f"group{gi}",
        phase_id=gi,
    )
    df = utils.apply_stim_treatments_to_df_acquire(
        df,
        stim_phase_1,
        condition,
        n_fovs_per_well=n_fovs_per_well if len(group_fovs) >= (n_fovs_per_well or 1) else None,
        add_stim_exposure_group=ADD_STIM_EXPOSURE_GROUP,
        regular_spacing_between_stimulations=REGULAR_SPACING_BETWEEN_STIMULATIONS,
    )
    df = parse_treatment_name(df)
    pre, post = split_df_acquire(df, SPLIT_AT)
    halves.append((pre, post))
    print(f"group {gi}: {len(pre)} pre-drug rows ({int(pre['stim'].sum())} stim), "
          f"{len(post)} post-drug rows ({int(post['stim'].sum())} stim)")

# %%
# Confirm the frequency ramp actually produced different burn counts. If every
# treatment has the same count the ramp did nothing, which is the failure this
# whole notebook is arranged to avoid.
pre_all = pd.concat([h[0] for h in halves], ignore_index=True)
post_all = pd.concat([h[1] for h in halves], ignore_index=True)
burns = (
    pd.concat([pre_all, post_all], ignore_index=True)
    .groupby(["treatment_name", "stim_every_n_frames"])["stim"]
    .sum()
    .astype(int)
    .sort_index()
)
print(burns.to_string())
assert burns.nunique() > 1 or len(burns) == 1, (
    "every treatment fires the same number of burns, so the dose ramp is flat. "
    "Check stim_timesteps() against N_FRAMES and SPLIT_AT."
)

# %% [markdown]
# ## 8. Run: pre-drug half, pipette, post-drug half
#
# `continue_experiment` for everything after the first phase, so the Analyzer —
# and with it tracking state, timestep counters and filenames — carries across
# the drug addition. A fresh `Controller` per half restarts all three and makes
# the two halves' tracks unlinkable.

# %%
from faro.core.conversion import df_to_events
from faro.widgets import ExperimentStatusWidget

ctrl = Controller(mic, pipeline, writer=writer)
viewer.window.add_dock_widget(
    ExperimentStatusWidget(ctrl), name="experiment status", area="right"
)

events_pre = df_to_events(halves[0][0])
assert ctrl.validate_events(events_pre), "pre-drug validation reported problems"
handle = ctrl.run_experiment(events_pre, stim_mode="current")
print(f"group 0 pre-drug running: {len(events_pre)} events")

# %%
handle.wait()
print("pre-drug half done — PIPETTE THE DRUG, then run the next cell")

# %%
events_post = df_to_events(halves[0][1])
assert ctrl.validate_events(events_post), "post-drug validation reported problems"
handle = ctrl.continue_experiment(events_post, stim_mode="current")
print(f"group 0 post-drug running: {len(events_post)} events")

# %% [markdown]
# Repeat the two cells above for each remaining group: run
# `halves[i][0]`, pipette, then `halves[i][1]` — always through
# `ctrl.continue_experiment`.

# %% [markdown]
# ## 9. Post-processing

# %%
handle.wait()
ctrl.finish_experiment()
utils.generate_exp_data_from_tracks(path)

df_exp = pd.read_parquet(os.path.join(path, "exp_data.parquet"))
print(f"{len(df_exp)} cell rows")
phase_col = next((c for c in ("phase", "phase_name") if c in df_exp.columns), None)
if phase_col:
    print(df_exp.groupby(phase_col)["timestep"].agg(["min", "max", "count"]))
df_exp.head()

# %% [markdown]
# ## 10. Release the hardware

# %%
site.clear_roi(mic)
mic.post_experiment()
print("bridge closed")
