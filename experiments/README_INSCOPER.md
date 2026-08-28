# Inscoper variants of the `faro` experiment notebooks

Each `*_inscoper.ipynb` in this tree is a port of the notebook next to it, run
against a real Inscoper microscope through
`faro.microscope.Inscoper.InscoperMicroscope` instead of the Micro-Manager
demo devices, the pertzlab scopes (`Moench`, `Jungfrau`, `Niesen`) or the
`virtual_microscope` simulator.

Everything that describes *this installation* lives in one file,
[`inscoper_site.py`](inscoper_site.py) — the `faro` counterpart of
`inscoper_useq/scripts/frap_config.py`. Porting the set to another Inscoper
microscope is that one file, not one block per notebook.

Values in it were read from the hardware on **2026-08-26** and cross-checked
against the `useq_compat` harness (`inscoper_useq/scripts/useq_compat/_harness.py`).

---

## Notebook map

| Original | Inscoper variant | Status |
|---|---|---|
| `01_demo_microsocpe/demo_microscope.ipynb` | `demo_microscope_inscoper.ipynb` | run end to end on hardware |
| `02_demo_sim_optogenetic/demo_sim_optogenetic.ipynb` | `demo_sim_optogenetic_inscoper.ipynb` | run end to end on hardware |
| `02_.../demo_sim_optogenetic_napari_async.ipynb` | `demo_sim_optogenetic_napari_async_inscoper.ipynb` | run end to end on hardware |
| `03_demo_writer_implementation/tiff_writer.ipynb` | `tiff_writer_inscoper.ipynb` | run end to end on hardware |
| `03_.../ome_zarr_writer.ipynb` | `ome_zarr_writer_inscoper.ipynb` | run end to end on hardware |
| `03_.../ome_zarr_writer_plate.ipynb` | `ome_zarr_writer_plate_inscoper.ipynb` | run end to end on hardware |
| `03_.../convert_tiff_to_omezarr.ipynb` | — | no microscope; nothing to port |
| `11_.../stim_rtmsequence.ipynb` | `stim_rtmsequence_inscoper.ipynb` | run end to end on hardware |
| `11_.../stim_dfacquire.ipynb` | `stim_dfacquire_inscoper.ipynb` | run end to end on hardware |
| `11_.../stim_ramp_dfacquire.ipynb` | `stim_ramp_dfacquire_inscoper.ipynb` | run end to end on hardware |
| `11_.../stim_rtmsequence_demo_mic.ipynb` | — | replays images from disk via `ControllerSimulated`; a hardware port would defeat its purpose |
| `21_cell_migration/cell_migration.ipynb` | `cell_migration_inscoper.ipynb` | run end to end on hardware |
| `22_line_stimulation/line_stimulation.ipynb` | `line_stimulation_inscoper.ipynb` | run end to end on hardware |
| `23_.../random_stim_per_cell_14px_patches.ipynb` | `random_stim_per_cell_14px_patches_inscoper.ipynb` | run end to end on hardware |
| `23_.../fov_screening.ipynb` | `fov_screening_inscoper.ipynb` | run end to end on hardware |
| `23_.../analyze_stim_intensity.ipynb` | — | offline analysis; nothing to port |
| `25_region_percentage_stim/region_percentage_stim.ipynb` | `region_percentage_stim_inscoper.ipynb` | run end to end on hardware |
| `90_reanalysis/reanalysis.ipynb` | — | offline by design |

"Run end to end on hardware" means every code cell was executed against the
live microscope on 2026-08-26, with the time plans shortened, using
[`_inscoper_tools/nbtest.py`](_inscoper_tools/nbtest.py) — a cell-by-cell runner
with napari stubbed. Frames were acquired, masks were fired, tracks were
written. What is *not* claimed: no biological response was measured. Every burn
so far has been aimed and priced, not dosed — see "What is still open" below.

---

## The five things that do not port, and why each fails quietly

A notebook ported by search-and-replacing its `mic = ...` line **runs** and is
**wrong**. These are the reasons, in the order they bite.

### 1. The licence is resolved against the working directory

`Bridge` looks for `inscoper.lic` in `os.getcwd()` — for a notebook, the folder
the `.ipynb` sits in. Without it `loadSystemConfiguration()` returns *without
raising* and **no devices come up**: `getImageWidth()` answers 0, the stage
reads `(0, 0, 0)`, every snap is empty, and the first visible symptom is a
shape error several cells later.

`site.check_license()` says so before the load; `site.make_microscope()`
refuses to hand back a 0×0 microscope. Each notebook folder has its own
licence (git-ignored via `*.lic`).

The same signature appears when another process already holds the hardware, or
the controller is off — which is why the error message names all three.

### 2. The stimulation device is a FRAP galvo, not a DMD

`use_frap_as_slm` is `True` on this configuration: `getSLMDevice()` returns
`""`, and `mic.dmd` is a shim whose dimensions mirror the camera FOV. Stim
masks reach the galvo through `event.slm_image` in **camera pixels**, and the
*firmware* applies the calibration per point via `interpIndex`.

Consequences the notebooks encode:

* **Never call `mic.calibrate_dmd(...)`.** A DMD affine would transform the
  mask a second time and aim the beam wrong. It is currently harmless only by
  accident — an uncalibrated DMD degrades to an identity resize — so
  *calibrating* it is what breaks FRAP. `InscoperMicroscope.uses_dmd_affine` is
  `False` and `prepare_stim_mask` passes the mask through; the notebooks assert
  both.
* **The stim channel cannot be picked by name.** Every stock channel called
  `... WF Quad FRAP` is an *imaging* channel through the quad FRAP dichroic and
  ships `iLas2::Shutter = 0`, which scans the galvo **in the dark**. The
  stimulation channel is `all lasers`, validated against the configuration's own
  `Ilda.frap.ActiveModeChannel` by `site.designate_frap()`.
* The DMD focus-checkerboard cells in `23_...` have no meaning here, and
  `getProperty(dmd, "OverlapMode")` raises — this bridge exposes no device
  properties beyond `Core` and the camera's `Exposure`.

### 3. A FRAP burn is bounded by scan path, not by mask area

One fire covers **~36,000 px of scan path (~5 s at ~7200 px/s)**, and this rig's
row pitch is **4 px**. `plan_mask` **refuses** an over-budget region rather than
truncating it, so an oversized mask is an exception at the first stim event.

Cost of a *filled square*, measured with `plan_mask`:

| field | pitch 4 | pitch 8 | pitch 16 | pitch 32 | pitch 64 | pitch 128 |
|---|---|---|---|---|---|---|
| 256 px | 2.3 s | 1.2 s | 0.6 s | 0.3 s | 0.2 s | 0.1 s |
| 352 px | 4.3 s | — | — | — | — | — |
| 384 px | **REFUSED** | 2.6 s | 1.3 s | 0.7 s | 0.4 s | 0.2 s |
| 512 px | **REFUSED** | 4.6 s | 2.3 s | 1.2 s | 0.6 s | 0.3 s |
| 1024 px | **REFUSED** | **REFUSED** | **REFUSED** | 4.7 s | 2.4 s | 1.3 s |
| 2304 px | **REFUSED** | **REFUSED** | **REFUSED** | **REFUSED** | **REFUSED** | 6.1 s* |

\* 342 points — fits the point budget but 43,630 px of path, i.e. over what one
fire covers. A dot grid, not a flood.

So **`StimWholeFOV` cannot be fired on any field wider than ~352 px.** The
full-FOV notebooks (`11_*`) therefore crop the camera to **256 px** and say so:
the field they image is the field they can stimulate. That is a real change in
what the experiment measures — fewer cells per FOV — and the honest alternative
is to stop asking for a flood and stimulate patterned regions instead, which is
what `21_`, `23_` and `25_` do and is far cheaper:

| what fires | cost |
|---|---|
| whole 512 px field | REFUSED |
| 20 cells, r=30 px, on the full 2304 px frame | 2.3 s |
| `StimUp` at 20 % on ~300 cells, 512 px frame | 2.8 s |
| `RandomStimPerCell14pxPatches`, 9 dots | **0.0 s** (28 px) |

The asymmetry is the reason the single-cell notebooks port cleanly and the
full-FOV ones need a crop.

**Every stim notebook prices its real mask before running.** `validate_events`
cannot do this — the mask does not exist until a frame has been segmented — so
`site.probe_stim_mask()` snaps, segments, builds the mask and prices it. It also
**synthesises the `tracks` frame**, because `StimUp`, `StimUpDown` and
`StimLine` return an *empty* mask without one, and a pre-flight that prices an
empty mask always passes.

### 4. Stage coordinates are absolute

`stage_positions=[{"x": 0, "y": 0, "z": 0}]` — in the demo and simulator
notebooks — does not mean "stay here". It commands the **machine origin**: a
full-travel move off the sample, and on the focus axis a move *towards* the
objective. `utils.generate_fov_positions(mic, fake_fovs=n)` has the same
problem; its positions are all `(0, 0, None)`.

Every position in these notebooks is an offset from where the stage already is:
`site.at(mic, dx, dy, dz)`, `site.here(mic)`, `site.fovs_here(mic, n)`.
`site.here` also range-checks the focus read-back (a Ti2 z drive travels a few
millimetres; anything outside 100–10,000 µm is a µm/nm mix-up) and falls back to
`site.FOCUS_UM`.

### 5. The environment is not the one `pyproject.toml` describes

The `py313` conda env this microscope runs in was missing four packages the
originals assume. The **pattern** matters more than the list, because none of
them failed at import time where you could act on it:

* **`ome_writers`** — declared in `pyproject.toml`, absent until it was
  **installed on 2026-08-26** as part of this work. `OmeZarrWriter` constructs
  fine without it and fails inside `init_stream`, i.e. *after* `run_experiment`
  has started, surfacing as the run's `fatal_error` with a bare
  `ModuleNotFoundError` that names neither the writer nor the package. Nothing
  is acquired. `site.make_writer()` decides up front and prints what it chose;
  it now returns `OmeZarrWriter`, and still falls back on a machine without the
  package.
* **`cellpose`** — `site.make_segmentator()` falls back to
  `SegmentatorThreshold(threshold=0.1)`. Pass `multichannel=True` when
  `use_channel` is a *list*: Cellpose reduces a `(C, Y, X)` stack itself, the
  fallback does not, and unreduced `(C, Y, X)` labels fail inside
  `regionprops_table` with *"Label and intensity image shapes must match"* — on
  the first analysed frame, after the run started.
* **`imageio-ffmpeg` / PyAV** — `iio.imwrite("x.mp4", ...)` silently dispatches
  to the *tifffile* plugin and fails with `TiffWriter.write() got an unexpected
  keyword argument 'fps'`, naming neither the format nor the missing package.
  The movie cell falls back to GIF.

* **`napari-ome-zarr`** — declared, absent. Does not block acquisition; it is
  the napari reader for the stores `OmeZarrWriter` writes, so without it the
  output has to be opened with `zarr` directly (which the writer notebooks do).

Also absent: the pertzlab StarDist server (`izbniesen.izb.unibe.ch:8001`, not
reachable from here) and the `virtual_microscope` package.

To get the full set:

```
conda activate py313
pip install ome-writers napari-ome-zarr cellpose imageio-ffmpeg
```

`ome-writers` was installed and the environment re-checked afterwards
(`useq`, `pydantic` 2.11.9, `zarr` 3.1.5, `inscoper_api`, `faro`, `napari` all
still import; `useq` still enumerates). The other three are untested here.

---

## Channels

This configuration exposes one config group, `Channel`, whose members are the
`.cbc` files in `site.CHANNELS_DIR`:

```
395 WF  405  405b  470 WF Quad FRAP  470 WF  488  488b
550 WF Quad FRAP  550 WF  638b  640 WF  all lasers
TL WF Quad FRAP  TL WF  [TIRF] 405  [TIRF] 488  [TIRF] 638
```

**None** of the names the originals use resolve here — `miRFP`, `mScarlet3`,
`mCitrine`, `CyanStim`, `mTurquoise`, `mRuby2`, `phase-contrast`, `DAPI`,
`FITC`. `utils.validate_hardware` reports them as missing, so each notebook
re-maps its biology onto `site.IMAGING_CHANNELS`
(`470 WF`, `550 WF`, `640 WF`, `395 WF`, `TL WF`). All five snap with real
signal; `640 WF` is brightest on the current sample.

**Prefer plain `Channel` over `PowerChannel`.** An Inscoper `.cbc` already
selects its wavelength *and* its power — `470 WF` ships
`Lumencor_EpiFluorescence-470 Intensity (%) = 25` — so `Channel(config=...,
exposure=...)` is fully specified. `site.POWER_PROPERTIES` maps every channel to
its `(device, property)` for when you do want `PowerChannel`, but note this
bridge does not enumerate device properties, so `validate_events` **cannot
range-check a power** it is handed: a wrong value fails at the hardware, not at
validation.

`StimTreatment.stim_exposure_list` and `stim_power` are **bookkeeping** on this
microscope. The FRAP burn takes its laser line and power from the designated
channel's `.cbc`, its duration from the scan, and its dose from the repetition
count — so `stim_exposure_list=200` does not make the burn 200 ms long. Keep
`stim_channel_name` pointing at the FRAP channel so the recorded light path and
the fired one agree.

---

## Bugs and traps found while porting

Recorded because each cost real time, and each is still there.

1. **`for … else` in the existing `cell_migration_inscoper.ipynb`.** The FRAP
   preflight was written as `for problem in mic.frap_preflight(): print(...)` /
   `else: print("clear")`. A `for/else` runs its `else` unless the loop
   *breaks*, so it printed "FRAP preflight clear." **every time**, including
   when the preflight had complaints. Now an `assert`.

2. **`generate_df_acquire` divides by zero.** It computes
   `time_between_timesteps // time_per_fov` and then divides by it, so an
   interval *shorter* than the per-FOV cost raises
   `ZeroDivisionError: float floor division by zero` from inside `faro`, with
   nothing in the message about timing. `site.check_fov_timing()` refuses first,
   and warns when the FOVs will be split into sequential batches (which silently
   multiplies the effective interval).

3. **The two APIs name the phase column differently.** `generate_df_acquire(
   phase_name=...)` writes a column called **`phase`**;
   `RTMSequence(rtm_metadata={"phase_name": ...})` writes **`phase_name`**.
   Grouping on the wrong one is a `KeyError` at analysis time, long after the
   run.

4. **`StimUp` reads its fraction from the constructor, not from metadata** —
   unlike `StimPercentageOfCell`, which reads
   `metadata["stim_cell_percentage"]`. The original `25_` notebook sets
   `stim_cell_percentage` in `rtm_metadata` *and* instantiates `StimUp()` with
   no argument, so the metadata is recorded while the stimulator quietly uses
   its `0.2` default. Now passed explicitly, so the recorded and fired values
   are the same number.

5. **A dose ramp cannot be expressed through `faro` on FRAP.** The original
   `stim_ramp_dfacquire.ipynb` ramps `stim_exposure_list` over
   `range(0, 900, 10)`. On a lamp that is a dose ramp; here the exposure field
   does not reach the burn at all, so the ramp writes an increasing column while
   every burn delivers an identical dose — and analysis then reports a
   dose-response against a dose that never varied. The bridge *would* honour
   `event.metadata["frap"]["repetitions"]`, but faro's transport is
   `SLMImage(data, device, exposure)` (see `faro/core/_useq_compat.py` and
   `Controller._build_stim_slm`) with no field for it. The Inscoper variant
   ramps **frequency** instead — the number of burns, which is just
   `stim_timestep` — and names the treatments `"…every2f + stimfreq"` so the
   record cannot be misread as a pulse-width ramp.

6. **`stim_mask/` is not written when there is no segmentator.** Masks are
   stored by the pipeline's analysis worker, which only runs when there is
   something to segment. `22_line_stimulation` has `segmentators=None`, so no
   mask reaches disk even though it *was* computed and fired. `stim/` holds the
   camera frames captured during each burn, and that is the record — so the
   Inscoper variant verifies the sweep two ways: recomputed offline from
   `StimLine` (a pure function of the timestep) and measured from the `stim/`
   readout frames, which is the only one that proves light landed where the mask
   said.

7. **`StimLine` needs `metadata["stim"] = True`.** It returns an all-zero mask
   when that key is falsy, so a pricing loop that omits it prices *N* empty
   masks and reports that everything fits.

8. **`generate_exp_data_from_tracks` raises on an empty `tracks/`.**
   `ValueError: No objects to concatenate` — reached whenever a pipeline has no
   tracker. Not called in `22_`.

9. **Frames come back `int64`**, and `getPixelSizeUm()` returns `0.0`, so
   anything deriving a field of view from the pixel size gets zero.

---

## Running one

```
conda activate py313          # the only env with inscoper_api + faro + napari
cd experiments/21_cell_migration
jupyter lab cell_migration_inscoper.ipynb
```

The working directory must be the notebook's folder (see trap 1). Then, in
order: run the load cell, check the printed summary, click **MDA** in
napari-micromanager at least once and pick FOV positions — otherwise
`generate_fov_positions` raises a `KeyError` saying the widget is not
registered — then run the pricing cell before the run cell.

Every notebook ends by clearing the camera ROI and calling
`mic.post_experiment()`. That is a **full teardown** on this adapter — it closes
the bridge — so a second run in the same kernel needs a fresh
`site.make_microscope()`.

### Rehearsing without the GUI

Each notebook has a `USE_GUI_FOVS = True` switch. Set it `False` to use
`site.fovs_here(mic, n)` — *n* copies of the field the stage is on now — which
exercises the multi-position axis without moving off your sample.

For a rehearsal on *distinct* fields, pass a step:
`site.fovs_here(mic, 2, step_um=150)`. That only became meaningful on
2026-08-28: `xAxis`/`yAxis` are the only axes in this configuration that
declare no `Min`, `InscoperDeviceManagerV2.get_min_max` started `driver_min`
at `+inf` instead of `-Double.MAX_VALUE`, and every XY target was therefore
clamped to `Double.MAX_VALUE` — so before that a stepped list acquired one
field *n* times. The focus drive declares real limits, which is why z was
never affected.

To run a whole notebook headless, with napari stubbed and the time plans
shortened:

```
python _inscoper_tools/nbtest.py     _inscoper_tools/21_cell_migration_inscoper.py     N_FRAMES=6 TIME_BETWEEN_TIMESTEPS=8 USE_GUI_FOVS=False
```

It runs each `# %%` cell in order in one namespace, prints what the notebook
itself printed (the C++ log spam is filtered), and reports `n/N cells ok`. Each
`NAME=VALUE` argument rewrites a top-level `NAME = ...` assignment before the
run, which is how a 120-frame experiment becomes a 6-frame smoke test without
editing the notebook.

### Editing a notebook

The `.ipynb` files are **generated**. Each one has its source next to it in
[`_inscoper_tools/`](_inscoper_tools) as an annotated `.py` (`# %%` /
`# %% [markdown]`), which is what to edit — it reviews as a diff, and
`nbtest.py` runs it directly. Regenerate with:

```
python _inscoper_tools/py2nb.py     _inscoper_tools/21_cell_migration_inscoper.py     21_cell_migration/cell_migration_inscoper.ipynb
```

---

## `inscoper_site.py` reference

| | |
|---|---|
| `check_license()` | confirm `inscoper.lic` is where the bridge will look |
| `make_microscope(roi=…, frap_channel=…)` | load, crop, designate FRAP, preflight, report |
| `set_roi(mic, n)` / `clear_roi(mic)` | Hamamatsu SUBARRAY crop (`mmc.setROI` is not the route) |
| `designate_frap(mic, name)` | name the stim channel + full preflight; `[]` means clear |
| `price_mask(mic, mask)` | what firing this mask costs, or why it is refused |
| `price_full_field(mic)` | the `StimWholeFOV` case, with the table on refusal |
| `probe_stim_mask(mic, seg, stim, …)` | snap → segment → build mask (with tracks) → price |
| `here(mic)` / `at(mic, dx, dy, dz)` / `fovs_here(mic, n)` | positions relative to now |
| `goto_good_field(mic)` | drive to `GOOD_XY_UM` at `FOCUS_UM` |
| `check_fov_timing(interval, per_fov, n)` | refuse a plan `generate_df_acquire` cannot express |
| `make_writer(path)` | `OmeZarrWriter` if importable, else `TiffWriter` |
| `make_segmentator(multichannel=…)` | `CellposeV4` if importable, else threshold |
| `open_napari(mic)` / `relink_napari` / `unlink_napari` | the napari-micromanager link |
| `describe(mic)` | everything to confirm before acquiring |

Constants: `CONFIG_DIR`, `CHANNELS_DIR`, `CAMERA_NAME`, `CHANNEL_GROUP`,
`FOCUS_UM` (2363), `GOOD_XY_UM` (−5402, −601), `IMAGING_CHANNELS`,
`FRAP_CHANNEL` (`all lasers`), `POWER_PROPERTIES`, `SCAN_PATH_BUDGET_PX`
(36,000), `INTERPOINT_DISTANCE_PX` (4), `MAX_FULL_FIELD_PX` (352),
`FULL_FOV_NOTE`. Each is overridable by environment variable — see the
docstrings.

---

## What is still open

Written down because each is a real gap, not a rough edge, and because "the
notebook runs" is a much weaker claim than "the experiment works".

**Dose is unverified.** Every burn in this work was aimed and priced, never
measured. All the pixel evidence is about *where* the light went; none is about
*how much*. A bleach curve on a fluorescent sample is the missing measurement,
and until it exists no notebook here can honestly report a dose-response.

**Repetitions cannot be reached through faro.** The dose knob the FRAP path
actually has is `metadata["frap"]["repetitions"]`, which the bridge honours.
faro's stimulation transport is `SLMImage(data, device, exposure)` — see
`faro/core/_useq_compat.py` and `Controller._build_stim_slm` — and has no field
for it. Consequences:

* `stim_exposure` / `stim_power` are bookkeeping on this microscope. They are
  written into `exp_data.parquet` and reach nothing.
* A pulse-width ramp is inexpressible. `stim_ramp_dfacquire_inscoper.ipynb`
  ramps *frequency* instead and names its treatments accordingly, which is a
  different perturbation (same peak, different duty cycle) — deliberately, and
  labelled so, rather than a ramp that silently does not ramp.

Adding an optional `repetitions` to the faro→bridge transport is the change
that would close this, and it is small: the bridge side already exists.

**`stim_mask/` is not written when a pipeline has no segmentator.** Masks are
stored by the analysis worker, which only runs when there is something to
segment. `line_stimulation_inscoper.ipynb` verifies its sweep from `stim/`
readout frames instead. Worth fixing upstream — the mask exists, it is simply
not on the path that stores it.

**Whole-field stimulation costs a field.** The `11_*` notebooks crop to 256 px
so `StimWholeFOV` fits one fire. Splitting a large region across several fires
would restore the full frame at the cost of scan time, and `plan_mask` already
refuses in a way that says so; nothing implements the split yet.

**One microscope, one sample.** Every number here — the 4 px pitch, the 352 px
ceiling, the focus, the channel list — is this installation on 2026-08-26.
`inscoper_site.py` is the file to re-read on another rig, and `describe(mic)`
prints what a fresh machine actually reports.

**Untested substitutions.** `cellpose`, `imageio-ffmpeg` and `napari-ome-zarr`
are still absent, so `site.make_segmentator()` has only ever been exercised on
its threshold fallback, and the multi-channel path (`ProjectedSegmentator`,
which max-projects) is a stand-in for Cellpose's own channel handling, not an
equivalent to it.
