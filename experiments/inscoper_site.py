"""Site configuration and helpers shared by the ``*_inscoper`` notebooks.

Every path, device name, channel and safety check that describes *this*
Inscoper installation lives here, so porting the notebook set to another
Inscoper microscope is one file to edit instead of one block per notebook.
It plays the same role for ``rtm-pymmcore``/``faro`` that
``inscoper_useq/scripts/frap_config.py`` plays for the standalone
``inscoper_useq`` scripts, and the numbers below come from the same place:
the ``useq_compat`` harness (``scripts/useq_compat/_harness.py``), re-read
against the hardware on 2026-08-26.

Anything here can be overridden by an environment variable
(``INSCOPER_CONFIG_DIR``, ``INSCOPER_CHANNELS_DIR``, ``INSCOPER_CAMERA``,
``INSCOPER_FRAP_CHANNEL``, ``INSCOPER_FOCUS``), which keeps a temporary
change out of version control.

Why this module exists
----------------------
Five things about the Inscoper stack differ enough from the Micro-Manager
microscopes the original notebooks were written for that porting one by
search-and-replacing its ``mic = ...`` line produces something that *runs*
and is *wrong*:

1. **The licence is resolved against the current working directory.**
   ``Bridge`` looks for ``inscoper.lic`` in ``os.getcwd()``. Without it the
   configuration still "loads" -- no exception -- but no devices come up:
   ``getImageWidth()`` answers 0, the stage reads ``(0, 0, 0)``, and every
   snap is empty. :func:`check_license` turns that into a sentence, and
   :func:`make_microscope` refuses to hand back a 0x0 microscope.

2. **The stimulation device is a FRAP galvo, not a DMD.** Its calibration
   maps camera pixels to galvo units and the *firmware* applies it per
   point, so masks must arrive in camera pixels. Calling
   ``mic.calibrate_dmd(...)`` on this system is what *breaks*
   photoactivation -- it would transform the mask a second time. See
   :func:`designate_frap`.

3. **A FRAP burn is bounded by scan path, not by mask area.** One fire
   covers ~36,000 px of scan path (~5 s at ~7200 px/s), and this rig's row
   pitch is 4 px. ``StimWholeFOV`` therefore cannot be fired as-is on any
   field wider than ~352 px -- see :data:`FULL_FOV_NOTE` and
   :func:`price_mask`.

4. **Stage coordinates are absolute.** ``stage_positions=[(0, 0, 0)]`` in
   the originals means "the machine origin", a full-travel move away from
   whatever is under the objective -- and on the focus axis, a move towards
   it. :func:`here`, :func:`at` and :func:`fovs_here` express positions as
   small offsets from wherever the stage already is.

5. **The environment is not the one ``pyproject.toml`` describes.**
   ``ome_writers`` was declared but absent until it was installed on
   2026-08-26; ``cellpose`` still is, along with ``imageio-ffmpeg`` and
   ``napari-ome-zarr``. The pattern matters more than the list, because
   ``OmeZarrWriter`` did not fail at construction: it failed inside
   ``init_stream``, i.e. *after* ``run_experiment`` had started, as the
   run's ``fatal_error``. :func:`make_writer` and :func:`make_segmentator`
   decide up front and say what they chose, so a missing package costs a
   printed line instead of an acquisition.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# -- Installation constants -------------------------------------------------

CONFIG_DIR = os.environ.get(
    "INSCOPER_CONFIG_DIR",
    r"C:\Users\InscoperPasDev\Desktop\Remy\envUseq\FullSystemWithHamamatsu",
)
"""Inscoper system configuration folder (holds ``InterfaceFile.json``)."""

CHANNELS_DIR = os.environ.get(
    "INSCOPER_CHANNELS_DIR",
    r"C:\Inscoper CLAIRE\InscoperInterface-9.3.12\channels",
)
"""Folder holding the channel definitions (``.cbc`` files)."""

CAMERA_NAME = os.environ.get("INSCOPER_CAMERA", "Fusion_Right")
"""Main camera (``InterfaceFile.json`` -> cameras -> Name)."""

CHANNEL_GROUP = "Channel"
"""The only config group this configuration exposes."""

CAMERA_DEVICE = "Hamamatsu_Data"
"""Sub-device family carrying the Hamamatsu SUBARRAY (ROI) settings."""

FULL_FRAME_PX = 2304
"""Native frame of the Fusion camera on this rig."""

FOCUS_UM = float(os.environ.get("INSCOPER_FOCUS", "2363.0"))
"""In-focus z in micrometres, 20x objective. Read from the GUI 2026-08-26.

Only a fallback: :func:`here` prefers the drive's own read-back, which is
where the operator actually left it, and uses this to sanity-check it.

**Units.** ``useq`` speaks micrometres; ``SequenceBuilder`` multiplies
``event.z_pos`` by 1e3 on the way to the nanometre sub-device. So the number
here is the micrometre value the GUI shows, and a value 1000x too large is
what a units mix-up looks like -- which is why :data:`FOCUS_RANGE_UM` gates
it.
"""

GOOD_XY_UM = (-5402.0, -601.0)
"""A field known to be worth imaging, in micrometres. Recorded 2026-08-26.

Not used automatically -- :func:`here` reads the stage instead -- but it is
the position to drive back to when a run has wandered, and the reference for
"is the stage where I think it is?".
"""

FOCUS_RANGE_UM = (100.0, 10_000.0)
"""Plausible travel of the focus drive. Outside this is a units mistake."""

# Channel names as this configuration spells them, verified live against
# ``getAvailableConfigs("Channel")``.
ALL_CHANNELS = (
    "395 WF", "405", "405b", "470 WF Quad FRAP", "470 WF", "488", "488b",
    "550 WF Quad FRAP", "550 WF", "638b", "640 WF", "all lasers",
    "TL WF Quad FRAP", "TL WF", "[TIRF] 405", "[TIRF] 488", "[TIRF] 638",
)

IMAGING_CHANNELS = ("470 WF", "550 WF", "640 WF", "395 WF", "TL WF")
"""Widefield imaging channels, in the order the notebooks reach for them.

The originals name pertzlab channels (``miRFP``, ``mScarlet3``,
``mCitrine``, ``CyanStim``, ``phase-contrast``). Nothing resolves those here
-- ``utils.validate_hardware`` reports them as missing -- so each
``*_inscoper`` notebook maps its biology onto these.
"""

FRAP_CHANNEL = os.environ.get("INSCOPER_FRAP_CHANNEL", "all lasers")
"""The *stimulation* channel -- the one that opens the FRAP light path.

It cannot be picked by name. Every stock channel called ``... WF Quad FRAP``
is an *imaging* channel looking through the quad FRAP dichroic and ships
``iLas2::Shutter = 0``, which scans the galvo in the dark.
:func:`designate_frap` validates the choice against the configuration's own
``Ilda.frap.ActiveModeChannel``.
"""

POWER_PROPERTIES: dict[str, tuple[str, str]] = {
    "all lasers": ("iLas2", "BluePower"),
    "470 WF": ("Lumencor_EpiFluorescence", "470 Intensity (%)"),
    "550 WF": ("Lumencor_EpiFluorescence", "550 Intensity (%)"),
    "640 WF": ("Lumencor_EpiFluorescence", "640 Intensity (%)"),
    "395 WF": ("Lumencor_EpiFluorescence", "395 Intensity (%)"),
    "TL WF": ("NikonTi2", "TL Lamp Intensity (%)"),
}
"""Where a ``PowerChannel``'s ``power`` is pushed, per channel config.

Assigned to ``mic.POWER_PROPERTIES`` by :func:`make_microscope`. Without an
entry, ``InscoperMicroscope.resolve_power`` raises rather than dropping the
requested power silently.

**Prefer a plain ``Channel`` on this microscope.** An Inscoper ``.cbc``
already selects its wavelength *and* its power, so
``Channel(config=..., exposure=...)`` is fully specified and needs none of
this. Note too that this bridge does not enumerate device properties
(``getDevicePropertyNames`` answers ``[]`` for every device but ``Core`` and
the camera), so ``utils.validate_hardware`` cannot range-check a power it is
handed: a wrong value fails at the hardware, not at validation.
"""

EXPOSURE_LIMITS_MS = (0.001, 999_999.0)
"""Camera exposure limits, as the bridge reports them for ``Exposure``."""

# -- FRAP scan budget -------------------------------------------------------

SCAN_PATH_BUDGET_PX = 36_000
"""Scan path one FRAP fire covers, in px (~5 s at ~7200 px/s)."""

INTERPOINT_DISTANCE_PX = 4
"""Row pitch this rig's ILDA device reports. Not a default -- measured."""

MAX_FULL_FIELD_PX = 352
"""Largest square field that can be *entirely* stimulated in one fire.

At the 4 px pitch above, a filled square hatches into rows whose total scan
path passes :data:`SCAN_PATH_BUDGET_PX` between 352 px and 384 px:
352 -> 31,236 px (~4.3 s), 384 -> refused.
"""

FULL_FOV_NOTE = """\
Whole-field photoactivation through the FRAP galvo does not fit in one fire
on the native 2304x2304 frame. Cost of a *filled square*, measured with
plan_mask (2026-08-26); this rig's row pitch is 4 px:

    field    pitch 4    pitch 8    pitch 16   pitch 32   pitch 64   pitch 128
     256 px   2.3 s      1.2 s      0.6 s      0.3 s      0.2 s      0.1 s
     352 px   4.3 s      -          -          -          -          -
     384 px   REFUSED    2.6 s      1.3 s      0.7 s      0.4 s      0.2 s
     512 px   REFUSED    4.6 s      2.3 s      1.2 s      0.6 s      0.3 s
    1024 px   REFUSED    REFUSED    REFUSED    4.7 s      2.4 s      1.3 s
    2304 px   REFUSED    REFUSED    REFUSED    REFUSED    REFUSED    6.1 s*

    (* 342 points -- fits the point budget but 43,630 px of path, i.e. over
       the 36,000 px one fire covers. Coarse enough to be a dot grid, not a
       flood.)

plan_mask REFUSES rather than truncating, so an over-budget mask is an
exception at the first stim event, not a half-burnt region you find in
analysis. Two honest ways to want "full field" here:

  a) crop the camera -- set_roi(mic, 256) or set_roi(mic, 352) -- so the
     whole field really is stimulable at the device pitch. This is what the
     full-FOV *_inscoper notebooks do, and it speeds up readout as well; or
  b) stimulate patterned regions instead, which is what a galvo is for and
     is cheaper: 20 cells of r=30 px on the full 2304 frame costs 2.3 s.
"""


# -- Licence ----------------------------------------------------------------


def check_license(strict: bool = True) -> Path | None:
    """Confirm ``inscoper.lic`` is where the bridge will look for it.

    The bridge resolves the licence against ``os.getcwd()`` -- for a
    notebook, the folder the ``.ipynb`` lives in. A missing licence does not
    raise: the configuration loads with **no devices**, and the first
    visible symptom is ``getImageWidth() == 0`` several cells later.
    Checking first costs nothing and saves that hunt.
    """
    lic = Path(os.getcwd()) / "inscoper.lic"
    if lic.is_file():
        return lic
    message = (
        f"No inscoper.lic in the working directory ({os.getcwd()}).\n"
        "The Inscoper bridge resolves the licence against the CWD, and "
        "without it loadSystemConfiguration() succeeds but loads no devices: "
        "the image size reads 0x0, the stage reads (0, 0, 0) and every snap "
        "is empty.\n"
        "Copy a licence next to this notebook, or os.chdir() to a folder "
        "that has one."
    )
    if strict:
        raise FileNotFoundError(message)
    print("WARNING:", message)
    return None


# -- Microscope -------------------------------------------------------------


def make_microscope(
    *,
    roi: int | tuple[int, int, int, int] | None = None,
    frap_channel: str | None = FRAP_CHANNEL,
    channel_group: str = CHANNEL_GROUP,
    strict_license: bool = True,
    verbose: bool = True,
):
    """Load the Inscoper bridge behind an ``InscoperMicroscope``.

    Does the four things every notebook would otherwise repeat: check the
    licence, load the configuration, apply a camera ROI, and designate plus
    preflight the FRAP stimulation channel.

    Parameters
    ----------
    roi:
        ``None`` keeps the full 2304x2304 frame. An ``int`` *n* takes a
        centred ``n x n`` subarray -- the usual choice: it makes the field
        small enough for whole-field FRAP (see :data:`FULL_FOV_NOTE`) and
        cuts per-frame readout, so a 15 s time-lapse over several FOVs
        actually keeps up. A 4-tuple is ``(x, y, width, height)`` in sensor
        pixels.
    frap_channel:
        Channel photoactivation fires with, or ``None`` to skip designation
        (an imaging-only notebook needs none).
    """
    check_license(strict=strict_license)

    from faro.microscope.Inscoper import InscoperMicroscope

    mic = InscoperMicroscope(
        config_folder=CONFIG_DIR,
        channels_folder=CHANNELS_DIR,
        main_camera=CAMERA_NAME,
        channel_group=channel_group,
    )
    mic.POWER_PROPERTIES = dict(POWER_PROPERTIES)

    if mic.mmc.getImageWidth() == 0:
        raise RuntimeError(
            "The configuration loaded but the camera reports a 0x0 frame, so "
            "no devices came up. Three causes, in order of likelihood: the "
            "licence is not in the working directory (see check_license), "
            "another process already holds the hardware, or the controller "
            "is switched off. Nothing downstream of here can work -- a snap "
            "returns nothing and the stage reads (0, 0, 0) -- so this refuses "
            "rather than handing back a microscope that silently does nothing."
        )

    if roi is not None:
        set_roi(mic, roi, verbose=verbose)

    if frap_channel:
        designate_frap(mic, frap_channel, verbose=verbose)

    if verbose:
        describe(mic)
    return mic


def set_roi(mic, roi: int | tuple[int, int, int, int], *, verbose: bool = True) -> None:
    """Apply a Hamamatsu SUBARRAY crop, the way the GUI's ROI control does.

    ``mmc.setROI`` is not the route on this bridge -- the crop lives in
    firmware sub-devices on the camera (``SUBARRAY HPOS1`` and friends), and
    setting those is what changes ``getImageWidth()``.

    The DMD shim's dimensions are updated to match, because on a FRAP system
    they mirror the camera FOV: leaving them stale is what makes a stim mask
    arrive the wrong shape.
    """
    from inscoper_api import SubDeviceId

    full_w = mic.mmc.getImageWidth()
    full_h = mic.mmc.getImageHeight()
    if isinstance(roi, int):
        w = h = roi
        x = max(0, (full_w - w) // 2)
        y = max(0, (full_h - h) // 2)
    else:
        x, y, w, h = roi

    # The subarray is quantised in steps of 4 px. Rounding here, rather than
    # letting the firmware round silently, keeps getImageWidth() equal to
    # what was asked for -- which every mask shape downstream assumes.
    x, y = (int(x) // 4) * 4, (int(y) // 4) * 4
    w, h = (int(w) // 4) * 4, (int(h) // 4) * 4

    for sub, val in (
        ("SUBARRAY HPOS1", str(x)),
        ("SUBARRAY VPOS1", str(y)),
        ("SUBARRAY HSIZE1", str(w)),
        ("SUBARRAY VSIZE1", str(h)),
        ("SUBARRAY MODE1", "ON"),
    ):
        mic.mmc.setValue(SubDeviceId(CAMERA_DEVICE, sub), val)

    _sync_frame_size(mic)
    if verbose:
        print(
            f"[inscoper_site] ROI {mic.image_width}x{mic.image_height} "
            f"at ({x}, {y}) of {full_w}x{full_h}"
        )


def clear_roi(mic, *, verbose: bool = True) -> None:
    """Return the camera to its full frame."""
    from inscoper_api import SubDeviceId

    mic.mmc.setValue(SubDeviceId(CAMERA_DEVICE, "SUBARRAY MODE1"), "OFF")
    _sync_frame_size(mic)
    if verbose:
        print(f"[inscoper_site] ROI cleared: {mic.image_width}x{mic.image_height}")


def _sync_frame_size(mic) -> None:
    """Make the microscope's cached frame size agree with the camera."""
    mic.image_width = mic.mmc.getImageWidth()
    mic.image_height = mic.mmc.getImageHeight()
    dmd = getattr(mic, "dmd", None)
    if dmd is not None:
        dmd.width = mic.image_width
        dmd.height = mic.image_height
        dmd.sample_mask_on = np.full(
            (mic.image_height, mic.image_width), 255, dtype=np.uint8
        )
        dmd.sample_mask_off = np.zeros(
            (mic.image_height, mic.image_width), dtype=np.uint8
        )


# -- FRAP photoactivation ---------------------------------------------------


def designate_frap(mic, name: str = FRAP_CHANNEL, *, verbose: bool = True) -> list[str]:
    """Name the stimulation channel and run the full preflight.

    Returns the list of problems; empty means clear. Every problem this can
    report corresponds to a failure that produces **no light and no error**
    -- the beam scans with the shutter shut, the calibration does not match
    the current filter cube, or the fire is refused. In a 24 h feedback
    experiment each one looks like a successful run until analysis.
    """
    problems = list(mic.designate_frap_channel(name))
    problems += [p for p in mic.frap_preflight() if p not in problems]
    if verbose:
        if problems:
            for p in problems:
                print("FRAP PREFLIGHT:", p)
        else:
            print(f"[inscoper_site] FRAP preflight clear (channel {name!r}).")
    return problems


def frap_context(mic, pitch: int | None = None):
    """The hatching context the bridge will actually use at fire time.

    Row pitch comes from the ILDA device, as IIS reads it -- not from
    ``FrapRoiContext``'s own default, which would turn a segmented field
    into thousands of rows and a refusal.
    """
    from inscoper_useq.basic_element.roi import FrapRoiContext

    if pitch is not None:
        return FrapRoiContext(interpoint_distance=pitch, density_index=1)
    frap = mic.mmc.frap
    return FrapRoiContext(
        interpoint_distance=frap._interpoint_distance(),
        density_index=getattr(frap.frap_scanner_device, "density_index", None),
    )


def price_mask(mic, mask, *, pitch: int | None = None, verbose: bool = True):
    """Say what firing *mask* would cost, before an experiment commits to it.

    A FRAP burn is bounded by scan **path**, not by mask area, and the budget
    is one ~36,000 px pass (~5 s). ``plan_mask`` refuses an over-budget
    region rather than truncating it, so pricing a representative mask here
    turns a mid-run exception into a number you can design against.

    Returns the plan, or ``None`` if the region would be refused (the reason
    is printed).
    """
    from inscoper_useq.basic_element.roi import plan_mask

    ctx = frap_context(mic, pitch)
    try:
        plan = plan_mask(np.asarray(mask), ctx)
    except Exception as exc:  # noqa: BLE001 - the refusal *is* the answer
        if verbose:
            print(f"FRAP would REFUSE this mask: {exc}")
        return None
    if verbose:
        print(f"FRAP plan: {plan.describe()}")
    return plan


def sample_disc(mic, radius_fraction: float = 0.125) -> np.ndarray:
    """A centred disc the size of a plausible stim region, for pricing."""
    h, w = mic.mmc.getImageHeight(), mic.mmc.getImageWidth()
    yy, xx = np.ogrid[:h, :w]
    r = int(min(h, w) * radius_fraction)
    return ((xx - w // 2) ** 2 + (yy - h // 2) ** 2) <= r**2


def probe_stim_mask(
    mic,
    segmentator,
    stimulator,
    *,
    channel: str,
    exposure: float = 30.0,
    metadata: dict | None = None,
    verbose: bool = True,
):
    """Snap a frame, segment it, and price the mask the stimulator would fire.

    The closest thing to a dry run of the burn, and the only pre-flight that
    can answer the question that matters: *can the galvo fire what this
    pipeline will produce on this sample?* ``validate_events`` cannot -- the
    mask does not exist until a frame has been segmented -- and the first stim
    event is a bad place to find out, because ``plan_mask`` raises there.

    Synthesises the ``tracks`` frame the pipeline would normally supply.
    Several stimulators (``StimUp``, ``StimUpDown``, ``StimLine``) return an
    **empty** mask when ``tracks`` is ``None`` or empty, so pricing without it
    reports a free burn for a mask that would in fact be expensive -- a
    pre-flight that always passes is worse than none.

    Returns ``(labels, mask, plan)``; *plan* is ``None`` if the burn would be
    refused, and *mask* is ``None`` if nothing was segmented.
    """
    import pandas as pd

    mic.mmc.setConfig(CHANNEL_GROUP, channel)
    mic.mmc.setExposure(exposure)
    mic.mmc.snapImage()
    frame = mic.mmc.getImage()

    labels = segmentator.segment(frame)
    n = int(np.asarray(labels).max())
    if verbose:
        print(f"probe frame ({channel!r}, {exposure} ms): {n} object(s)")
    if n == 0:
        if verbose:
            print(
                "Nothing segmented, so there is no mask to price. Refocus "
                f"(the drive reads {here(mic)[2]:.0f} um; FOCUS_UM is "
                f"{FOCUS_UM}) or move to a field with cells."
            )
        return labels, None, None

    from skimage.measure import regionprops

    props = regionprops(np.asarray(labels))
    tracks = pd.DataFrame(
        {
            "timestep": 0,
            "label": [p.label for p in props],
            "particle": range(len(props)),
            "x": [p.centroid[0] for p in props],
            "y": [p.centroid[1] for p in props],
        }
    )

    mask, _ = stimulator.get_stim_mask(
        {"labels": labels}, metadata=metadata or {}, img=frame, tracks=tracks
    )
    lit = int(np.count_nonzero(mask))
    if verbose:
        print(f"stim mask: {lit} px lit of {np.asarray(mask).size}")
    if lit == 0:
        if verbose:
            print(
                "The stimulator returned an EMPTY mask, so nothing would "
                "fire. Check the metadata it reads (StimPercentageOfCell "
                "wants 'stim_cell_percentage'; StimUp takes its fraction as a "
                "constructor argument and is gated by a falsy 'stim_fov')."
            )
        return labels, mask, None

    plan = price_mask(mic, mask, verbose=verbose)
    if plan is None and verbose:
        print(
            "\nFix this BEFORE starting the run: stimulate a smaller fraction "
            "of each cell, crop further with set_roi, or move to a sparser "
            "field. plan_mask raises at the first stim event, so a run started "
            "in this state dies partway through."
        )
    return labels, mask, plan


def price_full_field(mic, *, verbose: bool = True):
    """Price a whole-field stim mask -- the ``StimWholeFOV`` case.

    Prints :data:`FULL_FOV_NOTE` when the field is too wide, since that is
    the one refusal a notebook cannot work around by trying again.
    """
    h, w = mic.mmc.getImageHeight(), mic.mmc.getImageWidth()
    plan = price_mask(mic, np.ones((h, w), bool), verbose=verbose)
    if plan is None and verbose:
        print()
        print(FULL_FOV_NOTE)
    return plan


# -- Stage ------------------------------------------------------------------


def here(mic) -> tuple[float, float, float]:
    """Current ``(x, y, z)`` in micrometres -- the anchor for every position.

    ``useq`` stage coordinates are absolute and the builder writes them
    straight to the axis, so a literal ``(0, 0, 0)`` does not mean "stay
    here": it commands the machine origin. Anchoring on the read-back keeps
    a notebook on the field the operator focused on.
    """
    x, y = mic.mmc.getXYPosition()
    z = mic.mmc.getZPosition()
    if not (FOCUS_RANGE_UM[0] <= z <= FOCUS_RANGE_UM[1]):
        print(
            f"WARNING: the focus drive reads {z:g} um, outside the plausible "
            f"travel range {FOCUS_RANGE_UM} um for this stage. Falling back "
            f"to FOCUS_UM = {FOCUS_UM}. Re-read the Focus field in the GUI "
            "and set inscoper_site.FOCUS_UM if that is wrong."
        )
        z = FOCUS_UM
    return float(x), float(y), float(z)


def at(mic, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
    """A stage position *dx/dy/dz* micrometres from :func:`here`."""
    x, y, z = here(mic)
    return {"x": x + dx, "y": y + dy, "z": z + dz}


def fovs_here(mic, n: int = 1, step_um: float = 0.0) -> list:
    """*n* ``FovPosition`` objects around the current stage position.

    The stand-in for ``utils.generate_fov_positions(mic, fake_fovs=n)``,
    whose positions are all ``(0, 0, None)`` -- the machine origin. Use this
    to rehearse a notebook without picking FOVs in the GUI; use
    ``generate_fov_positions(mic, viewer=viewer)`` for a real run.

    ``step_um`` defaults to 0, i.e. every FOV is the same field. That is
    deliberate for a rehearsal: it exercises the multi-position axis without
    moving the stage off the field you focused on. Pass a real step when you
    want distinct fields -- the notebooks use 150 um, because
    ``getPixelSizeUm()`` answers 0.0 on this bridge and the FOV width cannot
    be derived from it.

    A step now actually moves the stage. Until 2026-08-28 it could not:
    ``xAxis`` and ``yAxis`` declare no ``Min`` in ``InterfaceFile.json`` (the
    only two axes in this configuration that do not -- ``Focus`` declares
    ``-500000``/``1e7``, which is why only XY was affected), and
    ``InscoperDeviceManagerV2.get_min_max`` started ``driver_min`` at
    ``+inf`` where IIS starts at ``-Double.MAX_VALUE``, so
    ``max(driver_min, config_min)`` came back ``+inf`` and
    ``NumberDevice.convert_value`` clamped every XY target to
    ``Double.MAX_VALUE``. A stepped FOV list acquired one field *n* times.
    So ``step_um=0`` is a choice again, not the only honest value.
    """
    from faro.core.utils import FovPosition

    x, y, z = here(mic)
    return [
        FovPosition(x=x + i * step_um, y=y, z=z, name=f"here{i}") for i in range(n)
    ]


def check_fov_timing(
    time_between_timesteps: float, time_per_fov: float, n_fovs: int = 1
) -> None:
    """Refuse a time plan ``generate_df_acquire`` cannot express.

    ``generate_df_acquire`` computes ``time_between_timesteps //
    time_per_fov`` and then divides by it, so an interval *shorter* than the
    per-FOV cost gives 0 and raises ``ZeroDivisionError: float floor division
    by zero`` from inside ``faro``, with nothing in the message about timing.
    It is an easy setting to reach on this microscope, where shrinking the
    interval is the first thing anyone tries.

    Also warns when the FOVs cannot fit in one interval, which is legal --
    ``generate_df_acquire`` splits them into sequential batches -- but means
    the effective interval per FOV is a multiple of what was asked for.
    """
    if time_per_fov <= 0:
        raise ValueError("time_per_fov must be positive")
    per_interval = int(time_between_timesteps // time_per_fov)
    if per_interval < 1:
        raise ValueError(
            f"time_between_timesteps={time_between_timesteps} s is shorter "
            f"than time_per_fov={time_per_fov} s, so no FOV fits in one "
            "interval. generate_df_acquire would fail with a bare "
            "ZeroDivisionError. Either raise the interval above "
            f"{time_per_fov} s, or lower time_per_fov (shorter exposures, a "
            "smaller ROI via set_roi) to match what a FOV really costs."
        )
    if n_fovs > per_interval:
        batches = -(-n_fovs // per_interval)
        print(
            f"[inscoper_site] {n_fovs} FOVs at {time_per_fov} s each do not "
            f"fit in a {time_between_timesteps} s interval ({per_interval} "
            f"fit). They will run as {batches} sequential batches, so each "
            f"FOV is revisited every {batches * time_between_timesteps:.0f} s, "
            "not every "
            f"{time_between_timesteps:.0f} s."
        )


def goto_good_field(mic, *, verbose: bool = True) -> None:
    """Drive to :data:`GOOD_XY_UM` at :data:`FOCUS_UM`.

    For when a run has wandered and the notebook needs to be back on a field
    known to be worth imaging. Absolute, and therefore the one function here
    that moves the stage a long way -- call it deliberately.
    """
    x, y = GOOD_XY_UM
    mic.mmc.setXYPosition(x, y)
    mic.mmc.setZPosition(FOCUS_UM)
    if verbose:
        print(f"[inscoper_site] moved to x={x} y={y} z={FOCUS_UM} um")


# -- Pipeline pieces that depend on what is installed -----------------------


def make_writer(path: str, **kwargs):
    """``OmeZarrWriter`` when it can be built, else ``TiffWriter``.

    ``ome_writers`` **is** installed in the ``py313`` environment as of
    2026-08-26 -- it was missing until then, which is why this function
    exists -- so the OME-Zarr path is the one taken. The fallback stays
    because of *how* the absence used to present: ``OmeZarrWriter``
    constructs fine without the package and fails inside ``init_stream``,
    i.e. **after** ``run_experiment`` has started, surfacing as the run's
    ``fatal_error`` with a bare ``ModuleNotFoundError`` that names neither
    the writer nor the package. Nothing is acquired.

    Deciding here turns that into a printed line and a working run on any
    environment that has not had the package added yet.
    """
    from faro.core.writers import TiffWriter

    try:
        import ome_writers  # noqa: F401
        from faro.core.writers import OmeZarrWriter

        return OmeZarrWriter(storage_path=path, **kwargs)
    except ImportError:
        print(
            "[inscoper_site] ome_writers is not installed, so OmeZarrWriter "
            "would fail once the run had already started. Using TiffWriter "
            "(raw/labels/particles as TIFFs, tracks as parquet). "
            "Install with: pip install ome-writers"
        )
        return TiffWriter(storage_path=path)


def has_ome_writers() -> bool:
    """Whether the OME-Zarr path is available at all."""
    try:
        import ome_writers  # noqa: F401

        return True
    except ImportError:
        return False


def _project_channels(image: np.ndarray) -> np.ndarray:
    """Reduce a ``(C, Y, X)`` stack to ``(Y, X)`` by max projection.

    Only needed by the fallback segmentator. A ``SegmentationMethod`` with
    ``use_channel=[0, 1]`` hands the segmentator a ``(C, Y, X)`` stack, and a
    segmentator that does not reduce C returns ``(C, Y, X)`` labels. The
    feature extractors then fail inside ``regionprops_table`` with *"Label and
    intensity image shapes must match, except for channel (last) axis"* --
    which names the symptom and not the cause. Cellpose reduces the channels
    itself, which is why the originals never hit this.
    """
    arr = np.asarray(image)
    return arr.max(axis=0) if arr.ndim == 3 else arr


class ProjectedSegmentator:
    """Wraps a 2D-only segmentator so it accepts a ``(C, Y, X)`` stack.

    Max-projects the channels and delegates, so the labels come back 2D and
    match what the feature extractors expect. A max projection is the crude
    choice -- it is *not* what Cellpose does with multiple channels -- so this
    exists to keep a notebook runnable end to end, not to reproduce a
    multi-channel segmentation.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ProjectedSegmentator({self.inner!r})"

    def segment(self, image: np.ndarray) -> np.ndarray:
        return self.inner.segment(_project_channels(image))

    def __getattr__(self, name):
        # Pass through anything else the pipeline asks of a segmentator
        # (validate_pipeline inspects signatures and class attributes).
        return getattr(self.inner, name)


def make_segmentator(*, multichannel: bool = False, **kwargs):
    """``CellposeV4`` when installed, else a threshold segmentator.

    The originals segment with Cellpose, or with a remote ImagingServerKit at
    ``http://izbniesen.izb.unibe.ch:8001`` -- a pertzlab host not reachable
    from here. Falling back keeps a notebook runnable end to end; swap the real
    segmentator back in for a real experiment.

    Parameters
    ----------
    multichannel:
        Pass ``True`` when the ``SegmentationMethod`` uses a *list* of
        ``use_channel`` values. Cellpose handles a ``(C, Y, X)`` stack itself;
        the fallback does not, so it is wrapped in
        :class:`ProjectedSegmentator`. Getting this wrong is not a silent
        failure -- it surfaces as a shape error in ``regionprops_table`` -- but
        it surfaces on the first *analysed* frame, i.e. after the run started.
    **kwargs:
        Forwarded to ``CellposeV4``. Ignored by the fallback, which takes none
        of Cellpose's parameters.
    """
    try:
        from faro.segmentation.cellpose_v4 import CellposeV4

        return CellposeV4(**kwargs)
    except ImportError:
        from faro.segmentation.threshold import SegmentatorThreshold

        note = (
            "[inscoper_site] cellpose is not installed -- falling back to "
            "SegmentatorThreshold(threshold=0.1)"
        )
        inner = SegmentatorThreshold(threshold=0.1)
        if multichannel:
            print(
                note + ", max-projecting the channel axis so the labels come "
                "back 2D. Install with: pip install cellpose"
            )
            return ProjectedSegmentator(inner)
        print(note + ". Install with: pip install cellpose")
        return inner


# -- napari -----------------------------------------------------------------


def open_napari(mic, *, title: str | None = None):
    """Open napari with napari-micromanager bound to the Inscoper bridge.

    Returns ``(viewer, main_window)``.

    ``MainWindow(viewer, mmcore=...)`` is the documented form, but
    ``InscoperMicroscope.init_scope`` has already registered the bridge as
    the ``CMMCorePlus`` singleton, and the two-step assignment used here is
    what the existing Inscoper notebook does and what is known to work
    against this bridge.

    Pick FOV positions in the **MDA** dock widget after this; click the MDA
    button once, or ``utils.generate_fov_positions`` raises a ``KeyError``
    saying the widget is not registered.
    """
    import napari
    from napari_micromanager import MainWindow

    viewer = napari.Viewer(title=title or "Inscoper")
    with _z_moves_disarmed(mic.mmc):
        mm_wdg = MainWindow(viewer)
    mm_wdg._mmc = mic.mmc
    viewer.window.add_dock_widget(mm_wdg)
    return viewer, mm_wdg


@contextmanager
def _z_moves_disarmed(mmc):
    """Swallow focus commands for the duration of the block.

    ``pymmcore-widgets``' objective widget lowers the focus to 0 before an
    objective change and restores it afterwards (``_pre_change_hook`` /
    ``_post_change_hook`` in ``control/_objective_widget.py``). Building the
    widget calls ``setCurrentText``, which fires ``_on_combo_changed`` as
    though the objective had changed, so *opening napari* commands
    ``focus -> 0 -> previous``.

    That was invisible until 2026-08-27, because ``UseqBridge.setZPosition``
    staged the value and nothing moved. Now that the bridge drives the axis
    through a ``DEVICE_UPDATE`` recipe, the same startup would rack the
    objective through full travel and back with a sample in place. The
    objective is not changing here, so neither should the focus.

    A real objective change from the widget still gets its protective move:
    this only covers window construction.

    The XY clamp fixed on 2026-08-28 does not retire this guard, and it is
    worth saying why: that fix was in ``get_min_max``, and it only mattered
    for sub-devices whose driver reports no minimum. ``Focus`` reports real
    limits, so focus commands were never clamped -- the widget's
    ``focus -> 0 -> previous`` would travel the full drive. Patching this one
    name is enough because the widget calls ``setPosition(zdev, 0)`` and
    ``UseqBridge.setPosition`` routes straight to ``setZPosition``.
    """
    original = mmc.setZPosition

    def _refuse(val, *args, **kwargs):
        print(
            f"[inscoper_site] ignored a focus move to {val} um issued while "
            "the napari window was being built (objective-widget startup "
            "artefact, not a real objective change)"
        )

    mmc.setZPosition = _refuse
    try:
        yield
    finally:
        mmc.setZPosition = original


def relink_napari(viewer, mic):
    """Re-make the napari-micromanager live link after a run tore it down."""
    from napari_micromanager._core_link import CoreViewerLink

    return CoreViewerLink(viewer, mic.mmc)


def unlink_napari(mm_wdg) -> None:
    """Drop the live link, ignoring the case where there is none."""
    try:
        mm_wdg._core_link.cleanup()
    except Exception as exc:  # noqa: BLE001
        print(f"(no live link to clean up: {exc})")


# -- Reporting --------------------------------------------------------------


def describe(mic) -> None:
    """Print everything a notebook should confirm before it acquires."""
    mmc = mic.mmc
    w, h = mmc.getImageWidth(), mmc.getImageHeight()
    x, y, z = here(mic)
    frap_is_stim = bool(getattr(mmc, "use_frap_as_slm", False))
    print(f"camera        : {CAMERA_NAME}  {w}x{h} px")
    print(f"channel group : {mmc.getChannelGroup()}")
    print(f"channels      : {list(mmc.getAvailableConfigs(CHANNEL_GROUP))}")
    print(f"stage anchor  : x={x:.1f} y={y:.1f} z={z:.1f} um")
    print(f"stim device   : {'FRAP galvo' if frap_is_stim else 'SLM / DMD'}")
    print(f"FRAP channel  : {mic.frap_channel!r}")
    print(
        "DMD affine    : "
        + ("bypassed -- masks stay in camera px" if not mic.uses_dmd_affine
           else "IN USE (unexpected on this rig)")
    )
    print(f"writer        : {'OmeZarrWriter' if has_ome_writers() else 'TiffWriter (no ome_writers)'}")
