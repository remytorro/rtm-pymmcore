"""Inscoper microscope for rtm-pymmcore.

This module wraps the Inscoper ``UseqBridge`` (which directly implements
a ``CMMCorePlus``-like interface) behind the same ``AbstractMicroscope``
interface used by all other rtm-pymmcore microscopes.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator

import numpy as np
from qtpy.QtCore import QObject, Signal as QtSignal
from useq import MDAEvent
import locale 
import threading
import time

from faro.core.conversion import df_to_events
from faro.microscope.base import AbstractMicroscope

logger = logging.getLogger(__name__)


class _ViewerUpdater(QObject):
    """QObject living on the main thread that receives images via a Qt signal.

    Qt signals are thread-safe: emitting from the acquisition thread delivers
    the call to the main-thread event loop where the connected slot runs.
    """

    update_requested = QtSignal(object, str)  # (ndarray, label)

    def __init__(self, viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self.update_requested.connect(self._do_update)

    def _do_update(self, img: np.ndarray, label: str) -> None:
        try:
            if "Live" in self._viewer.layers:
                self._viewer.layers["Live"].data = img
                self._viewer.status = label
            else:
                self._viewer.add_image(
                    img,
                    name="Live",
                    metadata={"label": label},
                )
        except Exception as exc:
            print(f"[_ViewerUpdater] ERROR: {type(exc).__name__}: {exc}")


class _StorageOnlyPipeline:
    """Minimal pipeline stub that satisfies the :class:`~faro.core.controller.Analyzer`
    contract when no real :class:`~faro.core.pipeline.ImageProcessingPipeline`
    is available.

    It provides:
    * ``storage_path`` — so ``_do_store`` can save raw images to disk.
    * ``stimulator = None`` — so stim-related guards are skipped.
    * ``run()`` — no-op, so ``_try_submit_pipeline`` is harmless.
    * ``validate_pipeline()`` — always returns ``True``.

    This is created automatically by :meth:`InscoperMicroscope.run_experiment`
    when the user passes ``pipeline=[]`` or ``pipeline=None``.
    """

    def __init__(self, storage_path: str) -> None:
        import os

        self.storage_path = storage_path
        self.stimulator = None
        os.makedirs(os.path.join(storage_path, "raw"), exist_ok=True)

    def run(self, *, img=None, event=None, file_path=None):
        """No-op — skip segmentation / tracking / feature extraction."""
        return {"result": "STOP"}

    def validate_pipeline(self, events) -> bool:
        """Always valid — no pipeline constraints to check."""
        return True

class KeepDMDAlive:
    def __init__(self, mmc):
        self.mmc = mmc
        self.thread = None
        self.last_wakeup = 0
        self.is_running = False

    def wakeup_dmd(self):
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 200000.0)
        self.mmc.setSLMPixelsTo(self.mmc.getSLMDevice(), 255)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())

    def run(self):
        # Set locale to C/POSIX to ensure period as decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            # If 'C' is not available, try 'en_US.UTF-8' or 'en_US'
            for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
                try:
                    locale.setlocale(locale.LC_NUMERIC, loc)
                    break
                except locale.Error:
                    continue

        self.is_running = True
        self.last_wakeup = 0
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_wakeup > 60:  # Wake up every minute
                # Skip wakeup if a FRAP stimulation is currently in progress
                # to avoid interfering with the galvo sequence on the C++ bridge.
                frap_active = False
                frap_scanner = getattr(self.mmc, "_frap_scanner", None)
                if frap_scanner is not None:
                    frap_active = getattr(frap_scanner, "_frap_active", False)
                if not frap_active:
                    self.wakeup_dmd()
                    self.last_wakeup = current_time
            time.sleep(5)

    def stop(self):
        # Set locale to C/POSIX to ensure period as decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            # If 'C' is not available, try 'en_US.UTF-8' or 'en_US'
            for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
                try:
                    locale.setlocale(locale.LC_NUMERIC, loc)
                    break
                except locale.Error:
                    continue

        self.is_running = False
        self.thread.join()
        time.sleep(5)
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 100)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())



# ---------------------------------------------------------------------------
# Microscope
# ---------------------------------------------------------------------------


class InscoperMicroscope(AbstractMicroscope):
    """Inscoper adapter microscope — drop-in replacement for any rtm-pymmcore
    microscope class.

    Uses the Inscoper ``UseqBridge`` under the hood, which implements a
    ``CMMCorePlus``-compatible API so it integrates naturally with the
    :class:`~faro.microscope.base.AbstractMicroscope` interface.
    """

    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False
    DMD_CALIBRATION_PROFILE = {
        "channel_group": "Channel",
        "channel_config": "TL",
        "device_name": "NikonTi2",
        "property_name": "TL Lamp Intensity (%)",
        "power": 100,
    }

    # Maps a channel config name to the (device, property) that sets its
    # light-source power. Consulted by :meth:`resolve_power`; the calibration
    # channel falls back to ``DMD_CALIBRATION_PROFILE`` when unlisted.
    POWER_PROPERTIES: dict[str, tuple[str, str]] = {}

    def resolve_power(self, channel):
        """Return ``(device, property, power)`` for *channel*, or None.

        Mirrors :meth:`PyMMCoreMicroscope.resolve_power`: None means the
        channel carries no power to push (a plain ``Channel``, or a
        ``PowerChannel`` with ``power`` unset). When ``power`` IS set but no
        mapping resolves, this raises rather than returning None, since a
        silent None means the requested power never reaches the hardware.
        """
        power = getattr(channel, "power", None)
        if power is None:
            return None
        mapping = self.POWER_PROPERTIES.get(channel.config)
        if mapping is None:
            profile = self.DMD_CALIBRATION_PROFILE or {}
            if channel.config == profile.get("channel_config"):
                mapping = (profile["device_name"], profile["property_name"])
        if mapping is None:
            known = sorted(self.POWER_PROPERTIES)
            raise ValueError(
                f"Channel {channel.config!r} sets power={power}, but no "
                f"power-property mapping resolves for it, so the requested "
                f"power would be silently ignored. Add an explicit "
                f"POWER_PROPERTIES entry mapping it to its (device, "
                f"property), e.g. ('NikonTi2', 'TL Lamp Intensity (%)'). "
                f"Currently mapped channels: {known or '(none)'}."
            )
        device, prop = mapping
        return device, prop, power

    def __init__(
        self,
        config_folder: str,
        channels_folder: str,
        main_camera: str,
        channel_group: str = "Channel",
        use_only_pfs: bool = False,
        use_autofocus_event: bool = False,
        affine_calibration_matrix=None,
        frap_channel: str | None = None,
    ) -> None:
        super().__init__()

        self.CHANNEL_GROUP = channel_group
        self.frap_channel = frap_channel
        self.USE_ONLY_PFS = use_only_pfs
        self.USE_AUTOFOCUS_EVENT = use_autofocus_event
        self.config_folder = config_folder
        self.channels_folder = channels_folder
        self.main_camera = main_camera

        # Late import to avoid hard dependency at module level
        from inscoper_useq import UseqBridge

        self.mmc = UseqBridge()
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

        self.affine_calibration_matrix = affine_calibration_matrix
        self.wakeup_dmd = None
        self.init_scope()

    def init_scope(self) -> None:
        """Initialize the microscope.
        """
        self.mmc.loadSystemConfiguration(
            config_folder=self.config_folder,
            channels_folder=self.channels_folder,
            main_camera=self.main_camera,
        )

        # # Force camera initialisation if event-based init didn't fire.
        # # In some contexts (e.g. Jupyter notebooks) the device dispatcher's
        # # init events may not reach the CameraManager, leaving
        # # map_camera_device empty even though all_cameras_to_load is populated.
        # from inscoper_useq.basic_element.device._all_devices import AllDevices

        # _cm = AllDevices.camera_manager
        # if _cm is not None and not _cm.map_camera_device:
        #     logger.info(
        #         "Forcing camera initialisation for %d cameras...",
        #         len(_cm.all_cameras_to_load),
        #     )
        #     for cam_dev in _cm.all_cameras_to_load:
        #         _cm.init_camera(cam_dev)

        # logger.info(
        #     "Inscoper bridge initialised (config=%s, camera=%s)",
        #     self.config_folder,
        #     self.main_camera,
        # )

        # UseqBridge implements the CMMCorePlus-like API directly.
        # No adapter needed — same pattern as MMDemo using CMMCorePlus.
        self.mmc.setChannelGroup(self.CHANNEL_GROUP)
        try:
            slm = self.mmc.getSLMDevice()
            if slm:
                self.slm_dev = slm
                self.slm_width = self.mmc.getSLMWidth(slm)
                self.slm_height = self.mmc.getSLMHeight(slm)
        except Exception:
            logger.debug("No SLM device — slm_dev/width/height left as None.")


        # Register as the CMMCorePlus singleton so that napari-micromanager
        # widgets (SnapButton, LiveButton, ChannelWidget, etc.) that call
        # CMMCorePlus.instance() internally will get the UseqBridge.
        # This mirrors what CMMCorePlus.__init__ does automatically.
        import pymmcore_plus.core._mmcore_plus as _mmcore_mod

        _mmcore_mod._instance = self.mmc

        # Initialise DMD if a calibration profile is available.
        try:
            from faro.core.dmd import DMD

            self.dmd = DMD(
                self.mmc,
                resolve_power=self.resolve_power,
                affine_matrix=self.affine_calibration_matrix,
            )
            logger.info("DMD initialised (profile=%s).", self.DMD_CALIBRATION_PROFILE)

            # FRAP-as-SLM fix: when the bridge uses a FRAP galvo scanner
            # instead of a physical SLM, getSLMDevice() returns "" and DMD
            # ends up with height=0, width=0.  Substitute camera dimensions
            # so that affine_transform produces correctly-shaped masks and
            # calibration points are generated over the camera FOV.
            if getattr(self.mmc, "use_frap_as_slm", False):
                cam_h = self.mmc.getImageHeight()
                cam_w = self.mmc.getImageWidth()
                if (self.dmd.height == 0 or self.dmd.width == 0) and cam_h > 0 and cam_w > 0:
                    self.dmd.height = cam_h
                    self.dmd.width = cam_w
                    self.dmd.sample_mask_on = np.full((cam_h, cam_w), 255, dtype=np.uint8)
                    self.dmd.sample_mask_off = np.zeros((cam_h, cam_w), dtype=np.uint8)
                    # Use the FRAP device name so SLMImage.device is meaningful
                    frap_scanner = getattr(self.mmc, "_frap_scanner", None)
                    if frap_scanner is not None:
                        self.dmd.name = frap_scanner.device_name
                    logger.info(
                        "DMD dimensions overridden for FRAP-as-SLM: %dx%d (camera FOV).",
                        cam_w, cam_h,
                    )
        except Exception:
            logger.exception("Failed to initialise DMD — calibrate_dmd() will be unavailable.")
        
        if self.slm_dev:
            self.wakeup_dmd = KeepDMDAlive(self.mmc)
            self.wakeup_dmd.run()

        self.image_height = self.mmc.getImageHeight()
        self.image_width = self.mmc.getImageWidth()

    def run_mda(self, event_iter: Iterator[MDAEvent]):
        """Start an MDA acquisition sequence. Returns the acquisition thread."""
        return self.mmc.run_mda(event_iter)

    def connect_frame(self, callback: Callable[[np.ndarray, MDAEvent], None]) -> None:
        """Connect a ``frameReady`` callback: ``callback(img, event)``."""
        self.mmc.mda.events.frameReady.connect(callback)

    def disconnect_frame(
        self, callback: Callable[[np.ndarray, MDAEvent], None]
    ) -> None:
        """Disconnect a previously connected ``frameReady`` callback."""
        self.mmc.mda.events.frameReady.disconnect(callback)

    # ------------------------------------------------------------------
    # DMD calibration
    # ------------------------------------------------------------------

    def calibrate_dmd(
        self,
        calibration_channel,
        verbose: bool = False,
        n_points: int = 15,
        radius: int = 4,
        exposure: float = 25,
        marker_style: str = "x",
        calibration_points_DMD=None,
    ) -> None:
        """Calibrate the DMD affine transform (camera ↔ DMD coordinate mapping).

        Mirrors the :meth:`Moench.calibrate_dmd` interface so notebooks are
        portable across microscope backends.

        Parameters
        ----------
        calibration_channel : Channel | PowerChannel
            Light path used to image the projected DMD spots. Passed per
            experiment (e.g. a UV or a cyan channel) rather than fixed on
            the microscope.
        verbose : bool
            If *True*, display calibration images and residual plots.
        n_points : int
            Number of calibration points to project (default 15).
        radius : int
            Radius of the projected calibration disk in DMD pixels (default 4).
        exposure : float
            Camera exposure time in ms for each calibration snap (default 25).
        marker_style : str
            Matplotlib marker style for calibration overlay (default ``"x"``).
        calibration_points_DMD : list | None
            Explicit list of ``(row, col)`` DMD coordinates to use as
            calibration points.  When *None*, points are selected automatically
            via :meth:`~faro.core.dmd.DMD.select_well_distributed_points`.
        """
        from faro.core.dmd import DMD

        if not isinstance(self.dmd, DMD):
            print(f"{self.dmd=}")
            raise RuntimeError(
                "No DMD is attached to this InscoperMicroscope.  Provide a "
                "dmd_calibration_profile (or set DMD_CALIBRATION_PROFILE on a "
                "subclass) when constructing the microscope."
            )
        if self.dmd.affine is not None:
            logger.info("DMD already calibrated — skipping.")
            return
        if self.wakeup_dmd is not None:
            self.wakeup_dmd.stop()
        try:
            self.dmd.calibrate(
                calibration_channel,
                verbose=verbose,
                n_points=n_points,
                radius=radius,
                exposure=exposure,
                marker_style=marker_style,
                calibration_points_DMD=calibration_points_DMD,
            )
        finally:
            if self.wakeup_dmd is not None:
                self.wakeup_dmd.run()

    def cancel_mda(self) -> None:
        """Cancel the running MDA acquisition."""
        self.mmc.mda.cancel()

    def resolve_group(self, config_name: str) -> str:
        """Return the channel group, delegating to the bridge."""
        return self.mmc.getChannelGroup() or self.CHANNEL_GROUP

    # ------------------------------------------------------------------
    # FRAP photoactivation
    # ------------------------------------------------------------------

    def designate_frap_channel(self, name: str) -> list[str]:
        """Name the channel photoactivation fires with. Returns its problems.

        Worth doing explicitly, because the channel cannot be guessed from its
        name: on this configuration every stock channel called ``... WF Quad
        FRAP`` is an *imaging* channel through the quad FRAP dichroic and ships
        ``iLas2::Shutter = 0``, which scans the galvo in the dark. The
        stimulation channel here is ``all lasers``.
        """
        self.frap_channel = name
        self.mmc.set_frap_channel(name)
        problems = self._frap_channel_problems(name)
        if problems:
            logger.warning(
                "FRAP channel %r does not open the FRAP path: %s", name, "; ".join(problems)
            )
        else:
            logger.info("FRAP channel set to %r.", name)
        return problems

    def _frap_scanner_device(self):
        try:
            from inscoper_useq.basic_element.device._all_devices import AllDevices

            scanner = AllDevices.get_frap()
        except Exception:
            scanner = None
        if scanner is None:
            frap_device = getattr(self.mmc, "_frap_device", None)
            scanner = getattr(frap_device, "frap_scanner_device", None)
        return scanner

    def _frap_channel_problems(self, name: str | None) -> list[str]:
        scanner = self._frap_scanner_device()
        if scanner is None:
            return ["no FRAP scanner device in this configuration"]
        preset = self.mmc.get_frap_channel_preset()
        if preset is None:
            return [f"channel {name!r} is not loaded" if name else "no FRAP channel"]
        try:
            return list(scanner.channel_problems(preset))
        except Exception:
            return []

    def frap_preflight(self, sample_mask: np.ndarray | None = None) -> list[str]:
        """Everything that would make photoactivation silently do nothing.

        Every check here corresponds to a failure that produces **no light and
        no error** — the beam scans in the dark, or the ROI arrives empty, or
        the fire is refused. In a 24 h feedback experiment each one looks like
        a successful run until analysis. Call this before ``run_experiment``.

        Returns a list of problems; empty means clear.
        """
        problems: list[str] = []

        if not getattr(self.mmc, "use_frap_as_slm", False):
            problems.append(
                "bridge.use_frap_as_slm is False, so stim masks will be sent to "
                "an SLM rather than the FRAP galvo."
            )

        scanner = self._frap_scanner_device()
        if scanner is None:
            problems.append("no FRAP scanner device — nothing can be fired.")
            return problems

        # The channel. Without one that satisfies Ilda.frap.ActiveModeChannel
        # the galvo scans with the FRAP path shut.
        if self.frap_channel:
            problems += [
                f"FRAP channel {self.frap_channel!r}: {p}"
                for p in self._frap_channel_problems(self.frap_channel)
            ]
        else:
            preset = self.mmc.get_frap_channel_preset()
            if preset is None:
                problems.append(
                    "no FRAP channel designated and none of the loaded channels "
                    "opens the FRAP path. Call designate_frap_channel('all lasers')."
                )

        # The calibration, looked up exactly as firing will look it up — the
        # store keys on camera plus discriminant devices, and a calibration
        # registered under a different device state is invisible to fire().
        frap = getattr(self.mmc, "frap", None)
        if frap is None:
            problems.append("bridge.frap is None — no FrapDevice was built.")
        else:
            try:
                preset = self.mmc.get_frap_channel_preset()
                status = preset.get_status() if preset is not None else None
                if not frap.is_calibrated(self.main_camera, status):
                    problems.append(
                        f"no FRAP calibration for camera {self.main_camera!r} under "
                        "the channel's device state, so fire() will refuse. Run "
                        "script_frap_calibration.py, and note that the store keys "
                        "on the discriminant devices (the filter cube here), so a "
                        "calibration made under a different cube does not match."
                    )
            except Exception as exc:
                problems.append(f"could not check the FRAP calibration: {exc}")

        # The mask conversion, on a real mask if one is supplied.
        if sample_mask is not None:
            try:
                from inscoper_useq.basic_element.roi import FrapRoiContext, plan_mask

                ctx = FrapRoiContext(
                    interpoint_distance=frap._interpoint_distance() if frap else 4,
                    density_index=getattr(scanner, "density_index", None),
                )
                plan = plan_mask(np.asarray(sample_mask), ctx)
                logger.info("FRAP preflight: sample mask -> %s", plan.describe())
            except Exception as exc:
                problems.append(f"this mask cannot be sent as ROIs: {exc}")

        return problems

    def prepare_stim_mask(self, mask: np.ndarray) -> np.ndarray:
        """Hand the stim mask on in **camera** pixels when FRAP is the stim device.

        The FRAP galvo is not an SLM sitting in its own pixel grid. Its
        calibration (``FrapCalibration``, a quadratic-rotate fit) maps camera
        pixels straight to galvo DAC units, and the *firmware* applies it per
        point via ``interpIndex``. So a mask must arrive in camera space, and
        applying the DMD's camera->DMD affine first would transform it twice.

        That bug is currently invisible: with the DMD uncalibrated,
        ``affine_transform`` degrades to an identity resize, and the Inscoper
        adapter has already set ``dmd.width/height`` to the camera FOV — so the
        resize is a no-op. Calibrating the DMD on this system would silently
        start aiming FRAP at the wrong place. Hence the explicit opt-out rather
        than relying on that accident.
        """
        if getattr(self.mmc, "use_frap_as_slm", False):
            return mask
        return super().prepare_stim_mask(mask)

    @property
    def uses_dmd_affine(self) -> bool:
        """False when the FRAP galvo is the stim device.

        Mirrors the :meth:`prepare_stim_mask` opt-out above: with FRAP as the
        stim device the DMD affine is never applied, so ``run_experiment``'s
        pre-flight must not demand ``calibrate_dmd()`` — running it here is
        exactly what would aim the beam wrong.
        """
        if getattr(self.mmc, "use_frap_as_slm", False):
            return False
        return super().uses_dmd_affine

    # ------------------------------------------------------------------
    # Live napari viewer
    # ------------------------------------------------------------------

    def set_viewer(self, viewer) -> None:
        """Attach a napari viewer for live image display during acquisition.

        Call this before ``run_experiment()`` to see each acquired frame
        in real-time.

        Parameters
        ----------
        viewer : napari.Viewer
            An open napari viewer instance.
        """
        self._viewer = viewer
        self._viewer_updater = _ViewerUpdater(viewer)

    def _display_frame(self, img: np.ndarray, event: MDAEvent) -> None:
        """Update the napari viewer with the latest acquired frame.

        Connected via :meth:`connect_frame` so the Controller's own
        ``frameReady`` handler remains unaffected.  Dispatches via a Qt signal
        so the actual viewer update runs on the main thread.
        """
        updater = getattr(self, "_viewer_updater", None)
        if updater is None:
            return

        metadata = event.metadata or {}
        label = (
            f"t{metadata.get('timestep', '?')}"
            f"_p{metadata.get('fov', '?')}"
            f"_{metadata.get('fname', '')}"
        )
        updater.update_requested.emit(img.copy(), label)

    # ------------------------------------------------------------------
    # Pipeline — accept optional storage_path
    # ------------------------------------------------------------------

    def set_pipeline(self, pipeline, *, storage_path: str = None) -> None:
        """Set the image-processing pipeline.

        This overrides the base ``set_pipeline`` to accept an optional
        ``storage_path``.  When a *real* ``ImageProcessingPipeline`` is
        provided the ``storage_path`` is read from it automatically.  When
        ``pipeline`` is empty (``[]`` or ``None``), ``storage_path``
        **must** be supplied so that raw images can still be saved.

        Parameters
        ----------
        pipeline : ImageProcessingPipeline | list | None
            The pipeline object, or ``[]`` / ``None`` to skip processing.
        storage_path : str, optional
            Explicit path for image storage.  Falls back to
            ``pipeline.storage_path`` when not provided.
        """
        self.pipeline = pipeline
        if storage_path is not None:
            self._storage_path = storage_path
        elif hasattr(pipeline, "storage_path"):
            self._storage_path = pipeline.storage_path
        else:
            self._storage_path = None

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def run_experiment(self, df_acquire) -> None:
        """Run the experiment from a *df_acquire* DataFrame.

        Converts *df_acquire* to :class:`~faro.core.data_structures.RTMEvent`
        objects via :func:`~faro.core.conversion.df_to_events`, then
        delegates to :class:`~faro.core.controller.Controller`.

        If the pipeline is empty (``[]`` or ``None``), a lightweight
        :class:`_StorageOnlyPipeline` stub is used so that raw images are
        still saved to disk without running segmentation / tracking /
        feature-extraction.

        If a napari viewer is open but :meth:`set_viewer` was not called
        explicitly, the viewer is detected automatically via
        ``napari.current_viewer()``.
        """
        # -- Auto-detect napari viewer if not already set --------------------
        if getattr(self, "_viewer_updater", None) is None:
            try:
                import napari

                viewer = napari.current_viewer()

                # napari.current_viewer() can return None in Jupyter notebooks
                # with %gui qt because the viewer is created on the Qt thread but
                # current_viewer() reads a module-level variable set on a different
                # thread.  Fall back to scanning Qt top-level windows.
                if viewer is None:
                    try:
                        from qtpy.QtWidgets import QApplication
                        for widget in QApplication.topLevelWidgets():
                            if hasattr(widget, '_qt_viewer'):
                                viewer = getattr(widget._qt_viewer, 'viewer', None)
                                if viewer is not None:
                                    break
                    except Exception:
                        pass

                if viewer is not None:
                    self.set_viewer(viewer)
                    logger.info("napari viewer auto-detected — live display enabled.")
                else:
                    logger.warning(
                        "No napari viewer found — live display disabled. "
                        "Call mic.set_viewer(viewer) before run_experiment() "
                        "to enable it."
                    )
                    print(
                        "[InscoperMicroscope] WARNING: No napari viewer found. "
                        "Call mic.set_viewer(viewer) before run_experiment() for live display."
                    )
            except ImportError:
                pass  # napari not installed — skip

        # -- Resolve pipeline ------------------------------------------------
        pipeline = getattr(self, "pipeline", None)

        if not pipeline or not hasattr(pipeline, "storage_path"):
            sp = getattr(self, "_storage_path", None)
            if sp is None:
                raise ValueError(
                    "pipeline is empty and no storage_path was provided. "
                    "Call mic.set_pipeline(pipeline=[], "
                    "storage_path='path/to/output') first."
                )
            pipeline = _StorageOnlyPipeline(sp)
            logger.info(
                "No real pipeline — using _StorageOnlyPipeline (storage_path=%s)",
                sp,
            )

        # -- Pause DMD keep-alive so it doesn't interfere with acquisition ---
        if self.wakeup_dmd is not None:
            self.wakeup_dmd.stop()

        # -- Connect live napari display -------------------------------------
        if getattr(self, "_viewer_updater", None) is not None:
            self.connect_frame(self._display_frame)
            logger.info("Live napari display enabled.")

        # -- Build Controller and run ----------------------------------------
        from faro.core.controller import Controller

        self.controller = Controller(self, pipeline)

        events = df_to_events(df_acquire)
        try:
            # Run the controller in a background thread so the Qt event
            # loop stays alive and napari can display incoming images.
            import threading
            import time as _time

            _exc: list[BaseException] = []

            def _run():
                try:
                    self.controller.run_experiment(events)
                except BaseException as e:
                    _exc.append(e)

            t = threading.Thread(target=_run, daemon=True, name="ControllerRun")
            t.start()

            # Pump Qt events while waiting so napari repaints and signal
            # deliveries (ensure_main_thread, Qt signals) are processed.
            #
            # Use processEvents() with a 50 ms time limit so that a single
            # slow Qt handler (e.g. napari layer update) cannot block the
            # entire loop iteration and freeze the UI.
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QEventLoop

            app = QApplication.instance()
            while t.is_alive():
                if app is not None:
                    app.processEvents(QEventLoop.AllEvents, 50)
                _time.sleep(0.01)
            t.join()

            if _exc:
                raise _exc[0]
        finally:
            self.controller.finish_experiment()
            if getattr(self, "_viewer_updater", None) is not None:
                self.disconnect_frame(self._display_frame)
            if self.wakeup_dmd is not None:
                self.wakeup_dmd.run()

    def post_experiment(self) -> None:
        """Cleanup after the experiment.

        Waits for any remaining acquisition to finish, then closes the
        Inscoper bridge.

        Note: upstream now distinguishes ``post_experiment`` (runs *between*
        experiments, may keep devices warm) from :meth:`shutdown` (instance is
        being discarded). This one still performs the full teardown, so a
        second run in the same session needs a fresh microscope. Move the
        ``mmc.close()`` into :meth:`shutdown` only if back-to-back runs are
        wanted.
        """
        try:
            self.mmc.wait_for_acquisition(timeout=30)
        except Exception:
            logger.exception("Error waiting for acquisition to finish")

        try:
            self.mmc.close()
            logger.info("Inscoper bridge closed.")
        except Exception:
            logger.exception("Error closing Inscoper bridge")

        # Stop the DMD keep-alive thread so it doesn't outlive the bridge.
        if self.wakeup_dmd is not None:
            try:
                self.wakeup_dmd.stop()
            except Exception:
                logger.debug("wakeup_dmd.stop() failed — already stopped?")

        gc.collect()

    def shutdown(self) -> None:
        """Release all hardware held by this microscope.

        Implements the hook ``AbstractMicroscope`` added upstream: stop the
        DMD keep-alive thread and unload devices so SLM/COM handles are freed
        and the process can exit cleanly. Here that is the same teardown as
        :meth:`post_experiment`, and it is safe to call twice.
        """
        self.post_experiment()
