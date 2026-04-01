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

from rtm_pymmcore.core.conversion import df_to_events
from rtm_pymmcore.microscope.base import AbstractMicroscope

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
            logger.error(f"[_ViewerUpdater] ERROR: {type(exc).__name__}: {exc}")


class _StorageOnlyPipeline:
    """Minimal pipeline stub that satisfies the :class:`~rtm_pymmcore.core.controller.Analyzer`
    contract when no real :class:`~rtm_pymmcore.core.pipeline.ImageProcessingPipeline`
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


class InscoperMicroscope(AbstractMicroscope):
    """Inscoper adapter microscope — drop-in replacement for any rtm-pymmcore
    microscope class.

    Uses the Inscoper ``UseqBridge`` under the hood, which implements a
    ``CMMCorePlus``-compatible API so it integrates naturally with the
    :class:`~rtm_pymmcore.microscope.base.AbstractMicroscope` interface.
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

    def __init__(
        self,
        config_folder: str,
        channels_folder: str,
        main_camera: str,
        channel_group: str = "Channel",
        use_only_pfs: bool = False,
        use_autofocus_event: bool = False,
        affine_calibration_matrix=None,
    ) -> None:
        super().__init__()

        self.CHANNEL_GROUP = channel_group
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
            from rtm_pymmcore.core.dmd import DMD

            self.dmd = DMD(
                self.mmc,
                self.DMD_CALIBRATION_PROFILE,
                affine_matrix=self.affine_calibration_matrix,
            )
            logger.info("DMD initialised (profile=%s).", self.DMD_CALIBRATION_PROFILE)
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

    def calibrate_dmd(
        self,
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
            via :meth:`~rtm_pymmcore.core.dmd.DMD.select_well_distributed_points`.
        """
        from rtm_pymmcore.core.dmd import DMD

        if not isinstance(self.dmd, DMD):
            logger.info(f"{self.dmd=}")
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

    def run_experiment(self, df_acquire) -> None:
        """Run the experiment from a *df_acquire* DataFrame.

        Converts *df_acquire* to :class:`~rtm_pymmcore.core.data_structures.RTMEvent`
        objects via :func:`~rtm_pymmcore.core.conversion.df_to_events`, then
        delegates to :class:`~rtm_pymmcore.core.controller.Controller`.

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
        from rtm_pymmcore.core.controller import Controller

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
