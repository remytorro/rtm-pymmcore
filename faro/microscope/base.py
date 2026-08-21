from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator
from threading import Thread

# Silence the FutureWarning from napari-micromanager importing deprecated pymmcore_plus.mda.handlers
warnings.filterwarnings(
    "ignore",
    message="The 'pymmcore_plus.mda.handlers' module is deprecated",
    category=FutureWarning,
)

# PYMM_SIGNALS_BACKEND is hard-set to 'psygnal' in faro/__init__.py before any
# submodule (including this one) is imported. A setdefault here would be a
# no-op anyway: pymmcore_widgets pre-empts to 'qt' whenever it imports first.

import numpy as np
from useq import MDAEvent

from faro.core.dmd import DMD


class AbstractMicroscope:
    """Base class defining the microscope interface.

    The Controller depends only on this interface — it never touches
    pymmcore-plus directly.  Subclasses implement the four MDA callables
    plus optional ``resolve_group`` / ``resolve_power`` for channel
    resolution.
    """

    os.environ["QT_LOGGING_RULES"] = (
        "*.debug=false; *.warning=false"  # Fix to suppress PyQT warnings from napari-micromanager when running in a Jupyter notebook
    )

    dmd = None                      # optional DMD device
    use_autofocus_event = False     # optional autofocus
    dmd_needs_to_be_waken = False   # optional DMD wake

    def __init__(self):
        self.dmd = None

    def init_scope(self):
        """Initialize the microscope scope."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    # ------------------------------------------------------------------
    # MDA interface — used by Controller
    # ------------------------------------------------------------------

    def run_mda(self, event_iter: Iterator[MDAEvent]) -> Thread:
        """Start MDA acquisition. Returns thread/handle."""
        raise NotImplementedError

    def connect_frame(self, callback: Callable[[np.ndarray, MDAEvent], None]) -> None:
        """Connect frameReady callback: callback(img, event)."""
        raise NotImplementedError

    def disconnect_frame(self, callback: Callable[[np.ndarray, MDAEvent], None]) -> None:
        """Disconnect frameReady callback."""
        raise NotImplementedError

    def cancel_mda(self) -> None:
        """Cancel running MDA."""
        raise NotImplementedError

    def resolve_group(self, config_name) -> str:
        """Return channel group for config name. Optional override."""
        return ""

    def resolve_power(self, channel):
        """Return (device, property, power) or None. Optional override."""
        return None

    def prepare_stim_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map a camera-space stim *mask* into the stim device's own space.

        The pipeline always produces masks in **camera** pixels, because that is
        where segmentation happened. Getting them to the illuminator is
        device-specific, so the microscope owns it: a DMD needs the calibrated
        camera->DMD affine applied here, while a device that does its own
        coordinate transform downstream must receive camera pixels untouched.

        Overriding this is the supported way to opt out of the DMD warp — doing
        it by leaving the DMD uncalibrated works only by accident, and breaks
        the moment somebody calibrates it.
        """
        return self.dmd.affine_transform(mask)

    @property
    def uses_dmd_affine(self) -> bool:
        """Whether stim masks go through the DMD's camera->DMD affine.

        ``True`` for a real DMD, which needs :meth:`calibrate_dmd` before it
        can aim. Subclasses that override :meth:`prepare_stim_mask` to bypass
        the warp (e.g. a FRAP galvo whose firmware applies its own
        calibration) return ``False`` so the calibration check below does not
        demand a calibration their light path never reads.
        """
        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_hardware(self, events) -> bool:
        """Validate events against hardware capabilities.

        Base implementation only runs the DMD calibration check; subclasses
        compose extra checks (channel configs, exposure limits, device
        properties, etc.) by calling ``super().validate_hardware(events)``.
        """
        return self._validate_dmd_calibration(events)

    def _validate_dmd_calibration(self, events) -> bool:
        """Warn if events contain stim but the DMD isn't calibrated.

        Skipped when the microscope has no DMD, does not route masks
        through the DMD affine (:attr:`uses_dmd_affine`), or the events
        have no stim channels — non-DMD setups and non-stim experiments don't
        need it. Without this check, the failure surfaces deep in the
        first stim event as ``DMD.affine_transform`` raising
        ``ValueError("DMD not calibrated...")``.
        """
        if self.dmd is None:
            return True
        if not self.uses_dmd_affine:
            return True
        if getattr(self.dmd, "affine", None) is not None:
            return True
        if not any(getattr(ev, "stim_channels", ()) for ev in events):
            return True
        warnings.warn(
            "DMD not calibrated (affine matrix is None) but events contain "
            "stim channels. Run mic.calibrate_dmd() before starting the "
            "experiment.",
            UserWarning,
        )
        return False

    # ------------------------------------------------------------------
    # DMD
    # ------------------------------------------------------------------

    def calibrate_dmd(self, calibration_channel):
        """Calibrate the DMD. Always runs the calibration when called."""
        if isinstance(self.dmd, DMD):
            self.dmd.calibrate(calibration_channel)

    def post_experiment(self):
        """Post-process the experiment. Optional override."""
        pass

    def shutdown(self):
        """Release all hardware resources held by this microscope.

        Unlike :meth:`post_experiment` (which runs *between* experiments
        and may keep devices warm for the next run), ``shutdown`` signals
        that this microscope instance is being discarded. Subclasses
        should stop any background threads (DMD wakeup loops, etc.) and
        unload devices so that COM ports / SLM handles are released and
        the Python process can exit cleanly.

        Base implementation is a no-op; subclasses override as needed.
        """
        pass
