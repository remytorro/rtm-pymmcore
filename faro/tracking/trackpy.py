import numpy as np
import pandas as pd
from faro.tracking.base import Tracker
import trackpy


class TrackerTrackpy(Tracker):
    def __init__(self, search_range=50, memory=3, adaptive_stop=3, adaptive_step=0.95):
        super().__init__()
        self.search_range = search_range
        self.memory = memory
        self.adaptive_stop = adaptive_stop
        self.adaptive_step = adaptive_step

    def track_cells(
        self, df_old: pd.DataFrame, df_new: pd.DataFrame, fov_state
    ) -> pd.DataFrame:
        """Track cells in a dataframe using trackpy library.
        Args:
            df_old: Previous tracking DataFrame.
            df_new: New detections with columns 'x', 'y', 'label'.
            fov_state: FovState instance holding linker and counter."""

        required_columns = ["x", "y", "label"]

        # Empty frame (no detections): still need to advance the linker
        if df_new.empty:
            if fov_state.linker is not None:
                fov_state.linker.next_level(
                    np.empty((0, 2)), fov_state.fov_timestep_counter
                )
            return df_old

        missing_columns = [col for col in required_columns if col not in df_new.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        coordinates = np.array(
            df_new[["x", "y"]]
        )  # Convert the df to an array of shape (shape: N, ndim) for trackpy

        # First-frame init when either df_old is empty (true first frame for
        # this FOV) or the linker hasn't been built yet (df_old recovered
        # from the parquet file fallback after a get_predecessor timeout, or
        # carried forward across early empty-detection frames). Without the
        # linker check next_level below crashes with NoneType.next_level.
        needs_init = df_old.empty or fov_state.linker is None
        if needs_init:
            if not df_old.empty and fov_state.linker is None:
                print(
                    f"[trackpy] linker is None but df_old has {len(df_old)} rows "
                    "(likely a recovered/stale predecessor). Discarding df_old "
                    "and starting a fresh linker for this FOV — particle IDs "
                    "from earlier frames will not carry over."
                )
                df_old = pd.DataFrame()
            fov_state.linker = trackpy.linking.Linker(
                search_range=self.search_range,
                memory=self.memory,
                adaptive_stop=self.adaptive_stop,
                adaptive_step=self.adaptive_step,
            )

            fov_state.linker.init_level(
                coordinates, fov_state.fov_timestep_counter
            )  # extract positions and convert to horizontal list
            df_new["particle"] = fov_state.linker.particle_ids
            df_new["fov_timestep"] = fov_state.fov_timestep_counter
            df_tracked = df_new

        else:
            # this is not the first frame
            try:
                fov_state.linker.next_level(
                    coordinates, fov_state.fov_timestep_counter
                )  # extract positions and convert to horizontal list
            except Exception as exc:
                # A trackpy Linker that raises inside next_level is left
                # half-updated and is dead for good: the first failure here was
                # SubnetOversizeException (search_range too large for the cell
                # density -- 48 points in one subnet against trackpy's limit of
                # 30), and every later frame then raised
                # "'NoneType' object has no attribute 'add_point'" from the
                # linker's own internals. Reusing it cost a whole acquisition:
                # 33 consecutive failed frames, no tracks, therefore no stim
                # masks, therefore no FRAP, while the controller waited out its
                # mask timeout on every stim event (lag reached 199 s).
                #
                # So treat the linker as unrecoverable and start a new one on
                # this frame's own detections. Tracking survives; the price is
                # that particle IDs restart here, which is why this says so
                # loudly rather than logging at debug level.
                print(
                    f"[trackpy] linker failed at frame "
                    f"{fov_state.fov_timestep_counter} "
                    f"({type(exc).__name__}: {exc}). A linker that raises is "
                    "left in an inconsistent state, so it is being discarded "
                    "and rebuilt from this frame's detections. PARTICLE IDS "
                    "RESTART HERE -- tracks before and after this frame are "
                    "not the same cells. If this repeats, search_range="
                    f"{self.search_range} is too large for this sample's "
                    "density: lower it (and adaptive_stop) rather than letting "
                    "every frame rebuild."
                )
                fov_state.linker = trackpy.linking.Linker(
                    search_range=self.search_range,
                    memory=self.memory,
                    adaptive_stop=self.adaptive_stop,
                    adaptive_step=self.adaptive_step,
                )
                fov_state.linker.init_level(
                    coordinates, fov_state.fov_timestep_counter
                )
                df_new["particle"] = fov_state.linker.particle_ids
                df_new["fov_timestep"] = fov_state.fov_timestep_counter
                return df_new.reset_index(drop=True)

            df_new["particle"] = fov_state.linker.particle_ids
            df_new["fov_timestep"] = fov_state.fov_timestep_counter
            df_tracked = pd.concat([df_old, df_new])

        # this is against a in trackpy, where the same ID gets assigned twice in one frame
        df_tracked = df_tracked.drop_duplicates(subset=["particle", "fov_timestep"])
        df_tracked = df_tracked.reset_index(drop=True)
        return df_tracked
