"""Wall-clock frame synchronization across multiple live camera streams.

:class:`LiveSynchronizer` polls all registered :class:`LiveCameraStream`
instances at a configurable rate, then groups frames whose timestamps are
within *tolerance_s* of each other.  It yields :class:`SyncBundle` objects
— one per synchronized group — to the registered consumer callback.

The synchronizer deliberately drops frames that cannot be matched within the
tolerance window so that real-time performance is maintained.

Usage::

    from caliscope.live_pipeline.live_camera_stream import LiveCameraStream
    from caliscope.live_pipeline.live_synchronizer import LiveSynchronizer

    streams = {0: LiveCameraStream(0), 1: LiveCameraStream(1)}
    for s in streams.values():
        s.start()

    def on_bundle(bundle):
        print(bundle.cam_ids, bundle.timestamps)

    sync = LiveSynchronizer(streams, fps=30, tolerance_s=0.033)
    sync.start(on_bundle)
    ...
    sync.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from caliscope.live_pipeline.live_camera_stream import FrameData, LiveCameraStream

logger = logging.getLogger(__name__)


@dataclass
class SyncBundle:
    """A synchronized collection of frames from multiple cameras.

    Parameters
    ----------
    sync_index:
        Monotonically increasing counter for this synchronizer session.
    frames:
        Mapping from *cam_id* → :class:`FrameData`.  Cameras that could not
        be matched within the tolerance window are absent from this dict.
    reference_timestamp:
        Wall-clock time used as the anchor for this synchronization cycle.
    """

    sync_index: int
    frames: dict[int, FrameData]
    reference_timestamp: float

    @property
    def cam_ids(self) -> list[int]:
        return list(self.frames.keys())

    @property
    def timestamps(self) -> dict[int, float]:
        return {cam_id: fd.timestamp for cam_id, fd in self.frames.items()}

    def max_offset_ms(self) -> float:
        """Max timestamp deviation from reference, in milliseconds."""
        if not self.frames:
            return 0.0
        offsets = [abs(fd.timestamp - self.reference_timestamp) * 1000 for fd in self.frames.values()]
        return max(offsets)


BundleCallback = Callable[[SyncBundle], None]


class LiveSynchronizer:
    """Synchronize frames from multiple :class:`LiveCameraStream` instances.

    Parameters
    ----------
    streams:
        Mapping *cam_id* → :class:`LiveCameraStream`.  The cam_ids in this
        dict become the canonical camera identifiers in emitted bundles.
    fps:
        Target output rate.  The synchronizer polls at *1/fps* seconds and
        emits at most one bundle per poll cycle.
    tolerance_s:
        Two frames are considered simultaneous when their timestamps differ
        by less than this value (seconds).  Default 33 ms (≈ 1 frame @ 30 fps).
    require_all_cameras:
        When True, only emit a bundle if every camera contributed a frame.
        When False, emit partial bundles (some cameras may be absent).
    """

    def __init__(
        self,
        streams: dict[int, LiveCameraStream],
        fps: float = 30.0,
        tolerance_s: float = 0.033,
        require_all_cameras: bool = False,
    ) -> None:
        self.streams = streams
        self.fps = fps
        self.tolerance_s = tolerance_s
        self.require_all_cameras = require_all_cameras

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sync_index = 0
        self._callback: BundleCallback | None = None

        # Track the most recently emitted frame index per camera so we never
        # emit the same frame twice in the same bundle stream.
        self._last_emitted: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, callback: BundleCallback) -> None:
        """Start the synchronization loop; call *callback* for each bundle."""
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sync_loop,
            name="LiveSynchronizer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "LiveSynchronizer started: %d cameras, %.1f fps, tolerance %.0f ms",
            len(self.streams),
            self.fps,
            self.tolerance_s * 1000,
        )

    def stop(self) -> None:
        """Stop the synchronization loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("LiveSynchronizer stopped after %d bundles", self._sync_index)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        interval = 1.0 / self.fps
        while not self._stop_event.is_set():
            loop_start = time.time()

            bundle = self._build_bundle()
            if bundle is not None and self._callback is not None:
                self._callback(bundle)

            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _build_bundle(self) -> SyncBundle | None:
        """Collect the latest frame from each camera and build a bundle."""
        # Gather current snapshots
        snapshots: dict[int, FrameData] = {}
        for cam_id, stream in self.streams.items():
            data = stream.get_latest_frame_data()
            if data is not None:
                snapshots[cam_id] = data

        if not snapshots:
            return None

        # Use the earliest timestamp as the reference anchor.
        # This ensures the fastest camera is always included; slower cameras
        # are included only if they fall within *tolerance_s* of the leader.
        timestamps = np.array([fd.timestamp for fd in snapshots.values()])
        reference_ts = float(np.min(timestamps))

        # Keep only frames within tolerance of the reference timestamp
        matched: dict[int, FrameData] = {}
        for cam_id, fd in snapshots.items():
            if abs(fd.timestamp - reference_ts) <= self.tolerance_s:
                # Skip if we already emitted this exact frame
                if self._last_emitted.get(cam_id) == fd.frame_index:
                    continue
                matched[cam_id] = fd

        if not matched:
            return None

        if self.require_all_cameras and len(matched) < len(self.streams):
            logger.debug(
                "Bundle dropped: only %d/%d cameras matched within %.0f ms",
                len(matched),
                len(self.streams),
                self.tolerance_s * 1000,
            )
            return None

        # Record emitted frame indices
        for cam_id, fd in matched.items():
            self._last_emitted[cam_id] = fd.frame_index

        bundle = SyncBundle(
            sync_index=self._sync_index,
            frames=matched,
            reference_timestamp=reference_ts,
        )
        self._sync_index += 1
        return bundle
