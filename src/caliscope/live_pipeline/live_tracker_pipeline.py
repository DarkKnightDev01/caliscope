"""Core real-time tracking pipeline.

:class:`LiveTrackerPipeline` receives :class:`SyncBundle` objects from
:class:`LiveSynchronizer`, runs caliscope's :class:`Tracker` on each
camera's frame (optionally in parallel), and emits :class:`TrackedBundle`
results to registered consumers.

Design goals
------------
* **Drop frames over queuing** — if tracking falls behind, old bundles are
  discarded rather than accumulated.
* **Thread-safe** — the pipeline can be started/stopped from any thread.
* **Pluggable tracker** — any :class:`caliscope.tracker.Tracker` subclass works.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from caliscope.packets import FramePacket, PointPacket
from caliscope.tracker import Tracker
from caliscope.live_pipeline.live_synchronizer import SyncBundle

logger = logging.getLogger(__name__)


@dataclass
class TrackedBundle:
    """Tracked results for one synchronized frame group.

    Parameters
    ----------
    sync_index:
        Matches the originating :class:`SyncBundle`.
    frame_packets:
        Mapping *cam_id* → :class:`FramePacket` with 2-D landmark data.
    reference_timestamp:
        Wall-clock reference timestamp from the originating bundle.
    """

    sync_index: int
    frame_packets: dict[int, FramePacket]
    reference_timestamp: float


TrackedCallback = Callable[[TrackedBundle], None]

# Maximum number of pending bundles before the oldest is dropped
_MAX_QUEUE_SIZE = 2


class LiveTrackerPipeline:
    """Runs caliscope's tracker on synchronized frame bundles in real-time.

    Parameters
    ----------
    tracker:
        Any :class:`caliscope.tracker.Tracker` implementation.
    parallel:
        When True, tracker.get_points() is called for all cameras
        concurrently via a ThreadPoolExecutor.
    max_workers:
        Number of worker threads used for parallel tracking.  Defaults to
        the number of cameras seen in the first bundle.
    """

    def __init__(
        self,
        tracker: Tracker,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.tracker = tracker
        self.parallel = parallel
        self.max_workers = max_workers

        self._input_queue: queue.Queue[SyncBundle | None] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._callback: TrackedCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._processed_count = 0
        self._dropped_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, callback: TrackedCallback) -> None:
        """Start the tracking worker thread.

        Parameters
        ----------
        callback:
            Called on the worker thread for each :class:`TrackedBundle`.
        """
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            name="LiveTrackerPipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("LiveTrackerPipeline started (parallel=%s)", self.parallel)

    def stop(self) -> None:
        """Signal the worker thread to stop and wait for it to exit."""
        self._stop_event.set()
        # Unblock the worker if it is waiting for input
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info(
            "LiveTrackerPipeline stopped: %d processed, %d dropped",
            self._processed_count,
            self._dropped_count,
        )

    def push_bundle(self, bundle: SyncBundle) -> None:
        """Submit a :class:`SyncBundle` for tracking.

        If the internal queue is full, the oldest item is evicted and *bundle*
        takes its place, ensuring real-time behaviour.
        """
        try:
            self._input_queue.put_nowait(bundle)
        except queue.Full:
            # Evict the oldest bundle and insert the new one
            try:
                self._input_queue.get_nowait()
                self._dropped_count += 1
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(bundle)
            except queue.Full:
                self._dropped_count += 1

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                bundle = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if bundle is None:  # shutdown sentinel
                break

            tracked = self._process_bundle(bundle)
            if tracked is not None and self._callback is not None:
                self._callback(tracked)

            self._processed_count += 1

    def _process_bundle(self, bundle: SyncBundle) -> TrackedBundle | None:
        """Run the tracker on every camera frame in *bundle*."""
        if self.parallel and len(bundle.frames) > 1:
            frame_packets = self._process_parallel(bundle)
        else:
            frame_packets = self._process_sequential(bundle)

        return TrackedBundle(
            sync_index=bundle.sync_index,
            frame_packets=frame_packets,
            reference_timestamp=bundle.reference_timestamp,
        )

    def _process_sequential(self, bundle: SyncBundle) -> dict[int, FramePacket]:
        result: dict[int, FramePacket] = {}
        for cam_id, frame_data in bundle.frames.items():
            fp = self._track_single(cam_id, frame_data.frame, frame_data.frame_index, frame_data.timestamp)
            result[cam_id] = fp
        return result

    def _process_parallel(self, bundle: SyncBundle) -> dict[int, FramePacket]:
        result: dict[int, FramePacket] = {}
        workers = min(self.max_workers, len(bundle.frames))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._track_single,
                    cam_id,
                    frame_data.frame,
                    frame_data.frame_index,
                    frame_data.timestamp,
                ): cam_id
                for cam_id, frame_data in bundle.frames.items()
            }
            for future in as_completed(futures):
                cam_id = futures[future]
                try:
                    result[cam_id] = future.result()
                except Exception:
                    logger.exception("Tracker failed on cam %d", cam_id)
        return result

    def _track_single(
        self,
        cam_id: int,
        frame: NDArray[Any],
        frame_index: int,
        frame_time: float,
    ) -> FramePacket:
        """Run the tracker on a single frame and wrap the result."""
        try:
            point_packet = self.tracker.get_points(frame, cam_id=cam_id)
        except Exception:
            logger.exception("Tracker.get_points raised on cam %d", cam_id)
            point_packet = PointPacket(
                point_id=np.array([], dtype=np.int32),
                img_loc=np.empty((0, 2), dtype=np.float64),
            )

        return FramePacket(
            cam_id=cam_id,
            frame_index=frame_index,
            frame_time=frame_time,
            frame=frame,
            points=point_packet,
            draw_instructions=self.tracker.scatter_draw_instructions,
        )
