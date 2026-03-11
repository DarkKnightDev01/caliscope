"""Optional recording of live sessions to disk.

When recording is active, :class:`LiveRecorder` simultaneously:

* Writes raw video frames to ``<output_dir>/cam_<id>.mp4`` via
  ``cv2.VideoWriter``.
* Accumulates tracked 2-D landmark data and flushes it to
  ``<output_dir>/xy_<tracker_name>.csv`` on :meth:`stop`.

The output format mirrors caliscope's existing convention so that recorded
sessions can be re-processed through the standard
:class:`~caliscope.managers.synchronized_stream_manager.SynchronizedStreamManager`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from caliscope.live_pipeline.live_tracker_pipeline import TrackedBundle

logger = logging.getLogger(__name__)

# Four-CC code for MP4 output.  H264 is widely supported; fall back to mp4v.
_FOURCC_H264 = cv2.VideoWriter.fourcc(*"avc1")  # type: ignore[attr-defined]
_FOURCC_MP4V = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]


class LiveRecorder:
    """Records video and landmark CSV data from live tracking sessions.

    Parameters
    ----------
    output_dir:
        Directory where output files are written.  Created if it does not
        exist.
    tracker_name:
        Name of the active tracker (e.g. ``"SIMPLE_HOLISTIC"``).  Used to
        construct the CSV filename ``xy_<tracker_name>.csv``.
    fps:
        Frame-rate used for the output video writers.
    """

    def __init__(
        self,
        output_dir: Path,
        tracker_name: str = "TRACKER",
        fps: float = 30.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tracker_name = tracker_name
        self.fps = fps

        self._writers: dict[int, cv2.VideoWriter] = {}
        self._csv_rows: list[dict] = []
        self._lock = threading.Lock()
        self._recording = False
        self._frame_size: dict[int, tuple[int, int]] = {}  # cam_id → (w, h)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enable recording; writers are created lazily on the first frame."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._recording = True
        logger.info("LiveRecorder started → %s", self.output_dir)

    def stop(self) -> None:
        """Flush data to disk and release all video writers."""
        self._recording = False

        with self._lock:
            for cam_id, writer in self._writers.items():
                writer.release()
                logger.info("Released VideoWriter for cam %d", cam_id)
            self._writers.clear()

            self._flush_csv()

        logger.info("LiveRecorder stopped")

    def record(self, bundle: TrackedBundle) -> None:
        """Write one synchronized frame bundle to disk.

        This method is thread-safe and may be called from any thread.
        """
        if not self._recording:
            return

        with self._lock:
            for cam_id, fp in bundle.frame_packets.items():
                if fp.frame is None:
                    continue

                writer = self._get_or_create_writer(cam_id, fp.frame)
                if writer is not None:
                    writer.write(fp.frame)

                row_data = fp.to_tidy_table(bundle.sync_index)
                if row_data is not None:
                    # to_tidy_table returns a dict of lists; expand to rows
                    count = len(row_data["sync_index"])
                    for i in range(count):
                        self._csv_rows.append({k: v[i] for k, v in row_data.items()})

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_writer(
        self, cam_id: int, frame: Any
    ) -> cv2.VideoWriter | None:
        if cam_id in self._writers:
            return self._writers[cam_id]

        h, w = frame.shape[:2]
        video_path = self.output_dir / f"cam_{cam_id}.mp4"

        writer = cv2.VideoWriter(
            str(video_path),
            _FOURCC_MP4V,
            self.fps,
            (w, h),
        )
        if not writer.isOpened():
            logger.error("Failed to open VideoWriter for cam %d at %s", cam_id, video_path)
            return None

        self._writers[cam_id] = writer
        logger.info("Opened VideoWriter for cam %d → %s (%dx%d @ %.0f fps)", cam_id, video_path, w, h, self.fps)
        return writer

    def _flush_csv(self) -> None:
        if not self._csv_rows:
            logger.info("No landmark data to write")
            return

        csv_path = self.output_dir / f"xy_{self.tracker_name}.csv"
        df = pd.DataFrame(self._csv_rows)
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d landmark rows → %s", len(self._csv_rows), csv_path)
        self._csv_rows.clear()
