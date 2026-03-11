"""Multi-camera synchronized recorder.

:class:`MultiCameraRecorder` opens several :class:`~caliscope.live_capture.LiveStream`
instances, grabs frames from all of them at a configurable rate, and writes
per-camera ``cam_<N>.mp4`` files to an output directory.

The output directory layout matches what caliscope's existing pipeline expects::

    output_dir/
        cam_0.mp4
        cam_1.mp4
        cam_2.mp4
        cam_3.mp4
        frame_times.csv    ← wall-clock timestamps for every written frame

Frame synchronization strategy
-------------------------------
Each :class:`~caliscope.live_capture.LiveStream` runs its own capture thread at the
camera's native frame rate.  A single *writer thread* wakes up at the
requested *target FPS*, reads the latest frame from every camera, and writes
them to the corresponding :class:`cv2.VideoWriter`.  Because all cameras are
sampled at the same instant (wall clock), the resulting files are temporally
aligned.

The trade-off is that if one camera is momentarily unavailable its latest
stale frame is repeated rather than dropped, keeping the per-camera files the
same length.  Caliscope's :class:`~caliscope.managers.synchronized_stream_manager.SynchronizedStreamManager`
is tolerant of such duplication.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.live_capture.live_stream import LiveStream

logger = logging.getLogger(__name__)

# Fourcc code used for all output files.  mp4v gives broad compatibility
# across platforms.  On Windows, avc1/H.264 would also work but requires a
# licensed codec; mp4v is always available via OpenCV.
# The type:ignore suppresses a false-positive from static analysers that
# do not have stubs for cv2.VideoWriter_fourcc (it is a runtime function).
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]


class MultiCameraRecorder:
    """Record synchronized video from multiple webcam devices.

    Args:
        camera_indices: OpenCV device indices to capture from.
        output_dir: Directory where ``cam_<N>.mp4`` files will be written.
            Created if it does not exist.
        width: Desired frame width (0 = device default).
        height: Desired frame height (0 = device default).
        fps: Target recording frame rate (default 30).
    """

    def __init__(
        self,
        camera_indices: list[int],
        output_dir: Path,
        width: int = 0,
        height: int = 0,
        fps: float = 30.0,
    ) -> None:
        if not camera_indices:
            raise ValueError("camera_indices must not be empty.")

        self.camera_indices = list(camera_indices)
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.fps = fps

        self._streams: dict[int, LiveStream] = {}
        self._writers: dict[int, cv2.VideoWriter] = {}
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._csv_rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Open all cameras and start writing frames.

        Raises:
            RuntimeError: If any camera fails to open.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_streams()
        self._open_writers()

        self._stop_event.clear()
        self._csv_rows = []
        self._writer_thread = threading.Thread(
            target=self._write_loop,
            name="MultiCameraRecorder-writer",
            daemon=True,
        )
        self._writer_thread.start()
        logger.info(
            "Recording started: %d camera(s) → %s", len(self._streams), self.output_dir
        )

    def stop_recording(self) -> None:
        """Stop recording and release all resources.

        Blocks until the writer thread has finished and all video files are
        flushed.  Writes ``frame_times.csv`` to :attr:`output_dir`.
        """
        self._stop_event.set()

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)
            self._writer_thread = None

        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()

        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

        self._write_timestamps_csv()
        logger.info("Recording stopped. Files saved to %s", self.output_dir)

    def __enter__(self) -> "MultiCameraRecorder":
        self.start_recording()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_recording()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_streams(self) -> None:
        """Open a :class:`LiveStream` for every requested camera index."""
        for cam_idx in self.camera_indices:
            stream = LiveStream(
                camera_index=cam_idx,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )
            stream.open()
            stream.start()
            self._streams[cam_idx] = stream

    def _open_writers(self) -> None:
        """Create a :class:`cv2.VideoWriter` for every open stream."""
        for position, cam_idx in enumerate(self.camera_indices):
            stream = self._streams[cam_idx]
            out_path = self.output_dir / f"cam_{position}.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                _FOURCC,
                self.fps,
                (stream.actual_width, stream.actual_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {out_path}.")
            self._writers[cam_idx] = writer
            logger.debug(
                "VideoWriter opened: %s (%dx%d @ %.1f fps)",
                out_path,
                stream.actual_width,
                stream.actual_height,
                self.fps,
            )

    def _write_loop(self) -> None:
        """Periodically sample all cameras and write frames to disk."""
        interval = 1.0 / self.fps
        frame_index = 0

        while not self._stop_event.is_set():
            loop_start = time.time()

            row: dict[str, object] = {"frame_index": frame_index}

            for position, cam_idx in enumerate(self.camera_indices):
                stream = self._streams.get(cam_idx)
                writer = self._writers.get(cam_idx)
                if stream is None or writer is None:
                    continue

                frame, ts = stream.get_latest_frame()
                if frame is None:
                    # Camera not yet ready; write a black placeholder so all
                    # files stay the same length.
                    s = self._streams[cam_idx]
                    frame = np.zeros(
                        (s.actual_height or 480, s.actual_width or 640, 3),
                        dtype=np.uint8,
                    )
                    ts = time.time()

                writer.write(frame)
                row[f"cam_{position}_timestamp"] = ts

            self._csv_rows.append(row)
            frame_index += 1

            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _write_timestamps_csv(self) -> None:
        """Persist per-frame wall-clock timestamps to ``frame_times.csv``."""
        if not self._csv_rows:
            return

        csv_path = self.output_dir / "frame_times.csv"
        fieldnames = list(self._csv_rows[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._csv_rows)
        logger.info("Frame timestamps written to %s (%d rows).", csv_path, len(self._csv_rows))
