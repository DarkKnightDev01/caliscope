"""Thread-based live camera stream wrapping ``cv2.VideoCapture``.

Each :class:`LiveCameraStream` runs a background thread that continuously
reads frames from an OpenCV capture device.  The latest frame is available
via :meth:`get_latest_frame` without blocking.  Wall-clock timestamps are
attached to every frame.

Usage::

    stream = LiveCameraStream(device_index=0, width=1920, height=1080, fps=30)
    stream.start()

    frame, timestamp = stream.get_latest_frame()
    ...

    stream.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """A captured frame together with its wall-clock timestamp."""

    frame: NDArray[Any]
    timestamp: float  # seconds since epoch (time.time())
    frame_index: int
    cam_id: int


class LiveCameraStream:
    """Wraps ``cv2.VideoCapture`` and reads frames in a background thread.

    Parameters
    ----------
    device_index:
        OpenCV device index (e.g. 0, 1, 2 …).
    cam_id:
        Logical camera identifier used when creating :class:`FrameData` objects.
        Defaults to *device_index*.
    width, height:
        Requested capture resolution.  The driver may silently use a different
        resolution if the requested one is not supported.
    fps:
        Requested capture frame-rate.
    read_timeout:
        Seconds to wait for a new frame before declaring the camera stalled.
    """

    def __init__(
        self,
        device_index: int,
        cam_id: int | None = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        read_timeout: float = 5.0,
    ) -> None:
        self.device_index = device_index
        self.cam_id: int = cam_id if cam_id is not None else device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.read_timeout = read_timeout

        self._cap: cv2.VideoCapture | None = None
        self._latest: FrameData | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the capture device and begin the background capture loop."""
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.device_index}. "
                "Check that Iriun (or your webcam) is connected and running."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera %d opened: requested %dx%d@%dfps, got %dx%d@%.1ffps",
            self.device_index,
            self.width,
            self.height,
            self.fps,
            actual_w,
            actual_h,
            actual_fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"LiveCameraStream-{self.cam_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and release the capture device."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.read_timeout + 1)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera %d stream stopped", self.device_index)

    def get_latest_frame(self) -> tuple[NDArray[Any], float] | tuple[None, None]:
        """Return the most recently captured frame and its timestamp.

        Returns ``(None, None)`` if no frame has been captured yet.
        """
        with self._lock:
            if self._latest is None:
                return None, None
            return self._latest.frame, self._latest.timestamp

    def get_latest_frame_data(self) -> FrameData | None:
        """Return the most recent :class:`FrameData`, or ``None``."""
        with self._lock:
            return self._latest

    @property
    def is_running(self) -> bool:
        """True while the background capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_count(self) -> int:
        """Total number of frames captured since :meth:`start` was called."""
        return self._frame_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames from the device."""
        assert self._cap is not None

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Camera %d: failed to read frame", self.device_index)
                time.sleep(0.01)
                continue

            ts = time.time()
            self._frame_count += 1

            data = FrameData(
                frame=frame,
                timestamp=ts,
                frame_index=self._frame_count,
                cam_id=self.cam_id,
            )

            with self._lock:
                self._latest = data

        logger.debug("Camera %d capture loop exiting", self.device_index)
