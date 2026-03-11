"""Single-camera live capture stream.

:class:`LiveStream` wraps :class:`cv2.VideoCapture` and runs a background
thread that continuously grabs frames to keep the internal buffer fresh.
Callers read the latest frame (and its wall-clock timestamp) at any time via
:meth:`get_latest_frame`.

Design notes
------------
* The grab loop runs at native camera speed; the caller is responsible for
  consuming frames at the desired rate.
* Frames are *not* queued — only the most recent frame is retained.  This
  prevents unbounded memory growth when the consumer is slower than the camera.
* Thread safety: the latest frame/timestamp pair is protected by a
  :class:`threading.Lock`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LiveStream:
    """Continuously captures frames from a single webcam device.

    Args:
        camera_index: OpenCV device index.
        width: Desired frame width in pixels (0 = device default).
        height: Desired frame height in pixels (0 = device default).
        fps: Desired frames per second (0 = device default).

    Raises:
        RuntimeError: If the device cannot be opened.
    """

    def __init__(
        self,
        camera_index: int,
        width: int = 0,
        height: int = 0,
        fps: float = 0,
    ) -> None:
        self.camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Latest captured frame and its wall-clock timestamp (seconds since epoch)
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_timestamp: float = 0.0

        # Actual properties reported by the device after opening
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the capture device and apply requested settings.

        Raises:
            RuntimeError: If the device cannot be opened or yields no frame.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.camera_index}.")

        if self._width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Read back actual properties after applying settings
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Warm-up: grab one frame to confirm the device is responsive
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {self.camera_index} opened but returned no frame.")

        with self._lock:
            self._latest_frame = frame
            self._latest_timestamp = time.time()

        self._cap = cap
        logger.info(
            "Camera %d opened: %dx%d @ %.1f fps",
            self.camera_index,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
        )

    def start(self) -> None:
        """Start the background capture thread.

        :meth:`open` must be called first.

        Raises:
            RuntimeError: If the device has not been opened yet.
        """
        if self._cap is None:
            raise RuntimeError("Call open() before start().")
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"LiveStream-{self.camera_index}",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Capture thread started for camera %d.", self.camera_index)

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("Camera %d stream stopped.", self.camera_index)

    def __enter__(self) -> "LiveStream":
        self.open()
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> tuple[NDArray[np.uint8] | None, float]:
        """Return the most recently captured frame and its timestamp.

        Returns:
            Tuple of ``(frame, timestamp)`` where *frame* is a BGR NumPy
            array and *timestamp* is seconds since the Unix epoch.  Returns
            ``(None, 0.0)`` if no frame has been captured yet.
        """
        with self._lock:
            return self._latest_frame, self._latest_timestamp

    @property
    def is_running(self) -> bool:
        """``True`` while the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Grab frames continuously until :meth:`stop` is called."""
        cap = self._cap
        if cap is None:
            return

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Camera %d: grab failed, retrying…", self.camera_index)
                time.sleep(0.01)
                continue

            ts = time.time()
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = ts
