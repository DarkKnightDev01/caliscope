"""Camera detection utilities.

Scans available webcam device indices using OpenCV and reports their
capabilities (supported resolutions, actual FPS). Iriun virtual webcams
on Windows appear as standard DirectShow/V4L2 devices and are discovered
by the same index-scanning approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2

logger = logging.getLogger(__name__)

# Candidate resolutions probed during capability detection, ordered from
# highest to lowest so the first successful probe gives the maximum supported
# resolution.
_PROBE_RESOLUTIONS: list[tuple[int, int]] = [
    (3840, 2160),  # 4K UHD
    (2560, 1440),  # 2K QHD
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
    (640, 480),    # VGA (baseline)
]


@dataclass
class CameraInfo:
    """Metadata about a detected camera device."""

    index: int
    """OpenCV device index (0-based)."""

    width: int
    """Frame width reported by the camera at the detected resolution."""

    height: int
    """Frame height reported by the camera at the detected resolution."""

    fps: float
    """Frame rate reported by the camera (may be approximate)."""

    supported_resolutions: list[tuple[int, int]] = field(default_factory=list)
    """Resolutions confirmed to be accepted by the device (descending order)."""

    @property
    def resolution_label(self) -> str:
        """Human-readable resolution string, e.g. ``"1920x1080"``."""
        return f"{self.width}x{self.height}"

    def __str__(self) -> str:
        return (
            f"Camera {self.index}: {self.resolution_label} @ {self.fps:.1f} fps  "
            f"(supported: {', '.join(f'{w}x{h}' for w, h in self.supported_resolutions)})"
        )


class CameraDetector:
    """Enumerate available webcam devices.

    Iterates through device indices 0–``max_index`` (inclusive) and opens
    each one with :class:`cv2.VideoCapture`.  Devices that respond are
    probed for resolution and FPS capabilities.

    Args:
        max_index: Highest device index to probe (default 10).
        probe_resolutions: Whether to probe every candidate resolution for each
            device.  When *False* only the device's default resolution is
            recorded.  Probing takes a few extra seconds per camera.
    """

    def __init__(self, max_index: int = 10, probe_resolutions: bool = True) -> None:
        self.max_index = max_index
        self.probe_resolutions = probe_resolutions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self) -> list[CameraInfo]:
        """Scan device indices and return info for every reachable camera.

        Returns:
            List of :class:`CameraInfo` objects, one per detected device,
            sorted by device index.
        """
        found: list[CameraInfo] = []
        for idx in range(self.max_index + 1):
            info = self._probe_index(idx)
            if info is not None:
                found.append(info)
                logger.info("Detected %s", info)
        logger.info("Camera detection complete: %d device(s) found.", len(found))
        return found

    def detect_indices(self) -> list[int]:
        """Return a plain list of available device indices.

        Convenience wrapper around :meth:`detect` for callers that only need
        the indices (e.g. when constructing a :class:`~caliscope.live_capture.MultiCameraRecorder`).
        """
        return [info.index for info in self.detect()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _probe_index(self, index: int) -> CameraInfo | None:
        """Try to open device *index* and read one frame.

        Returns ``None`` if the device cannot be opened or yields no frame.
        """
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None

        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None

        # Read default width/height/fps before any resolution probing
        default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        default_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        supported: list[tuple[int, int]] = []

        if self.probe_resolutions:
            supported = self._probe_supported_resolutions(cap, default_w, default_h)
        else:
            supported = [(default_w, default_h)]

        # Use the highest confirmed resolution as the camera's reported size
        best_w, best_h = supported[0] if supported else (default_w, default_h)

        cap.release()
        return CameraInfo(
            index=index,
            width=best_w,
            height=best_h,
            fps=default_fps,
            supported_resolutions=supported,
        )

    @staticmethod
    def _probe_supported_resolutions(
        cap: cv2.VideoCapture,
        default_w: int,
        default_h: int,
    ) -> list[tuple[int, int]]:
        """Ask the capture device to switch to each candidate resolution.

        A resolution is considered *supported* when the device actually adopts
        it (i.e. the reported width/height match after :func:`cv2.VideoCapture.set`).

        The default resolution is always included in the result.
        """
        accepted: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for w, h in _PROBE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            res = (actual_w, actual_h)
            if res not in seen:
                seen.add(res)
                accepted.append(res)

        # Restore to default
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_h)

        # Sort descending by pixel count so highest resolution is first
        accepted.sort(key=lambda r: r[0] * r[1], reverse=True)
        return accepted
