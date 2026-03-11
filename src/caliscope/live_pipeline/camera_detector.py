"""Auto-detect available webcam devices by scanning OpenCV device indices.

Works with Iriun virtual webcams on Windows (they appear as standard
DirectShow devices) and any other OpenCV-compatible cameras.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2

logger = logging.getLogger(__name__)

# Common resolutions to probe when discovering camera capabilities
_PROBE_RESOLUTIONS: list[tuple[int, int]] = [
    (3840, 2160),  # 4K UHD
    (1920, 1080),  # 1080p
    (1280, 720),   # 720p
    (640, 480),    # VGA
]


@dataclass
class CameraInfo:
    """Information about a single detected camera device."""

    index: int
    name: str
    supported_resolutions: list[tuple[int, int]] = field(default_factory=list)

    def __str__(self) -> str:
        res_str = ", ".join(f"{w}x{h}" for w, h in self.supported_resolutions)
        return f"Camera {self.index}: {self.name} [{res_str}]"


def _probe_resolution(cap: cv2.VideoCapture, width: int, height: int) -> bool:
    """Try to set a resolution on an open capture; return True if accepted."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return actual_w == width and actual_h == height


def _get_camera_name(index: int) -> str:
    """Return a human-readable name for a camera index (best-effort)."""
    # OpenCV does not expose camera names portably, so we use a generic label.
    return f"Camera {index}"


def detect_cameras(
    max_index: int = 10,
    probe_resolutions: bool = True,
) -> list[CameraInfo]:
    """Scan device indices 0 through *max_index* and return available cameras.

    Parameters
    ----------
    max_index:
        Highest device index to try (inclusive).
    probe_resolutions:
        When True, attempt to set each resolution in *_PROBE_RESOLUTIONS* and
        record which ones the driver accepts.

    Returns
    -------
    list[CameraInfo]
        One entry per detected camera, in ascending index order.
    """
    cameras: list[CameraInfo] = []

    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        # Verify we can actually read a frame — some virtual devices open but
        # return nothing (e.g. unplugged Iriun phone).
        ok, _ = cap.read()
        if not ok:
            cap.release()
            logger.debug("Camera index %d opened but returned no frame; skipping", idx)
            continue

        info = CameraInfo(index=idx, name=_get_camera_name(idx))

        if probe_resolutions:
            for w, h in _PROBE_RESOLUTIONS:
                if _probe_resolution(cap, w, h):
                    info.supported_resolutions.append((w, h))
                    logger.debug("Camera %d supports %dx%d", idx, w, h)

        cap.release()
        cameras.append(info)
        logger.info("Detected %s", info)

    return cameras
