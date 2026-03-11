"""Live camera capture module for caliscope.

Enables real-time capture from multiple webcam devices (e.g. Iriun virtual
webcams on Windows) and records synchronized .mp4 files that caliscope's
existing processing pipeline can consume directly.

Usage (CLI)::

    # Detect available cameras
    python -m caliscope.live_capture.live_capture_cli detect

    # Preview all detected cameras
    python -m caliscope.live_capture.live_capture_cli preview

    # Record from all detected cameras
    python -m caliscope.live_capture.live_capture_cli record \\
        --output ./my_recording --duration 30 --resolution 1920x1080 --fps 30

Usage (API)::

    from caliscope.live_capture import CameraDetector, MultiCameraRecorder

    detector = CameraDetector()
    indices = detector.detect()

    recorder = MultiCameraRecorder(camera_indices=indices, output_dir=Path("./recording"))
    recorder.start_recording()
    time.sleep(30)
    recorder.stop_recording()
"""

from caliscope.live_capture.camera_detector import CameraDetector, CameraInfo
from caliscope.live_capture.live_stream import LiveStream
from caliscope.live_capture.multi_camera_recorder import MultiCameraRecorder

__all__ = [
    "CameraDetector",
    "CameraInfo",
    "LiveStream",
    "MultiCameraRecorder",
]
