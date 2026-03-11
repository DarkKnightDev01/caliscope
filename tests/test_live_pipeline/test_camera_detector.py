"""Unit tests for camera_detector.py using mocked cv2.VideoCapture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from caliscope.live_pipeline.camera_detector import CameraInfo, detect_cameras


class _FakeCapture:
    """Minimal cv2.VideoCapture mock."""

    def __init__(self, index: int, opens: bool = True, reads: bool = True) -> None:
        self._index = index
        self._opens = opens
        self._reads = reads
        # Default resolution reported back
        self._props: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self._opens

    def read(self):
        import numpy as np

        if self._reads:
            return True, np.zeros((480, 640, 3), dtype="uint8")
        return False, None

    def set(self, prop_id: int, value: float) -> bool:
        self._props[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        import cv2

        # Return the value that was set, so resolution probing succeeds
        return self._props.get(prop_id, 0.0)

    def release(self) -> None:
        pass


def _make_vc_factory(camera_indices: list[int]):
    """Return a VideoCapture factory that only 'opens' the listed indices."""
    import cv2

    def factory(index):
        if index in camera_indices:
            cap = _FakeCapture(index, opens=True, reads=True)
            # Pre-populate with default resolution so .get() reflects .set()
            cap._props = {
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            }
            return cap
        return _FakeCapture(index, opens=False, reads=False)

    return factory


class TestCameraInfo:
    def test_str_with_resolutions(self):
        info = CameraInfo(index=2, name="Camera 2", supported_resolutions=[(1920, 1080), (640, 480)])
        text = str(info)
        assert "Camera 2" in text
        assert "1920x1080" in text
        assert "640x480" in text

    def test_str_no_resolutions(self):
        info = CameraInfo(index=0, name="Camera 0")
        assert "Camera 0" in str(info)


class TestDetectCameras:
    def test_no_cameras(self):
        """When no device opens, detect_cameras returns an empty list."""
        with patch("caliscope.live_pipeline.camera_detector.cv2.VideoCapture") as mock_vc:
            mock_vc.side_effect = _make_vc_factory([])
            result = detect_cameras(max_index=4, probe_resolutions=False)
        assert result == []

    def test_single_camera_detected(self):
        """One camera at index 1 is found; index 0 is absent."""
        with patch("caliscope.live_pipeline.camera_detector.cv2.VideoCapture") as mock_vc:
            mock_vc.side_effect = _make_vc_factory([1])
            result = detect_cameras(max_index=3, probe_resolutions=False)

        assert len(result) == 1
        assert result[0].index == 1

    def test_multiple_cameras_detected(self):
        """Three cameras at indices 0, 2, 3 are all found."""
        with patch("caliscope.live_pipeline.camera_detector.cv2.VideoCapture") as mock_vc:
            mock_vc.side_effect = _make_vc_factory([0, 2, 3])
            result = detect_cameras(max_index=5, probe_resolutions=False)

        assert len(result) == 3
        indices = [c.index for c in result]
        assert indices == [0, 2, 3]

    def test_camera_that_opens_but_cannot_read_is_skipped(self):
        """A device that isOpened() but read() fails should not appear."""
        import cv2

        def factory(index):
            if index == 0:
                return _FakeCapture(0, opens=True, reads=False)
            return _FakeCapture(index, opens=False)

        with patch("caliscope.live_pipeline.camera_detector.cv2.VideoCapture") as mock_vc:
            mock_vc.side_effect = factory
            result = detect_cameras(max_index=2, probe_resolutions=False)

        assert result == []

    def test_resolution_probing_records_supported_resolutions(self):
        """When probe_resolutions=True, supported_resolutions is populated."""
        import cv2

        class ResolutionCapture(_FakeCapture):
            """Returns exactly what was set, simulating a 640x480 device."""

            def set(self, prop_id, value):
                # Only "accept" 640x480
                if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                    self._props[prop_id] = 640.0 if value == 640 else 0.0
                elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                    self._props[prop_id] = 480.0 if value == 480 else 0.0
                return True

        def factory(index):
            if index == 0:
                return ResolutionCapture(0, opens=True, reads=True)
            return _FakeCapture(index, opens=False)

        with patch("caliscope.live_pipeline.camera_detector.cv2.VideoCapture") as mock_vc:
            mock_vc.side_effect = factory
            result = detect_cameras(max_index=2, probe_resolutions=True)

        assert len(result) == 1
        # 640x480 should be in the supported list
        assert (640, 480) in result[0].supported_resolutions

    def test_max_index_respected(self):
        """Indices beyond max_index are never probed."""
        call_count = 0

        def counting_factory(index):
            nonlocal call_count
            call_count += 1
            return _FakeCapture(index, opens=False)

        with patch("caliscope.live_pipeline.camera_detector.cv2.VideoCapture") as mock_vc:
            mock_vc.side_effect = counting_factory
            detect_cameras(max_index=3, probe_resolutions=False)

        # Should have been called for indices 0, 1, 2, 3 — i.e. max_index+1 times
        assert call_count == 4
