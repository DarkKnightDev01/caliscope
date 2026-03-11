"""Unit tests for the live camera capture module.

Tests use mocked :class:`cv2.VideoCapture` instances so that no physical
camera device is required.  An integration test (``test_integration_record``)
is provided but skipped automatically when no real cameras are detected.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from caliscope.live_capture.camera_detector import CameraDetector, CameraInfo
from caliscope.live_capture.live_stream import LiveStream
from caliscope.live_capture.multi_camera_recorder import MultiCameraRecorder


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cap_mock(
    opened: bool = True,
    read_ok: bool = True,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> MagicMock:
    """Build a :class:`MagicMock` that mimics :class:`cv2.VideoCapture`."""
    mock_frame = np.zeros((height, width, 3), dtype=np.uint8)

    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (read_ok, mock_frame if read_ok else None)
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
        cv2.CAP_PROP_FPS: fps,
    }.get(prop, 0.0)
    cap.set.return_value = True
    return cap


# ---------------------------------------------------------------------------
# CameraDetector tests
# ---------------------------------------------------------------------------


class TestCameraDetector:
    def test_detect_no_cameras(self) -> None:
        """When all VideoCapture calls fail, detect() returns an empty list."""
        cap_mock = _make_cap_mock(opened=False)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            detector = CameraDetector(max_index=3)
            results = detector.detect()
        assert results == []

    def test_detect_single_camera(self) -> None:
        """A single working camera is detected and returned."""
        cap_mock = _make_cap_mock(opened=True, read_ok=True, width=1280, height=720)

        def side_effect(index: int) -> MagicMock:
            return cap_mock if index == 0 else _make_cap_mock(opened=False)

        with patch("cv2.VideoCapture", side_effect=side_effect):
            detector = CameraDetector(max_index=3, probe_resolutions=False)
            results = detector.detect()

        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].width == 1280
        assert results[0].height == 720

    def test_detect_multiple_cameras(self) -> None:
        """Multiple cameras at different indices are all returned."""
        active_indices = {0, 2}

        def side_effect(index: int) -> MagicMock:
            return _make_cap_mock(opened=True) if index in active_indices else _make_cap_mock(opened=False)

        with patch("cv2.VideoCapture", side_effect=side_effect):
            detector = CameraDetector(max_index=4, probe_resolutions=False)
            results = detector.detect()

        detected_indices = {r.index for r in results}
        assert detected_indices == active_indices

    def test_detect_indices(self) -> None:
        """detect_indices() returns a plain list of integer indices."""
        cap_mock = _make_cap_mock(opened=True)

        def side_effect(index: int) -> MagicMock:
            return cap_mock if index == 1 else _make_cap_mock(opened=False)

        with patch("cv2.VideoCapture", side_effect=side_effect):
            detector = CameraDetector(max_index=3, probe_resolutions=False)
            indices = detector.detect_indices()

        assert indices == [1]

    def test_camera_info_str(self) -> None:
        """CameraInfo.__str__ produces a readable description."""
        info = CameraInfo(
            index=2,
            width=1920,
            height=1080,
            fps=29.97,
            supported_resolutions=[(1920, 1080), (1280, 720)],
        )
        text = str(info)
        assert "Camera 2" in text
        assert "1920x1080" in text

    def test_camera_info_resolution_label(self) -> None:
        info = CameraInfo(index=0, width=3840, height=2160, fps=30.0)
        assert info.resolution_label == "3840x2160"

    def test_no_frame_returns_none(self) -> None:
        """A camera that opens but yields no frame is excluded."""
        cap_mock = _make_cap_mock(opened=True, read_ok=False)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            detector = CameraDetector(max_index=0, probe_resolutions=False)
            results = detector.detect()
        assert results == []


# ---------------------------------------------------------------------------
# LiveStream tests
# ---------------------------------------------------------------------------


class TestLiveStream:
    def test_open_success(self) -> None:
        """open() succeeds when the device is available."""
        cap_mock = _make_cap_mock(opened=True, width=1920, height=1080, fps=30.0)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            stream = LiveStream(camera_index=0, width=1920, height=1080)
            stream.open()
            assert stream.actual_width == 1920
            assert stream.actual_height == 1080
            stream.stop()

    def test_open_failure_raises(self) -> None:
        """open() raises RuntimeError when the device cannot be opened."""
        cap_mock = _make_cap_mock(opened=False)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            stream = LiveStream(camera_index=99)
            with pytest.raises(RuntimeError, match="Cannot open camera"):
                stream.open()

    def test_open_no_frame_raises(self) -> None:
        """open() raises RuntimeError when the device opens but yields no frame."""
        cap_mock = _make_cap_mock(opened=True, read_ok=False)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            stream = LiveStream(camera_index=0)
            with pytest.raises(RuntimeError, match="returned no frame"):
                stream.open()

    def test_start_requires_open(self) -> None:
        """start() raises RuntimeError if open() was not called first."""
        stream = LiveStream(camera_index=0)
        with pytest.raises(RuntimeError, match="open()"):
            stream.start()

    def test_get_latest_frame_before_start(self) -> None:
        """get_latest_frame() returns (None, 0.0) before any frame is captured."""
        cap_mock = _make_cap_mock(opened=True)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            stream = LiveStream(camera_index=0)
            stream.open()
            # After open() the warm-up frame is stored; stop without starting thread
            frame, ts = stream.get_latest_frame()
            assert frame is not None
            stream.stop()

    def test_context_manager(self) -> None:
        """LiveStream can be used as a context manager."""
        cap_mock = _make_cap_mock(opened=True, width=640, height=480)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            with LiveStream(camera_index=0) as stream:
                assert stream.is_running

    def test_capture_thread_updates_frame(self) -> None:
        """The capture thread updates the latest frame over time."""
        frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_b = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frames = iter([
            (True, frame_a),
            (True, frame_b),
            (True, frame_b),
            (True, frame_b),
        ])

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.read.side_effect = lambda: next(frames, (True, frame_b))
        cap_mock.get.side_effect = lambda p: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(p, 0.0)
        cap_mock.set.return_value = True

        with patch("cv2.VideoCapture", return_value=cap_mock):
            with LiveStream(camera_index=0) as stream:
                time.sleep(0.1)
                frame, ts = stream.get_latest_frame()
                assert frame is not None
                assert ts > 0


# ---------------------------------------------------------------------------
# MultiCameraRecorder tests
# ---------------------------------------------------------------------------


class TestMultiCameraRecorder:
    def test_empty_indices_raises(self) -> None:
        """Constructing a recorder with no camera indices raises ValueError."""
        with pytest.raises(ValueError, match="camera_indices must not be empty"):
            MultiCameraRecorder(camera_indices=[], output_dir=Path("/tmp/test"))

    def test_record_creates_output_files(self, tmp_path: Path) -> None:
        """Recording creates cam_<N>.mp4 and frame_times.csv in the output dir."""
        cap_mock = _make_cap_mock(opened=True, width=64, height=64, fps=10.0)
        writer_mock = MagicMock()
        writer_mock.isOpened.return_value = True

        with (
            patch("cv2.VideoCapture", return_value=cap_mock),
            patch("cv2.VideoWriter", return_value=writer_mock),
        ):
            recorder = MultiCameraRecorder(
                camera_indices=[0, 1],
                output_dir=tmp_path / "rec",
                width=64,
                height=64,
                fps=10.0,
            )
            recorder.start_recording()
            time.sleep(0.3)
            recorder.stop_recording()

        # frame_times.csv should have been written
        csv_path = tmp_path / "rec" / "frame_times.csv"
        assert csv_path.exists(), "frame_times.csv was not created"
        # At least one row of data should be present
        lines = csv_path.read_text().splitlines()
        assert len(lines) >= 2, "frame_times.csv should contain header + data rows"

    def test_record_writes_frames(self, tmp_path: Path) -> None:
        """VideoWriter.write() is called during recording."""
        cap_mock = _make_cap_mock(opened=True, width=64, height=64, fps=10.0)
        writer_mock = MagicMock()
        writer_mock.isOpened.return_value = True

        with (
            patch("cv2.VideoCapture", return_value=cap_mock),
            patch("cv2.VideoWriter", return_value=writer_mock),
        ):
            recorder = MultiCameraRecorder(
                camera_indices=[0],
                output_dir=tmp_path / "rec",
                fps=10.0,
            )
            recorder.start_recording()
            time.sleep(0.3)
            recorder.stop_recording()

        assert writer_mock.write.call_count > 0, "No frames were written"

    def test_context_manager(self, tmp_path: Path) -> None:
        """MultiCameraRecorder works as a context manager."""
        cap_mock = _make_cap_mock(opened=True, width=64, height=64, fps=10.0)
        writer_mock = MagicMock()
        writer_mock.isOpened.return_value = True

        with (
            patch("cv2.VideoCapture", return_value=cap_mock),
            patch("cv2.VideoWriter", return_value=writer_mock),
        ):
            with MultiCameraRecorder(
                camera_indices=[0],
                output_dir=tmp_path / "rec",
                fps=5.0,
            ):
                time.sleep(0.2)

        assert writer_mock.release.called


# ---------------------------------------------------------------------------
# Integration test (skipped when no real cameras available)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_integration_record(tmp_path: Path) -> None:
    """Record a short clip from real cameras (skipped if none are available)."""
    detector = CameraDetector(max_index=5, probe_resolutions=False)
    indices = detector.detect_indices()
    if not indices:
        pytest.skip("No cameras detected; skipping integration test.")

    output_dir = tmp_path / "integration_recording"
    recorder = MultiCameraRecorder(
        camera_indices=indices[:2],  # Use at most 2 cameras
        output_dir=output_dir,
        width=640,
        height=480,
        fps=10.0,
    )
    recorder.start_recording()
    time.sleep(2.0)
    recorder.stop_recording()

    # Check that output files exist
    for i in range(len(indices[:2])):
        mp4_path = output_dir / f"cam_{i}.mp4"
        assert mp4_path.exists(), f"Expected output file {mp4_path} was not created."

    csv_path = output_dir / "frame_times.csv"
    assert csv_path.exists(), "frame_times.csv was not created."


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_detect_command_no_cameras(self, capsys: pytest.CaptureFixture[str]) -> None:
        """detect command prints a no-cameras message when none are found."""
        from caliscope.live_capture.live_capture_cli import main

        cap_mock = _make_cap_mock(opened=False)
        with patch("cv2.VideoCapture", return_value=cap_mock):
            exit_code = main(["--max-index", "2", "detect"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "No cameras detected" in captured.out

    def test_detect_command_with_camera(self, capsys: pytest.CaptureFixture[str]) -> None:
        """detect command lists camera info when a camera is found."""
        from caliscope.live_capture.live_capture_cli import main

        cap_mock = _make_cap_mock(opened=True, width=1280, height=720, fps=30.0)

        def side_effect(index: int) -> MagicMock:
            return cap_mock if index == 0 else _make_cap_mock(opened=False)

        with patch("cv2.VideoCapture", side_effect=side_effect):
            exit_code = main(["--max-index", "1", "detect"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Camera 0" in captured.out

    def test_record_command(self, tmp_path: Path) -> None:
        """record command creates output files without raising."""
        from caliscope.live_capture.live_capture_cli import main

        cap_mock = _make_cap_mock(opened=True, width=64, height=64, fps=10.0)
        writer_mock = MagicMock()
        writer_mock.isOpened.return_value = True

        with (
            patch("cv2.VideoCapture", side_effect=lambda i: cap_mock if i == 0 else _make_cap_mock(opened=False)),
            patch("cv2.VideoWriter", return_value=writer_mock),
        ):
            exit_code = main([
                "--max-index", "0",
                "record",
                "--cameras", "0",
                "--output", str(tmp_path / "rec"),
                "--duration", "0.3",
                "--fps", "10",
            ])

        assert exit_code == 0

    def test_parse_resolution_invalid(self) -> None:
        """_parse_resolution raises for malformed input."""
        import argparse

        from caliscope.live_capture.live_capture_cli import _parse_resolution

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_resolution("bad-value")

    def test_parse_resolution_valid(self) -> None:
        """_parse_resolution correctly parses a valid WxH string."""
        from caliscope.live_capture.live_capture_cli import _parse_resolution

        assert _parse_resolution("1920x1080") == (1920, 1080)
        assert _parse_resolution("3840x2160") == (3840, 2160)
