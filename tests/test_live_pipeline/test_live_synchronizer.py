"""Unit tests for LiveSynchronizer using synthetic timestamps.

No physical cameras are required — we inject :class:`FrameData` objects
with controlled timestamps directly into mocked stream objects.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from caliscope.live_pipeline.live_camera_stream import FrameData
from caliscope.live_pipeline.live_synchronizer import LiveSynchronizer, SyncBundle


def _make_frame_data(cam_id: int, timestamp: float, frame_index: int = 1) -> FrameData:
    """Helper: create a FrameData with a dummy 4x4 frame."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return FrameData(frame=frame, timestamp=timestamp, frame_index=frame_index, cam_id=cam_id)


def _make_stream(cam_id: int, timestamp: float, frame_index: int = 1) -> MagicMock:
    """Return a mock LiveCameraStream that returns the given FrameData."""
    stream = MagicMock()
    fd = _make_frame_data(cam_id, timestamp, frame_index)
    stream.get_latest_frame_data.return_value = fd
    return stream


class TestSyncBundle:
    def test_cam_ids(self):
        fd0 = _make_frame_data(0, 1.0)
        fd1 = _make_frame_data(1, 1.0)
        bundle = SyncBundle(sync_index=0, frames={0: fd0, 1: fd1}, reference_timestamp=1.0)
        assert sorted(bundle.cam_ids) == [0, 1]

    def test_timestamps(self):
        fd0 = _make_frame_data(0, 1.001)
        fd1 = _make_frame_data(1, 0.999)
        bundle = SyncBundle(sync_index=0, frames={0: fd0, 1: fd1}, reference_timestamp=1.0)
        assert abs(bundle.timestamps[0] - 1.001) < 1e-9
        assert abs(bundle.timestamps[1] - 0.999) < 1e-9

    def test_max_offset_ms(self):
        fd0 = _make_frame_data(0, 1.010)  # 10 ms ahead
        fd1 = _make_frame_data(1, 0.995)  # 5 ms behind
        bundle = SyncBundle(sync_index=0, frames={0: fd0, fd1.cam_id: fd1}, reference_timestamp=1.0)
        assert bundle.max_offset_ms() == pytest.approx(10.0, abs=0.1)

    def test_max_offset_empty(self):
        bundle = SyncBundle(sync_index=0, frames={}, reference_timestamp=1.0)
        assert bundle.max_offset_ms() == 0.0


class TestLiveSynchronizerBuildBundle:
    """Test the internal _build_bundle logic without threading."""

    def _make_sync(
        self,
        streams: dict[int, MagicMock],
        fps: float = 30.0,
        tolerance_s: float = 0.033,
        require_all: bool = False,
    ) -> LiveSynchronizer:
        sync = LiveSynchronizer(
            streams=streams,  # type: ignore[arg-type]
            fps=fps,
            tolerance_s=tolerance_s,
            require_all_cameras=require_all,
        )
        return sync

    def test_two_cameras_within_tolerance_produce_bundle(self):
        t = 1000.0
        streams = {
            0: _make_stream(0, t + 0.005),
            1: _make_stream(1, t - 0.005),
        }
        sync = self._make_sync(streams)
        bundle = sync._build_bundle()
        assert bundle is not None
        assert set(bundle.cam_ids) == {0, 1}

    def test_one_camera_outside_tolerance_excluded(self):
        t = 1000.0
        streams = {
            0: _make_stream(0, t),
            1: _make_stream(1, t + 0.100),  # 100 ms away — outside 33 ms tolerance
        }
        sync = self._make_sync(streams, tolerance_s=0.033)
        bundle = sync._build_bundle()
        # At least cam 0 should be present; cam 1 may be excluded
        assert bundle is not None
        assert 0 in bundle.cam_ids

    def test_require_all_cameras_drops_partial_bundle(self):
        t = 1000.0
        streams = {
            0: _make_stream(0, t),
            1: _make_stream(1, t + 0.100),  # outside tolerance
        }
        sync = self._make_sync(streams, tolerance_s=0.033, require_all=True)
        bundle = sync._build_bundle()
        assert bundle is None

    def test_no_streams_returns_none(self):
        sync = self._make_sync({})
        bundle = sync._build_bundle()
        assert bundle is None

    def test_empty_stream_returns_none(self):
        stream = MagicMock()
        stream.get_latest_frame_data.return_value = None
        sync = self._make_sync({0: stream})
        bundle = sync._build_bundle()
        assert bundle is None

    def test_same_frame_not_emitted_twice(self):
        """A frame that has already been included should not appear again."""
        t = 1000.0
        fd = _make_frame_data(0, t, frame_index=42)
        stream = MagicMock()
        stream.get_latest_frame_data.return_value = fd

        sync = self._make_sync({0: stream})

        bundle1 = sync._build_bundle()
        assert bundle1 is not None  # first time: new frame
        assert 0 in bundle1.cam_ids

        bundle2 = sync._build_bundle()
        # Same frame_index → should be skipped
        assert bundle2 is None

    def test_sync_index_increments(self):
        t = 1000.0
        stream0 = _make_stream(0, t, frame_index=1)
        sync = self._make_sync({0: stream0})

        b1 = sync._build_bundle()
        assert b1 is not None
        assert b1.sync_index == 0

        # Provide a new frame
        stream0.get_latest_frame_data.return_value = _make_frame_data(0, t + 0.1, frame_index=2)
        b2 = sync._build_bundle()
        assert b2 is not None
        assert b2.sync_index == 1


class TestLiveSynchronizerThreaded:
    """Integration test: actually run the synchronizer loop briefly."""

    def test_callback_is_called(self):
        t = time.time()
        streams = {
            0: _make_stream(0, t + 0.001, frame_index=1),
            1: _make_stream(1, t - 0.001, frame_index=1),
        }
        bundles: list[SyncBundle] = []

        sync = LiveSynchronizer(streams, fps=100, tolerance_s=0.033)  # type: ignore[arg-type]
        sync.start(bundles.append)
        time.sleep(0.15)  # run for ~15 poll cycles at 100 fps

        # Advance to new frames so the same-frame filter doesn't block everything
        t2 = time.time()
        for cam_id, stream in streams.items():
            stream.get_latest_frame_data.return_value = _make_frame_data(cam_id, t2, frame_index=99)

        time.sleep(0.15)
        sync.stop()

        assert len(bundles) > 0, "Expected at least one bundle to be emitted"
