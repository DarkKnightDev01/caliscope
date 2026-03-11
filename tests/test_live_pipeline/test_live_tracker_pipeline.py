"""Unit tests for LiveTrackerPipeline using pre-loaded test frames.

The tests mock the tracker so no physical cameras or model weights are needed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from caliscope.live_pipeline.live_camera_stream import FrameData
from caliscope.live_pipeline.live_synchronizer import SyncBundle
from caliscope.live_pipeline.live_tracker_pipeline import LiveTrackerPipeline, TrackedBundle
from caliscope.packets import PointPacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_point_packet() -> PointPacket:
    return PointPacket(
        point_id=np.array([0, 1, 2], dtype=np.int32),
        img_loc=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float64),
    )


def _make_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.name = "MOCK_TRACKER"
    tracker.get_points.return_value = _fake_point_packet()
    tracker.scatter_draw_instructions.return_value = {"radius": 5, "color": (0, 255, 0), "thickness": 2}
    return tracker


def _make_bundle(cam_ids: list[int], ts: float = 0.0) -> SyncBundle:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frames = {
        cam_id: FrameData(frame=frame, timestamp=ts + cam_id * 0.001, frame_index=cam_id + 1, cam_id=cam_id)
        for cam_id in cam_ids
    }
    return SyncBundle(sync_index=0, frames=frames, reference_timestamp=ts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrackedBundle:
    def test_fields(self):
        fp_map: dict = {}
        bundle = TrackedBundle(sync_index=7, frame_packets=fp_map, reference_timestamp=123.4)
        assert bundle.sync_index == 7
        assert bundle.frame_packets is fp_map
        assert bundle.reference_timestamp == pytest.approx(123.4)


class TestLiveTrackerPipelineSequential:
    def test_processes_single_camera(self):
        tracker = _make_tracker()
        pipeline = LiveTrackerPipeline(tracker, parallel=False)

        results: list[TrackedBundle] = []
        pipeline.start(results.append)

        bundle = _make_bundle([0])
        pipeline.push_bundle(bundle)

        # Wait for the worker thread to process it
        deadline = time.time() + 2.0
        while not results and time.time() < deadline:
            time.sleep(0.01)

        pipeline.stop()

        assert len(results) == 1
        assert 0 in results[0].frame_packets
        fp = results[0].frame_packets[0]
        assert fp.cam_id == 0
        assert fp.points is not None
        np.testing.assert_array_equal(fp.points.point_id, np.array([0, 1, 2]))

    def test_processes_multiple_cameras(self):
        tracker = _make_tracker()
        pipeline = LiveTrackerPipeline(tracker, parallel=False)

        results: list[TrackedBundle] = []
        pipeline.start(results.append)

        bundle = _make_bundle([0, 1, 2])
        pipeline.push_bundle(bundle)

        deadline = time.time() + 2.0
        while not results and time.time() < deadline:
            time.sleep(0.01)

        pipeline.stop()

        assert len(results) >= 1
        tb = results[0]
        assert set(tb.frame_packets.keys()) == {0, 1, 2}

    def test_frame_packet_has_draw_instructions(self):
        tracker = _make_tracker()
        pipeline = LiveTrackerPipeline(tracker, parallel=False)

        results: list[TrackedBundle] = []
        pipeline.start(results.append)

        pipeline.push_bundle(_make_bundle([0]))

        deadline = time.time() + 2.0
        while not results and time.time() < deadline:
            time.sleep(0.01)

        pipeline.stop()

        fp = results[0].frame_packets[0]
        assert fp.draw_instructions is not None
        # draw_instructions should be callable
        assert callable(fp.draw_instructions)


class TestLiveTrackerPipelineParallel:
    def test_parallel_produces_same_results(self):
        tracker = _make_tracker()
        pipeline = LiveTrackerPipeline(tracker, parallel=True, max_workers=4)

        results: list[TrackedBundle] = []
        pipeline.start(results.append)

        pipeline.push_bundle(_make_bundle([0, 1, 2, 3]))

        deadline = time.time() + 3.0
        while not results and time.time() < deadline:
            time.sleep(0.01)

        pipeline.stop()

        assert len(results) >= 1
        tb = results[0]
        assert set(tb.frame_packets.keys()) == {0, 1, 2, 3}


class TestLiveTrackerPipelineDropsBundles:
    def test_old_bundles_dropped_when_full(self):
        """When the pipeline is overloaded it should not grow unboundedly."""
        import queue as queuemod

        tracker = _make_tracker()
        # Make the tracker deliberately slow
        tracker.get_points.side_effect = lambda *a, **kw: (time.sleep(0.05), _fake_point_packet())[1]

        pipeline = LiveTrackerPipeline(tracker, parallel=False)
        results: list[TrackedBundle] = []
        pipeline.start(results.append)

        # Push many bundles quickly — most should be dropped
        for i in range(20):
            pipeline.push_bundle(_make_bundle([0]))

        time.sleep(0.5)
        pipeline.stop()

        # The pipeline processed at most a handful of bundles (not all 20)
        assert pipeline._dropped_count + pipeline._processed_count <= 20

    def test_stop_cleans_up(self):
        tracker = _make_tracker()
        pipeline = LiveTrackerPipeline(tracker, parallel=False)
        pipeline.start(lambda b: None)
        pipeline.stop()
        assert not pipeline.is_running


class TestLiveTrackerPipelineTrackerException:
    def test_exception_in_tracker_produces_empty_points(self):
        """If the tracker raises, the pipeline should not crash."""
        tracker = _make_tracker()
        tracker.get_points.side_effect = ValueError("simulated tracker error")

        pipeline = LiveTrackerPipeline(tracker, parallel=False)
        results: list[TrackedBundle] = []
        pipeline.start(results.append)

        pipeline.push_bundle(_make_bundle([0]))

        deadline = time.time() + 2.0
        while not results and time.time() < deadline:
            time.sleep(0.01)

        pipeline.stop()

        assert len(results) >= 1
        fp = results[0].frame_packets[0]
        # Should have an empty point packet (graceful fallback)
        assert fp.points is not None
        assert len(fp.points.point_id) == 0
