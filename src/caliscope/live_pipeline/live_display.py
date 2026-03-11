"""Real-time display of live camera feeds with 2D landmark overlays.

Uses ``cv2.imshow`` for low-latency rendering (no Qt dependency).

The :class:`LiveDisplay` class arranges all camera frames into a grid and
draws tracked landmarks on each tile.  It runs on the *calling thread* — you
must call :meth:`update` and :meth:`process_keys` from your main loop.

Keyboard controls
-----------------
* ``q`` — quit (sets the :attr:`quit_requested` flag)
* ``r`` — toggle recording (sets :attr:`recording_requested`)
* ``Space`` — pause / resume tracking (sets :attr:`tracking_paused`)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.live_pipeline.live_tracker_pipeline import TrackedBundle

logger = logging.getLogger(__name__)

# Colour for the HUD text overlays (BGR)
_HUD_COLOR = (0, 255, 0)
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.6
_HUD_THICKNESS = 2

# Minimum tile size (pixels) so the grid is never invisible
_MIN_TILE_W = 320
_MIN_TILE_H = 180


class LiveDisplay:
    """Manages a single ``cv2.imshow`` window showing a grid of camera feeds.

    Parameters
    ----------
    window_name:
        Title of the OpenCV window.
    tile_width, tile_height:
        Resolution of each camera tile in the grid.  Frames are resized to
        fit.
    max_cols:
        Maximum number of columns in the grid.  Rows are added automatically.
    """

    def __init__(
        self,
        window_name: str = "Live Tracking",
        tile_width: int = 640,
        tile_height: int = 360,
        max_cols: int = 2,
    ) -> None:
        self.window_name = window_name
        self.tile_width = max(tile_width, _MIN_TILE_W)
        self.tile_height = max(tile_height, _MIN_TILE_H)
        self.max_cols = max_cols

        self.quit_requested = False
        self.recording_requested = False
        self.tracking_paused = False

        self._fps_counter = _FPSCounter()
        self._last_bundle: TrackedBundle | None = None
        self._window_created = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, bundle: TrackedBundle) -> None:
        """Render *bundle* to the display window.

        Call this from your main loop whenever a new :class:`TrackedBundle`
        arrives.
        """
        self._last_bundle = bundle
        self._fps_counter.tick()

        grid = self._build_grid(bundle)
        self._overlay_hud(grid, bundle)

        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

        cv2.imshow(self.window_name, grid)

    def process_keys(self, wait_ms: int = 1) -> None:
        """Poll for key-presses and update control flags.

        Parameters
        ----------
        wait_ms:
            Milliseconds passed to ``cv2.waitKey``.  Must be ≥ 1.
        """
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFF
        if key == ord("q"):
            self.quit_requested = True
        elif key == ord("r"):
            self.recording_requested = not self.recording_requested
            status = "started" if self.recording_requested else "stopped"
            logger.info("Recording %s by user request", status)
        elif key == ord(" "):
            self.tracking_paused = not self.tracking_paused
            status = "paused" if self.tracking_paused else "resumed"
            logger.info("Tracking %s by user request", status)

    def close(self) -> None:
        """Destroy the OpenCV window."""
        cv2.destroyWindow(self.window_name)
        self._window_created = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self, bundle: TrackedBundle) -> NDArray[Any]:
        """Compose all camera tiles into a single grid image."""
        cam_ids = sorted(bundle.frame_packets.keys())
        n = len(cam_ids)
        if n == 0:
            return np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)

        cols = min(n, self.max_cols)
        rows = (n + cols - 1) // cols

        grid_h = rows * self.tile_height
        grid_w = cols * self.tile_width
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, cam_id in enumerate(cam_ids):
            fp = bundle.frame_packets[cam_id]
            row = i // cols
            col = i % cols
            y0 = row * self.tile_height
            x0 = col * self.tile_width

            tile = self._make_tile(fp, cam_id)
            grid[y0 : y0 + self.tile_height, x0 : x0 + self.tile_width] = tile

        return grid

    def _make_tile(self, fp: Any, cam_id: int) -> NDArray[Any]:
        """Produce a single camera tile with landmarks drawn on it."""
        # Use frame_with_points which already draws circles via FramePacket
        raw = fp.frame_with_points
        if raw is None:
            tile = np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(raw, (self.tile_width, self.tile_height))

        # Camera label
        cv2.putText(
            tile,
            f"Cam {cam_id}",
            (8, 24),
            _HUD_FONT,
            _HUD_SCALE,
            _HUD_COLOR,
            _HUD_THICKNESS,
            cv2.LINE_AA,
        )

        # Draw connections (skeleton edges) if the tracker provides them
        if fp.points is not None and fp.draw_instructions is not None and raw is not None:
            self._draw_connections(tile, fp, raw.shape)

        return tile

    def _draw_connections(
        self,
        tile: NDArray[Any],
        fp: Any,
        original_shape: tuple[int, ...],
    ) -> None:
        """Draw skeleton edges between connected landmarks."""
        # We need the tracker to provide connected_points; if it doesn't, skip.
        try:
            connected = fp.draw_instructions.__self__.get_connected_points()  # type: ignore[attr-defined]
        except AttributeError:
            return

        if not connected:
            return

        orig_h, orig_w = original_shape[:2]
        scale_x = self.tile_width / orig_w
        scale_y = self.tile_height / orig_h

        point_map: dict[int, tuple[int, int]] = {}
        if fp.points is not None:
            for pid, loc in zip(fp.points.point_id, fp.points.img_loc):
                px = int(loc[0] * scale_x)
                py = int(loc[1] * scale_y)
                point_map[int(pid)] = (px, py)

        for id_a, id_b in connected:
            if id_a in point_map and id_b in point_map:
                cv2.line(tile, point_map[id_a], point_map[id_b], (200, 200, 200), 1, cv2.LINE_AA)

    def _overlay_hud(self, grid: NDArray[Any], bundle: TrackedBundle) -> None:
        """Overlay FPS counter and status indicators on the grid."""
        fps = self._fps_counter.fps
        h = grid.shape[0]

        status_parts = [f"FPS: {fps:.1f}", f"Sync: {bundle.sync_index}"]
        if self.tracking_paused:
            status_parts.append("PAUSED")
        if self.recording_requested:
            status_parts.append("[REC]")

        status_text = "  |  ".join(status_parts)
        cv2.putText(
            grid,
            status_text,
            (8, h - 10),
            _HUD_FONT,
            _HUD_SCALE,
            _HUD_COLOR,
            _HUD_THICKNESS,
            cv2.LINE_AA,
        )


class _FPSCounter:
    """Lightweight rolling FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> None:
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0
