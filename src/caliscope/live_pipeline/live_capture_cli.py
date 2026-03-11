"""CLI entry point for the live motion capture pipeline.

Usage examples::

    # List available webcam devices
    python -m caliscope.live_pipeline.live_capture_cli detect

    # Live tracking with 2D landmark preview
    python -m caliscope.live_pipeline.live_capture_cli track \\
        --cameras 0 1 2 3 --resolution 1920x1080 --fps 30

    # Live tracking with simultaneous recording
    python -m caliscope.live_pipeline.live_capture_cli track \\
        --cameras 0 1 2 --record --output ./my_session

    # Preview raw camera feeds without tracking
    python -m caliscope.live_pipeline.live_capture_cli preview \\
        --cameras 0 1 --resolution 1280x720
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_resolution(value: str) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` string into an (int, int) tuple."""
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Resolution must be in WIDTHxHEIGHT format, got: {value!r}"
        )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> None:
    """Detect and print available webcam devices."""
    from caliscope.live_pipeline.camera_detector import detect_cameras

    logging.basicConfig(level=logging.WARNING)
    print("Scanning for cameras…")
    cameras = detect_cameras(max_index=args.max_index, probe_resolutions=not args.no_probe)

    if not cameras:
        print("No cameras detected.")
        return

    print(f"\nFound {len(cameras)} camera(s):\n")
    for cam in cameras:
        print(f"  {cam}")
    print()


def cmd_preview(args: argparse.Namespace) -> None:
    """Display raw (untracked) feeds from selected cameras."""
    import cv2

    from caliscope.live_pipeline.live_camera_stream import LiveCameraStream

    width, height = args.resolution
    streams: dict[int, LiveCameraStream] = {}

    for cam_id, dev_idx in enumerate(args.cameras):
        stream = LiveCameraStream(dev_idx, cam_id=cam_id, width=width, height=height, fps=args.fps)
        streams[cam_id] = stream

    print(f"Starting preview for cameras {args.cameras} — press 'q' to quit")

    for s in streams.values():
        s.start()

    tile_w = max(640, width // 2)
    tile_h = max(360, height // 2)

    try:
        while True:
            tiles = []
            for cam_id in sorted(streams):
                frame, _ = streams[cam_id].get_latest_frame()
                if frame is None:
                    tile = _blank_tile(tile_w, tile_h, f"Cam {cam_id} — waiting…")
                else:
                    tile = cv2.resize(frame, (tile_w, tile_h))
                    _put_label(tile, f"Cam {cam_id}")
                tiles.append(tile)

            grid = _make_grid(tiles, max_cols=2)
            cv2.imshow("Preview", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        for s in streams.values():
            s.stop()
        cv2.destroyAllWindows()


def cmd_track(args: argparse.Namespace) -> None:
    """Run live 2D landmark tracking with optional recording."""
    import cv2

    from caliscope.live_pipeline.camera_detector import detect_cameras
    from caliscope.live_pipeline.live_camera_stream import LiveCameraStream
    from caliscope.live_pipeline.live_display import LiveDisplay
    from caliscope.live_pipeline.live_recorder import LiveRecorder
    from caliscope.live_pipeline.live_synchronizer import LiveSynchronizer
    from caliscope.live_pipeline.live_tracker_pipeline import LiveTrackerPipeline, TrackedBundle
    from caliscope.trackers import tracker_registry

    width, height = args.resolution
    fps = args.fps

    # ------------------------------------------------------------------
    # Build tracker
    # ------------------------------------------------------------------
    tracker_key = args.tracker
    if not tracker_registry.is_registered(tracker_key):
        available = tracker_registry.available_names()
        print(
            f"ERROR: Unknown tracker {tracker_key!r}.\n"
            f"Available trackers: {', '.join(available)}"
        )
        sys.exit(1)

    tracker = tracker_registry.create(tracker_key)
    logger.info("Using tracker: %s", tracker.name)

    # ------------------------------------------------------------------
    # Open camera streams
    # ------------------------------------------------------------------
    streams: dict[int, LiveCameraStream] = {}
    for cam_id, dev_idx in enumerate(args.cameras):
        stream = LiveCameraStream(dev_idx, cam_id=cam_id, width=width, height=height, fps=fps)
        try:
            stream.start()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        streams[cam_id] = stream

    # Allow cameras to warm up
    time.sleep(0.5)

    # ------------------------------------------------------------------
    # Build pipeline components
    # ------------------------------------------------------------------
    synchronizer = LiveSynchronizer(streams, fps=fps, tolerance_s=args.tolerance)
    pipeline = LiveTrackerPipeline(tracker, parallel=args.parallel)
    display = LiveDisplay(
        tile_width=max(320, width // 3),
        tile_height=max(180, height // 3),
        max_cols=2,
    )

    # Optional recorder
    recorder: LiveRecorder | None = None
    if args.record:
        output_dir = Path(args.output) if args.output else Path(f"live_session_{int(time.time())}")
        recorder = LiveRecorder(output_dir, tracker_name=tracker.name, fps=fps)
        recorder.start()
        display.recording_requested = True

    latest_bundle: list[TrackedBundle] = []

    def on_tracked(bundle: TrackedBundle) -> None:
        latest_bundle.clear()
        latest_bundle.append(bundle)
        if recorder is not None and recorder.is_recording:
            recorder.record(bundle)

    pipeline.start(on_tracked)
    synchronizer.start(pipeline.push_bundle)

    print("Live tracking started.  Keyboard: q=quit  r=toggle record  space=pause")

    try:
        while not display.quit_requested:
            if latest_bundle:
                bundle = latest_bundle[-1]

                if not display.tracking_paused:
                    display.update(bundle)
                else:
                    # Show last frame with "PAUSED" overlay but don't update landmarks
                    display.update(bundle)

                # Sync recording toggle with display flag
                if recorder is not None:
                    if display.recording_requested and not recorder.is_recording:
                        recorder.start()
                    elif not display.recording_requested and recorder.is_recording:
                        recorder.stop()

            display.process_keys(wait_ms=1)
    finally:
        synchronizer.stop()
        pipeline.stop()
        tracker.cleanup()
        if recorder is not None and recorder.is_recording:
            recorder.stop()
        for s in streams.values():
            s.stop()
        display.close()
        cv2.destroyAllWindows()

    print("Live tracking session ended.")


# ---------------------------------------------------------------------------
# Grid / display utilities (used by preview and track commands)
# ---------------------------------------------------------------------------


def _blank_tile(w: int, h: int, label: str = "") -> "Any":  # type: ignore[return]
    import cv2
    import numpy as np

    tile = np.zeros((h, w, 3), dtype=np.uint8)
    if label:
        cv2.putText(tile, label, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return tile


def _put_label(tile: "Any", label: str) -> None:
    import cv2

    cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def _make_grid(tiles: list, max_cols: int = 2) -> "Any":
    import cv2
    import numpy as np

    if not tiles:
        return np.zeros((180, 320, 3), dtype=np.uint8)

    cols = min(len(tiles), max_cols)
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    return grid


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m caliscope.live_pipeline.live_capture_cli",
        description="Real-time live motion capture pipeline for caliscope.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- detect ----
    p_detect = sub.add_parser("detect", help="Detect available webcam devices")
    p_detect.add_argument("--max-index", type=int, default=10, help="Highest device index to probe")
    p_detect.add_argument("--no-probe", action="store_true", help="Skip resolution probing (faster)")

    # ---- preview ----
    p_preview = sub.add_parser("preview", help="Show raw camera feeds without tracking")
    p_preview.add_argument("--cameras", type=int, nargs="+", default=[0], metavar="IDX", help="Device indices")
    p_preview.add_argument("--resolution", type=_parse_resolution, default=(1920, 1080), metavar="WxH")
    p_preview.add_argument("--fps", type=int, default=30)

    # ---- track ----
    p_track = sub.add_parser("track", help="Run live 2D tracking with landmark overlay")
    p_track.add_argument("--cameras", type=int, nargs="+", default=[0], metavar="IDX", help="Device indices")
    p_track.add_argument("--resolution", type=_parse_resolution, default=(1920, 1080), metavar="WxH")
    p_track.add_argument("--fps", type=int, default=30)
    p_track.add_argument(
        "--tracker",
        default="SIMPLE_HOLISTIC",
        help="Tracker key (see 'detect --list-trackers').  Default: SIMPLE_HOLISTIC",
    )
    p_track.add_argument(
        "--tolerance",
        type=float,
        default=0.033,
        metavar="SECS",
        help="Sync tolerance in seconds (default 0.033 = 33 ms)",
    )
    p_track.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel tracking")
    p_track.add_argument("--record", action="store_true", help="Record video and landmark CSVs")
    p_track.add_argument("--output", default=None, help="Output directory for recordings")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "preview":
        cmd_preview(args)
    elif args.command == "track":
        cmd_track(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
