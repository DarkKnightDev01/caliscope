"""CLI entry point for the live camera capture module.

Run with::

    python -m caliscope.live_capture.live_capture_cli <command> [options]

Commands
--------
detect
    List all detected webcam devices and their capabilities.

preview
    Show a live preview window for every detected (or specified) camera.
    Press **q** to quit.

record
    Record from one or more cameras to ``cam_<N>.mp4`` files.

Examples
--------
::

    # List available cameras
    python -m caliscope.live_capture.live_capture_cli detect

    # Preview all cameras
    python -m caliscope.live_capture.live_capture_cli preview

    # Record 30 s of 1080p @ 30 fps
    python -m caliscope.live_capture.live_capture_cli record \\
        --output ./my_recording --duration 30 --resolution 1920x1080 --fps 30

    # 4K capture from specific cameras
    python -m caliscope.live_capture.live_capture_cli record \\
        --cameras 0 1 2 3 --output ./my_recording --resolution 3840x2160

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from caliscope.live_capture.camera_detector import CameraDetector
from caliscope.live_capture.live_stream import LiveStream
from caliscope.live_capture.multi_camera_recorder import MultiCameraRecorder
from caliscope.logger import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_resolution(value: str) -> tuple[int, int]:
    """Parse a ``"WxH"`` resolution string into ``(width, height)``."""
    try:
        w_str, h_str = value.lower().split("x")
        return int(w_str), int(h_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Resolution must be in WxH format (e.g. 1920x1080), got: {value!r}"
        )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> int:
    """Detect and print available cameras."""
    detector = CameraDetector(max_index=args.max_index, probe_resolutions=True)
    cameras = detector.detect()

    if not cameras:
        print("No cameras detected.")
        return 0

    print(f"Found {len(cameras)} camera(s):\n")
    for cam in cameras:
        print(f"  {cam}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Show a live preview from all (or specified) cameras.

    Press **q** in any preview window to stop.
    """
    if args.cameras:
        indices = args.cameras
    else:
        detector = CameraDetector(max_index=args.max_index, probe_resolutions=False)
        indices = detector.detect_indices()

    if not indices:
        print("No cameras to preview.")
        return 1

    width, height = _parse_resolution(args.resolution) if args.resolution else (0, 0)

    streams: list[LiveStream] = []
    try:
        for idx in indices:
            s = LiveStream(camera_index=idx, width=width, height=height)
            s.open()
            s.start()
            streams.append(s)

        print("Previewing cameras. Press 'q' in any window to stop.")
        while True:
            for s in streams:
                frame, _ = s.get_latest_frame()
                if frame is not None:
                    win_name = f"Camera {s.camera_index}"
                    cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()

    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Record from one or more cameras to .mp4 files."""
    if args.cameras:
        indices = args.cameras
    else:
        detector = CameraDetector(max_index=args.max_index, probe_resolutions=False)
        indices = detector.detect_indices()

    if not indices:
        print("No cameras found. Use --cameras to specify indices explicitly.")
        return 1

    output_dir = Path(args.output)
    width, height = _parse_resolution(args.resolution) if args.resolution else (0, 0)
    fps: float = args.fps
    duration: float | None = args.duration

    print(
        f"Recording {len(indices)} camera(s) → {output_dir}"
        + (f" for {duration:.0f} s" if duration else " (press Ctrl-C to stop)")
        + f" at {fps:.0f} fps"
        + (f" {width}x{height}" if width and height else " (native resolution)")
    )

    recorder = MultiCameraRecorder(
        camera_indices=indices,
        output_dir=output_dir,
        width=width,
        height=height,
        fps=fps,
    )

    recorder.start_recording()
    try:
        if duration is not None:
            time.sleep(duration)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping recording…")
    finally:
        recorder.stop_recording()

    print(f"Done. Files saved to: {output_dir.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caliscope.live_capture",
        description="Live camera capture for caliscope",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=10,
        metavar="N",
        help="Highest device index to probe when auto-detecting cameras (default: 10).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ----- detect -----
    sub.add_parser("detect", help="List available webcam devices.")

    # ----- preview -----
    p_preview = sub.add_parser("preview", help="Show live preview from camera(s).")
    p_preview.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="Camera indices to preview. Auto-detects if omitted.",
    )
    p_preview.add_argument(
        "--resolution",
        type=str,
        metavar="WxH",
        help="Preview resolution (e.g. 1920x1080). Defaults to device native.",
    )

    # ----- record -----
    p_record = sub.add_parser("record", help="Record synchronized video from camera(s).")
    p_record.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="Camera indices to record. Auto-detects if omitted.",
    )
    p_record.add_argument(
        "--output",
        type=str,
        default="./caliscope_recording",
        metavar="DIR",
        help="Output directory (default: ./caliscope_recording).",
    )
    p_record.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Recording duration in seconds. Records until Ctrl-C if omitted.",
    )
    p_record.add_argument(
        "--resolution",
        type=str,
        default=None,
        metavar="WxH",
        help="Recording resolution (e.g. 1920x1080 or 3840x2160). Defaults to device native.",
    )
    p_record.add_argument(
        "--fps",
        type=float,
        default=30.0,
        metavar="FPS",
        help="Target frames per second (default: 30).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "detect": cmd_detect,
        "preview": cmd_preview,
        "record": cmd_record,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
