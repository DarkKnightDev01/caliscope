"""Real-time live motion capture pipeline for caliscope.

Provides a full live streaming pipeline that captures frames from multiple
Iriun virtual webcams simultaneously, synchronizes them by wall-clock time,
runs caliscope's tracker on each frame in real-time, and displays a live
preview window with 2D landmark overlays.

Usage (CLI)::

    # Detect available cameras
    python -m caliscope.live_pipeline.live_capture_cli detect

    # Run live tracking with preview
    python -m caliscope.live_pipeline.live_capture_cli track --cameras 0 1 2 3

    # Run live tracking with recording
    python -m caliscope.live_pipeline.live_capture_cli track --cameras 0 1 2 --record --output ./session
"""
