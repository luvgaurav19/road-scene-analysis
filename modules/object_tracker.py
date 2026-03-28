"""
object_tracker.py — Object Tracking Module
===========================================
Covers Module 5 of the CV course (Session 3–4):
  - CSRT tracker (primary — high accuracy)
  - KCF tracker (alternative — faster)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Available tracker factories ─────────────────────────────────
_TRACKER_TYPES = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
}


def track_objects(
    video_path: str,
    tracker_type: str = "csrt",
    save: bool = True,
) -> None:
    """Track a user-selected object across video frames.

    On the first frame the user draws a bounding-box ROI. The chosen
    tracker then follows that region throughout the video. FPS and
    tracking status are drawn on every frame.

    Args:
        video_path:   Path to the input video file.
        tracker_type: 'csrt' (default, high accuracy) or 'kcf' (faster).
        save:         Whether to write the annotated video to outputs/.

    Returns:
        None

    Example:
        >>> track_objects("assets/sample_video/traffic.mp4", tracker_type="csrt")
    """
    tracker_type = tracker_type.lower()
    if tracker_type not in _TRACKER_TYPES:
        raise ValueError(
            f"Unknown tracker '{tracker_type}'. Choose from {list(_TRACKER_TYPES)}."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # ── Fallback: generate synthetic frame sequence ─────────
        print(f"[WARN] No video file found at: {video_path}")
        print("       Running on synthetic frame sequence instead.")
        from main import generate_synthetic_road_image
        synth_frames = []
        base = generate_synthetic_road_image(640, 480)
        for i in range(30):
            frame = base.copy()
            # Shift a "vehicle" slightly each frame to simulate motion
            offset = i * 3
            cv2.rectangle(frame, (200 + offset, 300), (260 + offset, 340), (30, 30, 180), -1)
            synth_frames.append(frame)
        # Process synthetic frames without interactive ROI
        print(f"[INFO] Processing {len(synth_frames)} synthetic frames with {tracker_type.upper()} tracker.")
        # Use centre of first vehicle as initial ROI
        bbox = (200, 300, 60, 40)
        tracker = _TRACKER_TYPES[tracker_type]()
        tracker.init(synth_frames[0], bbox)
        for idx, frame in enumerate(synth_frames[1:], 1):
            success, box = tracker.update(frame)
            status = "Tracking" if success else "Lost"
            if success:
                x, y, w, h = (int(v) for v in box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {idx} | {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("[OK] Synthetic tracking demo complete.")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame of the video.")

    # Let the user select an ROI to track
    print("[INFO] Select an ROI on the first frame and press ENTER or SPACE.")
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if bbox == (0, 0, 0, 0):
        print("[WARN] No ROI selected — exiting tracker.")
        cap.release()
        return

    # Initialise tracker
    tracker = _TRACKER_TYPES[tracker_type]()
    tracker.init(frame, bbox)
    print(f"[INFO] Tracking with {tracker_type.upper()} tracker...")

    # Video writer setup
    fps_video = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if save:
        out_path = str(OUTPUT_DIR / f"tracking_{tracker_type}_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps_video, (width, height))
        print(f"[OK] Writing output video -> {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timer_start = time.perf_counter()
        success, box = tracker.update(frame)
        elapsed = time.perf_counter() - timer_start
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        if success:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            status = "Tracking"
            status_color = (0, 255, 0)
        else:
            status = "Lost"
            status_color = (0, 0, 255)

        # Overlay status and FPS
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {tracker_type.upper()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[OK] Tracking complete.")
