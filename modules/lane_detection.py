"""
lane_detection.py — Lane Detection Module
==========================================
Covers Module 4 of the CV course:
  - Hough Line Transform (Session 1–2)
  - Full lane-detection pipeline on images and video
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _region_of_interest(image: np.ndarray) -> np.ndarray:
    """Create a trapezoidal ROI mask for the lower half of the frame.

    The trapezoid is designed to isolate the road surface where lane
    markings are expected, discarding sky and roadside distractions.

    Args:
        image: Single-channel (grayscale/edge) image.

    Returns:
        Masked image with only the ROI pixels retained.
    """
    height, width = image.shape[:2]

    # Define a trapezoid roughly covering the road ahead
    vertices = np.array(
        [
            [
                (int(width * 0.1), height),            # bottom-left
                (int(width * 0.45), int(height * 0.6)),  # top-left
                (int(width * 0.55), int(height * 0.6)),  # top-right
                (int(width * 0.9), height),            # bottom-right
            ]
        ],
        dtype=np.int32,
    )

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def _draw_lines(
    image: np.ndarray,
    lines: Optional[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw detected lines on a copy of the image.

    Args:
        image:     Original BGR image.
        lines:     Output of cv2.HoughLinesP — shape (N, 1, 4).
        color:     BGR colour for the lines.
        thickness: Line thickness in pixels.

    Returns:
        Image with lines overlaid.
    """
    result = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result


# ── Main lane-detection pipeline ────────────────────────────────
def detect_lanes(image: np.ndarray, save: bool = True) -> np.ndarray:
    """Full lane-detection pipeline on a single image.

    Steps:
      1. Convert to grayscale
      2. Gaussian blur (5×5)
      3. Canny edge detection (50, 150)
      4. Region-of-interest masking (trapezoid)
      5. Probabilistic Hough Line Transform
      6. Draw lane lines on the original frame

    Args:
        image: Input BGR image.
        save:  Whether to save intermediate and final result plots.

    Returns:
        BGR image with detected lane lines drawn.

    Example:
        >>> result = detect_lanes(cv2.imread("highway.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    # 1. Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 4. ROI masking — keep only the road region
    roi = _region_of_interest(edges)

    # 5. Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        roi,
        rho=1,                   # distance resolution (pixels)
        theta=np.pi / 180,       # angle resolution (radians)
        threshold=50,            # min number of votes
        minLineLength=50,        # min line length (pixels)
        maxLineGap=150,          # max gap between segments
    )

    # 6. Draw lines on the original
    result = _draw_lines(image, lines, color=(0, 255, 0), thickness=3)

    if save:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        steps = [
            (image, "1. Original"),
            (gray, "2. Grayscale"),
            (blurred, "3. Blurred"),
            (edges, "4. Canny Edges"),
            (roi, "5. ROI Masked"),
            (result, "6. Detected Lanes"),
        ]
        for ax, (img, title) in zip(axes.flat, steps):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.suptitle("Lane Detection Pipeline", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "lane_detection_pipeline.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return result


# ── Video processing ────────────────────────────────────────────
def process_video_for_lanes(video_path: str, save: bool = True) -> None:
    """Apply lane detection frame-by-frame on a video file.

    Opens the video, processes each frame through `detect_lanes`,
    and displays the result in a window. Press 'q' to quit.

    If `save` is True, the processed video is also written to
    ``outputs/lane_detection_output.avi``.

    Args:
        video_path: Path to the input video file.
        save:       Whether to write the output video to disk.

    Returns:
        None

    Example:
        >>> process_video_for_lanes("assets/sample_video/highway.mp4")
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Read video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        out_path = str(OUTPUT_DIR / "lane_detection_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[OK] Writing output video -> {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame (save=False to avoid per-frame plots)
        lane_frame = detect_lanes(frame, save=False)

        if writer is not None:
            writer.write(lane_frame)

        cv2.imshow("Lane Detection", lane_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[OK] Video processing complete.")


# ── Parameter Comparison ────────────────────────────────────────
def compare_parameters(image: np.ndarray, save: bool = True) -> dict[str, dict]:
    """Compare Standard Hough Transform vs Probabilistic Hough Transform.

    Reports detection time and number of lines for each method.

    Args:
        image: Input BGR image.
        save:  Whether to save the comparison plot.

    Returns:
        Dict mapping method name to {'time_ms': float, 'line_count': int}.

    Example:
        >>> results = compare_parameters(cv2.imread("road.jpg"))
    """
    import time

    if image is None:
        raise FileNotFoundError("Input image is None -- check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    roi = _region_of_interest(edges)

    results = {}

    # 1. Standard Hough Transform
    t0 = time.perf_counter()
    lines_std = cv2.HoughLines(roi, 1, np.pi / 180, 100)
    t_std = (time.perf_counter() - t0) * 1000
    std_count = len(lines_std) if lines_std is not None else 0
    results["Standard Hough"] = {"time_ms": round(t_std, 2), "line_count": std_count}

    # 2. Probabilistic Hough Transform
    t0 = time.perf_counter()
    lines_prob = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    t_prob = (time.perf_counter() - t0) * 1000
    prob_count = len(lines_prob) if lines_prob is not None else 0
    results["Probabilistic Hough"] = {"time_ms": round(t_prob, 2), "line_count": prob_count}

    if save:
        # Draw results
        img_std = image.copy()
        if lines_std is not None:
            for line in lines_std:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(img_std, pt1, pt2, (0, 0, 255), 2)

        img_prob = _draw_lines(image, lines_prob, color=(0, 255, 0), thickness=2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Standard Hough\n{std_count} lines, {t_std:.1f}ms", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(img_prob, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Probabilistic Hough\n{prob_count} lines, {t_prob:.1f}ms", fontsize=11)
        axes[2].axis("off")

        plt.suptitle("Hough Transform Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "lane_comparison.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return results
