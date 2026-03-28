"""
main.py — Real-Time Road Scene Analysis System
===============================================
Interactive CLI entry point that demonstrates every module
of the CV pipeline. Includes a fully synthetic demo mode so
the project works out-of-the-box without any external dataset.

Author : VIT Bhopal, B.Tech CSE, 3rd Year
Course : Computer Vision (CSE3005/CSE3006)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Ensure the project root is on sys.path ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import preprocessing
from modules import edge_detection
from modules import lane_detection
from modules import corner_detection
from modules import object_detector
from modules import object_tracker
from modules import classifier

# ── Helpers ─────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_IMG_DIR = PROJECT_ROOT / "assets" / "sample_images"
SAMPLE_VID_DIR = PROJECT_ROOT / "assets" / "sample_video"


def _get_image_path() -> str:
    """Prompt the user for an image path, defaulting to a sample if available."""
    samples = list(SAMPLE_IMG_DIR.glob("*")) if SAMPLE_IMG_DIR.exists() else []
    if samples:
        print(f"\n  Available samples in assets/sample_images/:")
        for i, s in enumerate(samples, 1):
            print(f"    {i}. {s.name}")
        choice = input("  Enter image path (or press ENTER for first sample): ").strip()
        if not choice:
            return str(samples[0])
        return choice
    return input("  Enter image path: ").strip()


def _get_video_path() -> str:
    """Prompt the user for a video path, defaulting to a sample if available."""
    samples = list(SAMPLE_VID_DIR.glob("*")) if SAMPLE_VID_DIR.exists() else []
    if samples:
        print(f"\n  Available samples in assets/sample_video/:")
        for i, s in enumerate(samples, 1):
            print(f"    {i}. {s.name}")
        choice = input("  Enter video path (or press ENTER for first sample): ").strip()
        if not choice:
            return str(samples[0])
        return choice
    return input("  Enter video path: ").strip()


def _load_image(path: str) -> np.ndarray:
    """Load an image from disk and raise a clear error if it fails."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


# ── Synthetic Road Image Generator ──────────────────────────────
def generate_synthetic_road_image(
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Create a synthetic road image using basic OpenCV drawing primitives.

    The image contains:
      - A blue sky gradient at the top
      - A grey road surface with perspective lines converging to a vanishing point
      - White dashed lane markings
      - Simple rectangular "vehicle" shapes
      - A green roadside strip

    Args:
        width:  Image width in pixels (default 640).
        height: Image height in pixels (default 480).

    Returns:
        Synthetic road image as a BGR np.ndarray.

    Example:
        >>> road_img = generate_synthetic_road_image()
        >>> cv2.imshow("Synthetic Road", road_img)
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # ── Sky gradient (blue -> lighter blue) ──────────────────────
    horizon = int(height * 0.45)
    for y in range(horizon):
        ratio = y / horizon
        b = int(200 + 55 * ratio)
        g = int(150 + 80 * ratio)
        r = int(80 + 50 * ratio)
        img[y, :] = (b, g, r)

    # ── Ground / grass ──────────────────────────────────────────
    img[horizon:, :] = (34, 120, 50)  # dark green grass

    # ── Road surface ────────────────────────────────────────────
    vanish_x, vanish_y = width // 2, horizon
    road_pts = np.array([
        [int(width * 0.15), height],
        [vanish_x - 5, vanish_y],
        [vanish_x + 5, vanish_y],
        [int(width * 0.85), height],
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], (80, 80, 80))  # asphalt grey

    # ── Lane markings (dashed white lines) ──────────────────────
    # Left lane
    for t in np.linspace(0.1, 0.95, 12):
        y1 = int(vanish_y + (height - vanish_y) * t)
        y2 = int(vanish_y + (height - vanish_y) * min(t + 0.04, 1.0))
        x1 = int(vanish_x + (width * 0.35 - vanish_x) * t)
        x2 = int(vanish_x + (width * 0.35 - vanish_x) * min(t + 0.04, 1.0))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Right lane
    for t in np.linspace(0.1, 0.95, 12):
        y1 = int(vanish_y + (height - vanish_y) * t)
        y2 = int(vanish_y + (height - vanish_y) * min(t + 0.04, 1.0))
        x1 = int(vanish_x + (width * 0.65 - vanish_x) * t)
        x2 = int(vanish_x + (width * 0.65 - vanish_x) * min(t + 0.04, 1.0))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ── Centre line (solid yellow) ──────────────────────────────
    for t in np.linspace(0.05, 0.95, 40):
        y = int(vanish_y + (height - vanish_y) * t)
        x = vanish_x
        cv2.circle(img, (x, y), 1, (0, 200, 255), -1)

    # ── Simple "vehicles" ───────────────────────────────────────
    # Vehicle 1 — closer, right lane
    v1_x, v1_y = int(width * 0.58), int(height * 0.72)
    cv2.rectangle(img, (v1_x, v1_y), (v1_x + 50, v1_y + 30), (30, 30, 180), -1)
    cv2.rectangle(img, (v1_x + 5, v1_y + 2), (v1_x + 45, v1_y + 15), (60, 60, 200), -1)

    # Vehicle 2 — farther, left lane
    v2_x, v2_y = int(width * 0.38), int(height * 0.58)
    cv2.rectangle(img, (v2_x, v2_y), (v2_x + 30, v2_y + 18), (180, 30, 30), -1)
    cv2.rectangle(img, (v2_x + 3, v2_y + 2), (v2_x + 27, v2_y + 10), (200, 60, 60), -1)

    # Vehicle 3 — small, near vanishing point
    v3_x, v3_y = int(width * 0.48), int(height * 0.48)
    cv2.rectangle(img, (v3_x, v3_y), (v3_x + 15, v3_y + 10), (50, 50, 50), -1)

    return img


# ── Full Synthetic Demo ─────────────────────────────────────────
def generate_synthetic_demo() -> None:
    """Run the complete CV pipeline on a synthetically generated road image.

    This function creates a synthetic road scene using OpenCV drawing,
    then exercises every module in sequence so the project works
    out-of-the-box without any external dataset or video.

    Returns:
        None (results are displayed and saved to outputs/).
    """
    # Use non-interactive matplotlib backend so plots just save to disk
    matplotlib.use("Agg")

    print("\n" + "=" * 50)
    print("  SYNTHETIC DEMO — Full Pipeline")
    print("=" * 50)

    # 1. Generate synthetic road image
    print("\n[1/7] Generating synthetic road image...")
    road_img = generate_synthetic_road_image()
    save_path = OUTPUT_DIR / "synthetic_road.png"
    cv2.imwrite(str(save_path), road_img)
    print(f"  -> Saved {save_path}")

    # 2. Preprocessing
    print("\n[2/7] Running preprocessing module...")
    preprocessing.remove_noise(road_img, save=True)
    preprocessing.apply_histogram_equalization(road_img, save=True)
    preprocessing.apply_morphological_ops(road_img, save=True)
    preprocessing.apply_log_transform(road_img, save=True)
    preprocessing.flip_and_contrast(road_img, save=True)

    # 3. Edge Detection
    print("\n[3/7] Running edge detection module...")
    edge_detection.sobel_edge_detection(road_img, save=True)
    edge_detection.canny_edge_detection(road_img, low=50, high=150, save=True)
    edge_detection.compare_edge_methods(road_img, save=True)

    # 4. Lane Detection (single image)
    print("\n[4/7] Running lane detection module...")
    lane_detection.detect_lanes(road_img, save=True)

    # 5. Corner Detection
    print("\n[5/7] Running corner detection module...")
    corner_detection.harris_corner_detection(road_img, save=True)
    corner_detection.shi_tomasi_corners(road_img, save=True)

    # 6. Object Detection (contour fallback — no DNN model needed)
    print("\n[6/7] Running object detection module...")
    object_detector.detect_objects(road_img, confidence_threshold=0.5, save=True)

    # 7. KNN Classifier (synthetic data)
    print("\n[7/7] Running KNN classifier module...")
    model, acc, class_names = classifier.train_knn_classifier(save=True)
    label = classifier.predict(road_img, model, class_names)
    print(f"  -> Predicted class for synthetic road image: {label}")

    print("\n" + "=" * 50)
    print("  DEMO COMPLETE — Check the outputs/ folder!")
    print("=" * 50)


# ── CLI Menu ────────────────────────────────────────────────────
MENU = """
========================================
  Real-Time Road Scene Analysis System
========================================
  1. Image Preprocessing
  2. Edge Detection
  3. Lane Detection
  4. Corner Detection
  5. Object Detection
  6. Object Tracking
  7. KNN Image Classification
  8. Run Full Synthetic Demo
  0. Exit
========================================
"""


def main() -> None:
    """Interactive CLI menu — each option calls the respective module."""
    while True:
        print(MENU)
        choice = input("Select an option [0-8]: ").strip()

        try:
            if choice == "1":
                img = _load_image(_get_image_path())
                preprocessing.remove_noise(img)
                preprocessing.apply_histogram_equalization(img)
                preprocessing.apply_morphological_ops(img)
                preprocessing.apply_log_transform(img)
                preprocessing.flip_and_contrast(img)

            elif choice == "2":
                img = _load_image(_get_image_path())
                edge_detection.sobel_edge_detection(img)
                edge_detection.canny_edge_detection(img)
                edge_detection.compare_edge_methods(img)

            elif choice == "3":
                sub = input("  (a) Single image  (b) Video? [a/b]: ").strip().lower()
                if sub == "b":
                    lane_detection.process_video_for_lanes(_get_video_path())
                else:
                    img = _load_image(_get_image_path())
                    lane_detection.detect_lanes(img)

            elif choice == "4":
                img = _load_image(_get_image_path())
                corner_detection.harris_corner_detection(img)
                corner_detection.shi_tomasi_corners(img)

            elif choice == "5":
                img = _load_image(_get_image_path())
                object_detector.detect_objects(img)

            elif choice == "6":
                tracker_type = input("  Tracker type [csrt/kcf] (default csrt): ").strip() or "csrt"
                object_tracker.track_objects(_get_video_path(), tracker_type=tracker_type)

            elif choice == "7":
                data_dir = input("  Dataset directory (ENTER for synthetic): ").strip() or None
                model, acc, names = classifier.train_knn_classifier(data_dir)
                img_path = input("  Image to classify (ENTER to skip): ").strip()
                if img_path:
                    img = _load_image(img_path)
                    label = classifier.predict(img, model, names)
                    print(f"  -> Prediction: {label}")

            elif choice == "8":
                generate_synthetic_demo()

            elif choice == "0":
                print("Goodbye!")
                break

            else:
                print("Invalid option. Please enter 0-8.")

        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")

        input("\nPress ENTER to continue...")


if __name__ == "__main__":
    main()
