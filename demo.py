"""
demo.py -- Full Pipeline Demo (Zero Arguments)
================================================
Run this script to execute the complete CV pipeline on synthetic data.
Produces 14+ output files and a composite results grid.

Usage:  python demo.py
"""

from __future__ import annotations

import os
import sys
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Non-interactive matplotlib
import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np


def _step(number: int, total: int, label: str):
    """Print a step header."""
    print(f"  [{number}/{total}] {label:42s}", end="", flush=True)


def _done(elapsed: float):
    """Print step completion."""
    print(f" Done ({elapsed:.1f}s)")


def run_demo():
    """Execute the complete pipeline on a synthetic road image."""
    TOTAL = 7

    print()
    print("=" * 52)
    print("   Road Scene Analysis -- Full Pipeline Demo")
    print("=" * 52)
    print()

    # Import modules
    from main import generate_synthetic_road_image
    from modules import preprocessing, edge_detection, lane_detection
    from modules import corner_detection, object_detector, classifier
    from modules.metrics import generate_metrics_report
    from results_visualizer import create_composite_image

    # Generate synthetic input
    print("  Generating synthetic road image (640x480)...")
    road_img = generate_synthetic_road_image(640, 480)
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "synthetic_road.png"), road_img)
    print()

    # ── Step 1: Preprocessing ───────────────────────────────
    _step(1, TOTAL, "Running preprocessing pipeline...")
    t0 = time.perf_counter()
    preprocessing.remove_noise(road_img, save=True)
    preprocessing.apply_histogram_equalization(road_img, save=True)
    preprocessing.apply_morphological_ops(road_img, save=True)
    preprocessing.apply_log_transform(road_img, save=True)
    preprocessing.flip_and_contrast(road_img, save=True)
    _done(time.perf_counter() - t0)

    # ── Step 2: Edge Detection ──────────────────────────────
    _step(2, TOTAL, "Running edge detection...")
    t0 = time.perf_counter()
    edge_detection.sobel_edge_detection(road_img, save=True)
    edge_detection.canny_edge_detection(road_img, save=True)
    edge_detection.compare_edge_methods(road_img, save=True)
    _done(time.perf_counter() - t0)

    # ── Step 3: Lane Detection ──────────────────────────────
    _step(3, TOTAL, "Running lane detection...")
    t0 = time.perf_counter()
    lane_detection.detect_lanes(road_img, save=True)
    _done(time.perf_counter() - t0)

    # ── Step 4: Corner Detection ────────────────────────────
    _step(4, TOTAL, "Running corner detection...")
    t0 = time.perf_counter()
    corner_detection.harris_corner_detection(road_img, save=True)
    corner_detection.shi_tomasi_corners(road_img, save=True)
    _done(time.perf_counter() - t0)

    # ── Step 5: Object Detection ────────────────────────────
    _step(5, TOTAL, "Running object detection...")
    t0 = time.perf_counter()
    object_detector.detect_objects(road_img, save=True)
    _done(time.perf_counter() - t0)

    # ── Step 6: KNN Classification ──────────────────────────
    _step(6, TOTAL, "Running KNN classification...")
    t0 = time.perf_counter()
    model, acc, names = classifier.train_knn_classifier(save=True)
    label = classifier.predict(road_img, model, names)
    _done(time.perf_counter() - t0)
    print(f"         KNN accuracy: {acc:.2%} | Predicted: {label}")

    # ── Step 7: Metrics & Composite ─────────────────────────
    _step(7, TOTAL, "Generating metrics report...")
    t0 = time.perf_counter()
    generate_metrics_report(road_img)
    create_composite_image()
    _done(time.perf_counter() - t0)

    # ── Summary ─────────────────────────────────────────────
    output_files = [f for f in os.listdir(output_dir)
                    if f != ".gitkeep"]
    print()
    print("=" * 52)
    print("   DEMO COMPLETE")
    print(f"   {len(output_files)} output files saved to: outputs/")
    print(f"   Metrics report: outputs/metrics_report.json")
    print(f"   Pipeline grid:  outputs/FULL_PIPELINE_RESULTS.png")
    print("=" * 52)
    print()


if __name__ == "__main__":
    run_demo()
