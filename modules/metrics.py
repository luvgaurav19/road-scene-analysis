"""
metrics.py -- Evaluation Metrics Module
=======================================
Provides quantitative analysis functions for measuring
the performance of each CV pipeline stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def edge_density(edge_image: np.ndarray) -> float:
    """Compute the ratio of edge pixels to total pixels.

    Args:
        edge_image: Binary edge map (uint8, 0 or 255) from Canny/Sobel.

    Returns:
        Float in [0, 1] representing edge density.

    Example:
        >>> density = edge_density(canny_edges)
        >>> print(f"Edge density: {density:.4f}")
    """
    if edge_image is None:
        raise ValueError("edge_image is None.")
    total = edge_image.shape[0] * edge_image.shape[1]
    edge_pixels = np.count_nonzero(edge_image)
    return float(edge_pixels / total) if total > 0 else 0.0


def corner_quality_score(
    image: np.ndarray,
    corners: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
) -> float:
    """Compute the average Harris response value at detected corner locations.

    Args:
        image:      Input BGR image.
        corners:    Array of corner coordinates, shape (N, 2) with (x, y).
        block_size: Harris block size.
        ksize:      Sobel aperture size.
        k:          Harris free parameter.

    Returns:
        Average Harris response at the given corner locations.

    Example:
        >>> score = corner_quality_score(image, corner_coords)
    """
    if image is None:
        raise ValueError("image is None.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_f = np.float32(gray)
    harris_resp = cv2.cornerHarris(gray_f, block_size, ksize, k)
    if corners is None or len(corners) == 0:
        return 0.0
    values = []
    h, w = harris_resp.shape
    for pt in corners:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < h and 0 <= x < w:
            values.append(float(harris_resp[y, x]))
    return float(np.mean(values)) if values else 0.0


def lane_confidence(lines: Optional[np.ndarray]) -> float:
    """Compute a consistency score for detected lane line angles.

    Lower standard deviation of line angles indicates more consistent
    (and therefore more confident) lane detection.  Score is mapped
    to [0, 1] where 1 = perfectly consistent.

    Args:
        lines: Output of cv2.HoughLinesP, shape (N, 1, 4) or None.

    Returns:
        Confidence score in [0, 1].

    Example:
        >>> conf = lane_confidence(hough_lines)
    """
    if lines is None or len(lines) == 0:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    std_dev = np.std(angles)
    # Map: 0 std -> 1.0 score, 90 std -> 0.0 score
    confidence = max(0.0, 1.0 - std_dev / 90.0)
    return float(round(confidence, 4))


def classification_report_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate precision, recall, F1 summary for classification results.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: Optional list of class name strings.

    Returns:
        Dict with keys: 'accuracy', 'precision', 'recall', 'f1',
        and per-class breakdown.

    Example:
        >>> report = classification_report_summary(y_test, y_pred, ["road", "non_road"])
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    if class_names:
        report["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

    return report


def processing_time_benchmark(image: np.ndarray) -> "pd.DataFrame":
    """Benchmark each pipeline module and return timings.

    Args:
        image: Input BGR image to benchmark on.

    Returns:
        pandas DataFrame with columns: Operation, Time_ms, FPS_equivalent.
    """
    import pandas as pd
    from modules import preprocessing, edge_detection, lane_detection, corner_detection, object_detector, classifier

    import matplotlib
    matplotlib.use("Agg")

    benchmarks: List[Dict[str, Any]] = []

    def _bench(name: str, func, *args, **kwargs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        fps = 1000.0 / elapsed if elapsed > 0 else 0.0
        benchmarks.append({"Operation": name, "Time_ms": round(elapsed, 2), "FPS_equivalent": round(fps, 1)})

    _bench("Noise Removal", preprocessing.remove_noise, image, save=False)
    _bench("CLAHE Equalization", preprocessing.apply_histogram_equalization, image, save=False)
    _bench("Morphological Ops", preprocessing.apply_morphological_ops, image, save=False)
    _bench("Log Transform", preprocessing.apply_log_transform, image, save=False)
    _bench("Sobel Edge Detection", edge_detection.sobel_edge_detection, image, save=False)
    _bench("Canny Edge Detection", edge_detection.canny_edge_detection, image, save=False)
    _bench("Lane Detection", lane_detection.detect_lanes, image, save=False)
    _bench("Harris Corners", corner_detection.harris_corner_detection, image, save=False)
    _bench("Shi-Tomasi Corners", corner_detection.shi_tomasi_corners, image, save=False)
    _bench("Object Detection", object_detector.detect_objects, image, save=False)

    df = pd.DataFrame(benchmarks)
    print("\n=== Processing Time Benchmark ===")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(df.to_string(index=False))

    return df


def generate_metrics_report(image: np.ndarray, save_path: Optional[str] = None) -> Dict[str, Any]:
    """Run all metrics on an image and save a JSON report.

    Args:
        image:     Input BGR image.
        save_path: Path to save the JSON report (default: outputs/metrics_report.json).

    Returns:
        Dictionary containing all metrics.
    """
    import pandas as pd
    from modules import edge_detection, corner_detection, lane_detection, classifier

    import matplotlib
    matplotlib.use("Agg")

    report: Dict[str, Any] = {}

    # Edge density
    canny = edge_detection.canny_edge_detection(image, save=False)
    sobel = edge_detection.sobel_edge_detection(image, save=False)
    report["edge_density_canny"] = round(edge_density(canny), 6)
    report["edge_density_sobel"] = round(edge_density(sobel), 6)

    # Corner quality
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_st = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners_st is not None:
        corner_pts = corners_st.reshape(-1, 2)
        report["num_corners_shi_tomasi"] = len(corner_pts)
        report["corner_quality_score"] = round(corner_quality_score(image, corner_pts), 8)
    else:
        report["num_corners_shi_tomasi"] = 0
        report["corner_quality_score"] = 0.0

    # Harris corners count
    gray_f = np.float32(gray)
    harris_resp = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    harris_resp = cv2.dilate(harris_resp, None)
    harris_count = int(np.count_nonzero(harris_resp > 0.01 * harris_resp.max()))
    report["num_corners_harris"] = harris_count

    # Lane confidence
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    report["num_lanes_detected"] = int(len(lines)) if lines is not None else 0
    report["lane_confidence"] = lane_confidence(lines)

    # KNN classification
    try:
        model, acc, names = classifier.train_knn_classifier(save=False)
        from sklearn.model_selection import train_test_split
        from modules.classifier import _generate_synthetic_dataset
        X, y, _ = _generate_synthetic_dataset()
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        report["knn_metrics"] = classification_report_summary(y_test, y_pred, names)
    except Exception as e:
        report["knn_metrics"] = {"error": str(e)}

    # Benchmark
    try:
        bench_df = processing_time_benchmark(image)
        report["benchmark"] = bench_df.to_dict(orient="records")
    except Exception as e:
        report["benchmark"] = {"error": str(e)}

    # Save JSON
    if save_path is None:
        save_path = str(OUTPUT_DIR / "metrics_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] Metrics report saved -> {save_path}")

    return report
