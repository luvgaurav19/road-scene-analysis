"""
object_detector.py — Object Detection Module
=============================================
Covers Module 5 of the CV course (Session 1–2):
  - MobileNet-SSD via OpenCV DNN module
  - Fallback: contour-based detection when model files are absent
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MobileNet-SSD class labels (PASCAL VOC 21-class) ───────────
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# Assign a random but deterministic colour to each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# ── Paths to MobileNet-SSD model files ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROTOTXT = _PROJECT_ROOT / "MobileNetSSD_deploy.prototxt"
_CAFFEMODEL = _PROJECT_ROOT / "MobileNetSSD_deploy.caffemodel"


def _load_dnn_model() -> Optional[cv2.dnn.Net]:
    """Attempt to load the MobileNet-SSD Caffe model.

    Returns:
        cv2.dnn.Net if both prototxt and caffemodel exist, else None.
    """
    if _PROTOTXT.is_file() and _CAFFEMODEL.is_file():
        net = cv2.dnn.readNetFromCaffe(str(_PROTOTXT), str(_CAFFEMODEL))
        return net
    return None


def _contour_based_detection(
    image: np.ndarray,
    min_area: int = 500,
) -> List[Dict]:
    """Fallback: detect objects via contour analysis.

    Converts to grayscale -> threshold -> find contours.
    Useful when the DNN model is not available.

    Args:
        image:    Input BGR image.
        min_area: Minimum contour area to count as a detection.

    Returns:
        List of dicts with keys 'box', 'label', 'confidence'.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({
            "box": (x, y, x + w, y + h),
            "label": "object",
            "confidence": round(min(area / 5000, 1.0), 2),  # heuristic score
        })

    return detections


def detect_objects(
    image: np.ndarray,
    confidence_threshold: float = 0.5,
    save: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Detect objects in an image using MobileNet-SSD (or contour fallback).

    When the DNN model files are present, uses the MobileNet-SSD network
    via ``cv2.dnn``. Otherwise, falls back to contour-based detection.

    Args:
        image:                Input BGR image.
        confidence_threshold: Minimum confidence to keep a detection (0-1).
        save:                 Whether to save the annotated result.

    Returns:
        Tuple of (annotated_image, detections_list).
        Each detection dict has keys: 'box', 'label', 'confidence'.

    Example:
        >>> annotated, dets = detect_objects(cv2.imread("traffic.jpg"), 0.5)
        >>> for d in dets:
        ...     print(d['label'], d['confidence'])
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    net = _load_dnn_model()
    detections: List[Dict] = []
    result = image.copy()

    if net is not None:
        # ── DNN-based detection ─────────────────────────────────
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        net.setInput(blob)
        output = net.forward()

        for i in range(output.shape[2]):
            confidence = output[0, 0, i, 2]
            if confidence < confidence_threshold:
                continue

            class_id = int(output[0, 0, i, 1])
            label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
            box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            detections.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "confidence": round(float(confidence), 2),
            })

            color = COLORS[class_id].tolist()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(result, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"[DNN] Detected {len(detections)} object(s).")
    else:
        # ── Contour-based fallback ──────────────────────────────
        print("[INFO] MobileNet-SSD model not found — using contour-based fallback.")
        detections = _contour_based_detection(image)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = f"{det['label']}: {det['confidence']}"
            cv2.putText(result, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        print(f"[Contour] Detected {len(detections)} region(s).")

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Detections ({len(detections)})", fontsize=12)
        axes[1].axis("off")

        plt.suptitle("Object Detection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "detected_objects.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return result, detections
