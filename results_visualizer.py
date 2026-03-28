"""
results_visualizer.py -- Composite Results Image Generator
==========================================================
Loads all saved output images from outputs/ and creates a single
large matplotlib figure (4x4 grid) with technique labels.

Usage:  python results_visualizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# Map output filenames to display labels
IMAGE_MAP = [
    ("synthetic_road.png", "Original Synthetic Road"),
    ("preprocessing_noise_removal.png", "Noise Removal\n(Module 2, S1)"),
    ("preprocessing_clahe.png", "CLAHE Equalization\n(Module 2, S5)"),
    ("preprocessing_morphology.png", "Morphological Ops\n(Module 2, S4)"),
    ("preprocessing_log_transform.png", "Log Transform\n(Module 2, S3)"),
    ("preprocessing_flip_contrast.png", "Flip & Contrast\n(Module 1, S3)"),
    ("edge_sobel.png", "Sobel Edges\n(Module 3, S1)"),
    ("edge_canny.png", "Canny Edges\n(Module 3, S2)"),
    ("edge_comparison.png", "Edge Comparison\n(Module 3)"),
    ("lane_detection_pipeline.png", "Lane Detection\n(Module 4, S1-2)"),
    ("corners_harris.png", "Harris Corners\n(Module 3, S3-4)"),
    ("corners_shi_tomasi.png", "Shi-Tomasi Corners\n(Module 3, S3-4)"),
    ("detected_objects.png", "Object Detection\n(Module 5, S1-2)"),
    ("knn_confusion_matrix.png", "KNN Classifier\n(Module 5, S5)"),
    ("knn_accuracy_vs_k.png", "KNN Accuracy vs k\n(Module 5, S5)"),
    ("preprocessing_comparison.png", "Filter Comparison\n(PSNR Analysis)"),
]


def create_composite_image() -> None:
    """Load output images and create a 4x4 composite grid."""
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(28, 20))
    fig.suptitle(
        "Real-Time Road Scene Analysis -- Full Pipeline Results",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    for idx, ax in enumerate(axes.flat):
        if idx < len(IMAGE_MAP):
            filename, label = IMAGE_MAP[idx]
            img_path = OUTPUT_DIR / filename
            if img_path.is_file():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.set_title(label, fontsize=10, fontweight="bold", pad=6)
                else:
                    ax.text(0.5, 0.5, f"Cannot read\n{filename}", ha="center", va="center", fontsize=9)
                    ax.set_title(label, fontsize=10, pad=6)
            else:
                ax.text(0.5, 0.5, f"Not found:\n{filename}", ha="center", va="center",
                        fontsize=9, color="gray")
                ax.set_title(label, fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = OUTPUT_DIR / "FULL_PIPELINE_RESULTS.png"
    fig.savefig(str(save_path), dpi=80, bbox_inches="tight", facecolor="white")
    print(f"[OK] Composite image saved -> {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    # First generate outputs if they don't exist
    if not (OUTPUT_DIR / "synthetic_road.png").is_file():
        print("[INFO] Running synthetic demo to generate outputs first...")
        from main import generate_synthetic_demo
        generate_synthetic_demo()

    create_composite_image()
