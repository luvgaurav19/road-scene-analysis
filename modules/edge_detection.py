"""
edge_detection.py — Edge Detection Module
==========================================
Covers Module 3 of the CV course:
  - Sobel operator (Session 1)
  - Canny edge detection (Session 2)
  - Side-by-side comparison of methods
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Sobel Edge Detection ─────────────────────────────────────
def sobel_edge_detection(image: np.ndarray, save: bool = True) -> np.ndarray:
    """Detect edges using the Sobel operator in both X and Y directions.

    Computes separate X and Y gradient images, then combines them
    using the L2 norm approximation.

    Args:
        image: Input BGR image (np.ndarray).
        save:  Whether to save the result plot to outputs/.

    Returns:
        Combined Sobel edge magnitude image (uint8).

    Example:
        >>> edges = sobel_edge_detection(cv2.imread("road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients in X and Y using a 3×3 Sobel kernel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine using magnitude: sqrt(Gx² + Gy²)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalise to 0-255 for display
    sobel_combined = np.uint8(np.clip(magnitude / magnitude.max() * 255, 0, 255))

    if save:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Original (Gray)", "Sobel X", "Sobel Y", "Sobel Combined"]
        imgs = [
            gray,
            np.uint8(np.abs(sobel_x) / np.abs(sobel_x).max() * 255),
            np.uint8(np.abs(sobel_y) / np.abs(sobel_y).max() * 255),
            sobel_combined,
        ]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        plt.suptitle("Sobel Edge Detection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "edge_sobel.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return sobel_combined


# ── 2. Canny Edge Detection ─────────────────────────────────────
def canny_edge_detection(
    image: np.ndarray,
    low: int = 50,
    high: int = 150,
    save: bool = True,
) -> np.ndarray:
    """Detect edges using the Canny algorithm with tunable thresholds.

    Pre-applies Gaussian blur to reduce noise before edge detection.

    Args:
        image: Input BGR image.
        low:   Lower hysteresis threshold (default 50).
        high:  Upper hysteresis threshold (default 150).
        save:  Whether to save the result plot.

    Returns:
        Binary edge map (uint8, 0 or 255).

    Example:
        >>> edges = canny_edge_detection(cv2.imread("road.jpg"), low=30, high=100)
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur first to suppress noise that could cause false edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, low, high)

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Original (Gray)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(edges, cmap="gray")
        axes[1].set_title(f"Canny Edges (low={low}, high={high})", fontsize=12)
        axes[1].axis("off")

        plt.suptitle("Canny Edge Detection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "edge_canny.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return edges


# ── 3. Compare Edge Methods ─────────────────────────────────────
def compare_edge_methods(image: np.ndarray, save: bool = True) -> None:
    """Show a side-by-side comparison of Sobel, Canny, and Laplacian edge detection.

    This function creates a single figure with four panels:
    original (grayscale), Sobel combined, Canny, and Laplacian.

    Args:
        image: Input BGR image.
        save:  Whether to save the comparison plot.

    Returns:
        None (displays and optionally saves the plot).

    Example:
        >>> compare_edge_methods(cv2.imread("road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # 1. Sobel combined
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))

    # 2. Canny
    canny = cv2.Canny(blurred, 50, 150)

    # 3. Laplacian (bonus comparison)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, img, title in zip(
        axes,
        [gray, sobel, canny, laplacian],
        ["Original", "Sobel", "Canny (50,150)", "Laplacian"],
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle("Edge Detection — Method Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_path = OUTPUT_DIR / "edge_comparison.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")

    plt.show()


# ── Parameter Comparison ────────────────────────────────────────
def compare_parameters(image: np.ndarray, save: bool = True) -> dict[str, float]:
    """Compare Sobel vs Canny vs Laplacian with edge density metric.

    Edge density = (number of edge pixels) / (total pixels).

    Args:
        image: Input BGR image.
        save:  Whether to save the comparison plot.

    Returns:
        Dict mapping method name to its edge density value.

    Example:
        >>> densities = compare_parameters(cv2.imread("road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None -- check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Sobel (thresholded to binary)
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sx**2 + sy**2)
    sobel_bin = (sobel_mag > 50).astype(np.uint8) * 255

    # Canny
    canny = cv2.Canny(blurred, 50, 150)

    # Laplacian (thresholded)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap_bin = (np.abs(lap) > 20).astype(np.uint8) * 255

    total = gray.shape[0] * gray.shape[1]
    densities = {
        "Sobel": round(float(np.count_nonzero(sobel_bin) / total), 6),
        "Canny": round(float(np.count_nonzero(canny) / total), 6),
        "Laplacian": round(float(np.count_nonzero(lap_bin) / total), 6),
    }

    if save:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        items = [
            (gray, "Original"),
            (sobel_bin, f"Sobel\ndensity={densities['Sobel']:.4f}"),
            (canny, f"Canny\ndensity={densities['Canny']:.4f}"),
            (lap_bin, f"Laplacian\ndensity={densities['Laplacian']:.4f}"),
        ]
        for ax, (img, title) in zip(axes, items):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Edge Detection -- Parameter Comparison (Edge Density)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "edge_comparison_params.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return densities
