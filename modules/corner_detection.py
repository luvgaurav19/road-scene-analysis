"""
corner_detection.py — Corner Detection Module
==============================================
Covers Module 3 of the CV course:
  - Harris Corner Detector (Session 3–4)
  - Shi-Tomasi "Good Features to Track"
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def harris_corner_detection(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
    save: bool = True,
) -> np.ndarray:
    """Detect corners using the Harris corner detector.

    The Harris response R is computed for each pixel. Pixels with
    R > threshold_ratio × max(R) are marked as corners.

    Args:
        image:           Input BGR image.
        block_size:      Neighbourhood size for the structure tensor (default 2).
        ksize:           Aperture size for the Sobel derivative (default 3).
        k:               Harris detector free parameter (default 0.04).
        threshold_ratio: Fraction of max response used as corner threshold.
        save:            Whether to save the result plot.

    Returns:
        BGR image with red circles drawn at detected corner locations.

    Example:
        >>> result = harris_corner_detection(cv2.imread("intersection.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # Compute Harris response matrix
    harris_response = cv2.cornerHarris(gray_float, block_size, ksize, k)

    # Dilate to make corners more visible
    harris_response = cv2.dilate(harris_response, None)

    result = image.copy()

    # Threshold: mark pixels whose response exceeds threshold_ratio × max
    threshold = threshold_ratio * harris_response.max()
    corner_coords = np.argwhere(harris_response > threshold)

    # Draw small red circles at corner locations
    for y, x in corner_coords:
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)

    if save:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(harris_response, cmap="jet")
        axes[1].set_title("Harris Response Map", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Corners Detected ({len(corner_coords)})", fontsize=12)
        axes[2].axis("off")

        plt.suptitle("Harris Corner Detection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "corners_harris.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return result


def shi_tomasi_corners(
    image: np.ndarray,
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: int = 10,
    save: bool = True,
) -> np.ndarray:
    """Detect corners using the Shi-Tomasi method (Good Features to Track).

    Uses the minimum eigenvalue of the structure tensor instead of
    Harris's combined score — generally more stable for tracking.

    Args:
        image:         Input BGR image.
        max_corners:   Maximum number of corners to return.
        quality_level: Minimum accepted quality (fraction of best corner quality).
        min_distance:  Minimum Euclidean distance between returned corners.
        save:          Whether to save the result plot.

    Returns:
        BGR image with green circles at detected corners.

    Example:
        >>> result = shi_tomasi_corners(cv2.imread("road_sign.jpg"), max_corners=50)
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )

    result = image.copy()
    num_corners = 0

    if corners is not None:
        corners = np.intp(corners)  # convert to integer coordinates
        num_corners = len(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

    if save:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Shi-Tomasi Corners ({num_corners})", fontsize=12)
        axes[1].axis("off")

        plt.suptitle("Shi-Tomasi Corner Detection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "corners_shi_tomasi.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return result


# ── Parameter Comparison ────────────────────────────────────────
def compare_parameters(image: np.ndarray, save: bool = True) -> dict[str, dict]:
    """Compare Harris vs Shi-Tomasi vs FAST detector.

    Reports the number of keypoints detected and a quality score
    for each method.

    Args:
        image: Input BGR image.
        save:  Whether to save the comparison plot.

    Returns:
        Dict mapping method name to {'count': int, 'quality': float}.

    Example:
        >>> results = compare_parameters(cv2.imread("road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None -- check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)
    result_imgs = []
    results = {}

    # 1. Harris
    harris_resp = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    harris_resp_d = cv2.dilate(harris_resp, None)
    harris_mask = harris_resp_d > 0.01 * harris_resp_d.max()
    harris_count = int(np.count_nonzero(harris_mask))
    harris_quality = float(np.mean(harris_resp_d[harris_mask])) if harris_count > 0 else 0.0
    img_h = image.copy()
    img_h[harris_mask] = [0, 0, 255]
    result_imgs.append((img_h, f"Harris\n{harris_count} pts, q={harris_quality:.6f}"))
    results["Harris"] = {"count": harris_count, "quality": round(harris_quality, 8)}

    # 2. Shi-Tomasi
    corners_st = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
    img_st = image.copy()
    st_count = 0
    if corners_st is not None:
        st_count = len(corners_st)
        for c in corners_st:
            x, y = c.ravel()
            cv2.circle(img_st, (int(x), int(y)), 4, (0, 255, 0), -1)
    result_imgs.append((img_st, f"Shi-Tomasi\n{st_count} pts"))
    results["Shi-Tomasi"] = {"count": st_count, "quality": 0.0}

    # 3. FAST
    fast = cv2.FastFeatureDetector_create()
    kp_fast = fast.detect(gray, None)
    fast_count = len(kp_fast)
    img_fast = image.copy()
    cv2.drawKeypoints(image, kp_fast, img_fast, color=(255, 0, 0))
    result_imgs.append((img_fast, f"FAST\n{fast_count} pts"))
    results["FAST"] = {"count": fast_count, "quality": 0.0}

    if save:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")
        for ax, (img, title) in zip(axes[1:], result_imgs):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Corner Detector Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "corners_comparison.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return results
