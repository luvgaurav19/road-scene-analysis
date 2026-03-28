"""
preprocessing.py — Image Preprocessing Module
==============================================
Covers Module 1 & 2 of the CV course:
  - Noise removal (Gaussian + Median blur)
  - Histogram equalization (CLAHE)
  - Morphological operations (erosion, dilation, opening, closing)
  - Log transformation
  - Flip & contrast adjustment
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _validate_image(image, module_name: str = "Preprocessing") -> None:
    """Raise ValueError if *image* is not a valid NumPy image array."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError(
            f"[{module_name}] Expected a valid NumPy image array, got {type(image)}"
        )
    if len(image.shape) < 2:
        raise ValueError(
            f"[{module_name}] Image must be at least 2D, got shape {image.shape}"
        )


# ── Helper: save a before/after figure ──────────────────────────
def _show_before_after(
    before: np.ndarray,
    after: np.ndarray,
    title_before: str = "Before",
    title_after: str = "After",
    save_name: Optional[str] = None,
) -> None:
    """Display and optionally save a before/after comparison plot.

    Args:
        before: Original image (BGR or grayscale).
        after:  Processed image (BGR or grayscale).
        title_before: Title for the left subplot.
        title_after:  Title for the right subplot.
        save_name: If provided, the figure is saved to outputs/<save_name>.png.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convert BGR -> RGB for display if colour image
    def _to_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    cmap_before = "gray" if before.ndim == 2 else None
    cmap_after = "gray" if after.ndim == 2 else None

    axes[0].imshow(_to_rgb(before), cmap=cmap_before)
    axes[0].set_title(title_before, fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(_to_rgb(after), cmap=cmap_after)
    axes[1].set_title(title_after, fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()

    if save_name:
        save_path = OUTPUT_DIR / f"{save_name}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")

    plt.show()


# ── 1. Noise Removal ────────────────────────────────────────────
def remove_noise(image: np.ndarray, save: bool = True) -> np.ndarray:
    """Remove noise using Gaussian blur followed by median blur.

    Combines a 5×5 Gaussian kernel (σ determined automatically) with
    a 5×5 median filter for salt-and-pepper noise suppression.

    Args:
        image: Input BGR image (np.ndarray).
        save:  Whether to save the before/after plot to outputs/.

    Returns:
        Denoised image (np.ndarray, same shape/dtype as input).

    Example:
        >>> denoised = remove_noise(cv2.imread("road.jpg"))
    """
    _validate_image(image, "Preprocessing")

    # Step 1: Gaussian blur — reduces high-frequency noise
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 2: Median blur — effective for salt-and-pepper noise
    denoised = cv2.medianBlur(gaussian, 5)

    if save:
        _show_before_after(image, denoised, "Original (Noisy)", "After Noise Removal",
                           save_name="preprocessing_noise_removal")
    return denoised


# ── 2. Histogram Equalization (CLAHE) ───────────────────────────
def apply_histogram_equalization(image: np.ndarray, save: bool = True) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast-Limited Adaptive Histogram Equalization).

    Converts to LAB colour space, applies CLAHE on the L channel,
    and converts back to BGR. This avoids colour distortion.

    Args:
        image: Input BGR image.
        save:  Whether to save the before/after plot.

    Returns:
        Contrast-enhanced BGR image.

    Example:
        >>> enhanced = apply_histogram_equalization(cv2.imread("dark_road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    # Convert to LAB so we only equalise the *lightness* channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # CLAHE limits over-amplification of noise in homogeneous regions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    if save:
        _show_before_after(image, result, "Original", "CLAHE Equalized",
                           save_name="preprocessing_clahe")
    return result


# ── 3. Morphological Operations ─────────────────────────────────
def apply_morphological_ops(image: np.ndarray, save: bool = True) -> dict[str, np.ndarray]:
    """Demonstrate erosion, dilation, opening, and closing on a binary image.

    A 5×5 rectangular kernel is used for all operations.

    Args:
        image: Input BGR image.
        save:  Whether to save the result grid.

    Returns:
        Dictionary with keys 'erosion', 'dilation', 'opening', 'closing',
        each mapping to the corresponding result image.

    Example:
        >>> results = apply_morphological_ops(cv2.imread("road.jpg"))
        >>> eroded = results['erosion']
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    # Convert to grayscale -> threshold to get binary image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # erosion -> dilation
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # dilation -> erosion

    results = {
        "erosion": erosion,
        "dilation": dilation,
        "opening": opening,
        "closing": closing,
    }

    if save:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        titles = ["Binary", "Erosion", "Dilation", "Opening", "Closing"]
        images = [binary, erosion, dilation, opening, closing]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        plt.suptitle("Morphological Operations", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "preprocessing_morphology.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return results


# ── 4. Log Transformation ───────────────────────────────────────
def apply_log_transform(image: np.ndarray, save: bool = True) -> np.ndarray:
    """Apply log transformation to expand dark pixel intensities.

    Formula: s = c · log(1 + r), where c is a scaling constant chosen
    so that the maximum output value is 255.

    Args:
        image: Input BGR image.
        save:  Whether to save the before/after plot.

    Returns:
        Log-transformed image (uint8).

    Example:
        >>> log_img = apply_log_transform(cv2.imread("dark_road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    # Work in float to avoid overflow
    img_float = image.astype(np.float64) + 1.0  # +1 to avoid log(0)
    c = 255.0 / np.log(1.0 + np.max(img_float))
    log_image = (c * np.log(img_float)).astype(np.uint8)

    if save:
        _show_before_after(image, log_image, "Original", "Log Transformed",
                           save_name="preprocessing_log_transform")
    return log_image


# ── 5. Flip & Contrast ──────────────────────────────────────────
def flip_and_contrast(image: np.ndarray, save: bool = True) -> np.ndarray:
    """Demonstrate horizontal flip and contrast reduction.

    Contrast is reduced by scaling pixel values closer to the mean
    using  result = α · image + (1 - α) · mean, with α = 0.5.

    Args:
        image: Input BGR image.
        save:  Whether to save the result grid.

    Returns:
        Contrast-reduced, horizontally-flipped image.

    Example:
        >>> result = flip_and_contrast(cv2.imread("road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    # Horizontal flip (simulates rear-view mirror perspective)
    flipped = cv2.flip(image, 1)

    # Reduce contrast: blend towards the mean intensity
    alpha = 0.5
    mean_val = np.mean(flipped).astype(np.float64)
    low_contrast = cv2.convertScaleAbs(
        flipped, alpha=alpha, beta=int(mean_val * (1 - alpha))
    )

    if save:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, img, title in zip(
            axes,
            [image, flipped, low_contrast],
            ["Original", "Horizontally Flipped", "Flipped + Low Contrast"],
        ):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        plt.suptitle("Flip & Contrast Demo", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "preprocessing_flip_contrast.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return low_contrast


# ── Parameter Comparison ────────────────────────────────────────
def compare_parameters(image: np.ndarray, save: bool = True) -> dict[str, float]:
    """Compare Gaussian vs Median vs Bilateral filter with PSNR values.

    Creates a 3-way comparison showing each filter result and its
    Peak Signal-to-Noise Ratio relative to the original image.

    Args:
        image: Input BGR image.
        save:  Whether to save the comparison plot.

    Returns:
        Dict mapping filter name to its PSNR value.

    Example:
        >>> psnr_results = compare_parameters(cv2.imread("road.jpg"))
    """
    if image is None:
        raise FileNotFoundError("Input image is None -- check the file path.")

    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    median = cv2.medianBlur(image, 5)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)

    # PSNR: higher = closer to original
    psnr_gauss = float(cv2.PSNR(image, gaussian))
    psnr_median = float(cv2.PSNR(image, median))
    psnr_bilat = float(cv2.PSNR(image, bilateral))

    results = {
        "Gaussian (5x5)": round(psnr_gauss, 2),
        "Median (5x5)": round(psnr_median, 2),
        "Bilateral (d=9)": round(psnr_bilat, 2),
    }

    if save:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        items = [
            (image, "Original"),
            (gaussian, f"Gaussian\nPSNR={psnr_gauss:.2f} dB"),
            (median, f"Median\nPSNR={psnr_median:.2f} dB"),
            (bilateral, f"Bilateral\nPSNR={psnr_bilat:.2f} dB"),
        ]
        for ax, (img, title) in zip(axes, items):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Filter Comparison (with PSNR)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = OUTPUT_DIR / "preprocessing_comparison.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return results
