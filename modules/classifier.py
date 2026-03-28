"""
classifier.py — KNN Image Classifier Module
============================================
Covers Module 5 of the CV course (Session 5):
  - HOG feature extraction
  - KNN training and prediction via scikit-learn
  - Confusion matrix and accuracy reporting
  - Synthetic data generation for out-of-the-box demo
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── HOG feature extraction ──────────────────────────────────────
def _extract_hog_features(image: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Extract HOG (Histogram of Oriented Gradients) features from an image.

    Resizes to a fixed size, converts to grayscale if needed, then
    computes a HOG descriptor using OpenCV's built-in HOGDescriptor.

    Args:
        image: Input image (BGR or grayscale).
        size:  Target (width, height) for resizing before HOG.

    Returns:
        1-D feature vector (np.ndarray of float32).
    """
    resized = cv2.resize(image, size)
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # HOG parameters tuned for small patches
    hog = cv2.HOGDescriptor(
        _winSize=size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    features = hog.compute(resized)
    return features.flatten()


# ── Synthetic dataset generation ────────────────────────────────
def _generate_synthetic_dataset(
    n_per_class: int = 100,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a simple synthetic 2-class dataset (road vs non-road).

    - **road** images: horizontal grey stripe with dashed white lines.
    - **non-road** images: random coloured blobs (sky/trees proxy).

    Args:
        n_per_class: Number of samples per class.

    Returns:
        (features_array, labels_array, class_names)
    """
    features_list = []
    labels_list = []
    class_names = ["road", "non_road"]

    rng = np.random.RandomState(42)

    for i in range(n_per_class):
        # ── Class 0: road-like image ────────────────────────────
        img_road = np.full((64, 64, 3), (100, 100, 100), dtype=np.uint8)
        # White dashed centre line
        for x_start in range(0, 64, 12):
            cv2.line(img_road, (x_start, 32), (x_start + 6, 32), (255, 255, 255), 2)
        # Add slight noise
        noise = rng.randint(-10, 10, img_road.shape, dtype=np.int16)
        img_road = np.clip(img_road.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        features_list.append(_extract_hog_features(img_road))
        labels_list.append(0)

        # ── Class 1: non-road image (nature-like) ──────────────
        img_non = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Draw random green blobs (like foliage)
        for _ in range(5):
            cx, cy = rng.randint(10, 54, 2)
            r = rng.randint(5, 15)
            cv2.circle(img_non, (int(cx), int(cy)), int(r), (0, int(rng.randint(100, 255)), 0), -1)

        features_list.append(_extract_hog_features(img_non))
        labels_list.append(1)

    return np.array(features_list), np.array(labels_list), class_names


# ── Public API ──────────────────────────────────────────────────

def train_knn_classifier(
    data_dir: Optional[str] = None,
    n_neighbors: int = 5,
    save: bool = True,
) -> Tuple[KNeighborsClassifier, float, list[str]]:
    """Train a KNN classifier on a small image dataset.

    If ``data_dir`` is provided, images are loaded from subfolders
    (each subfolder name = class label). Otherwise a synthetic dataset
    is generated automatically so the project works without external data.

    Args:
        data_dir:    Path to the dataset root (optional).
        n_neighbors: Number of neighbours for KNN (default 5).
        save:        Whether to save the confusion matrix plot.

    Returns:
        (trained_model, accuracy, class_names)

    Example:
        >>> model, acc, names = train_knn_classifier()
        >>> print(f"Accuracy: {acc:.2%}")
    """
    if data_dir is not None and Path(data_dir).is_dir():
        # Load real dataset from disk
        features_list = []
        labels_list = []
        class_names = sorted(
            d for d in os.listdir(data_dir) if (Path(data_dir) / d).is_dir()
        )
        label_map = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = Path(data_dir) / class_name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                features_list.append(_extract_hog_features(img))
                labels_list.append(label_map[class_name])

        X = np.array(features_list)
        y = np.array(labels_list)
    else:
        print("[INFO] No dataset directory provided — generating synthetic data.")
        X, y, class_names = _generate_synthetic_dataset()

    # Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[KNN] Accuracy: {acc:.2%}  (k={n_neighbors}, test size=25%)")

    if save:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(f"KNN Confusion Matrix (k={n_neighbors}, acc={acc:.2%})", fontsize=12)
        plt.tight_layout()
        save_path = OUTPUT_DIR / "knn_confusion_matrix.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return knn, acc, class_names


def predict(
    image: np.ndarray,
    model: KNeighborsClassifier,
    class_names: list[str],
) -> str:
    """Classify a single image using a trained KNN model.

    Args:
        image:       Input image (BGR).
        model:       Trained KNeighborsClassifier.
        class_names: List mapping label indices to human-readable names.

    Returns:
        Predicted class name (str).

    Example:
        >>> model, _, names = train_knn_classifier()
        >>> label = predict(cv2.imread("patch.jpg"), model, names)
        >>> print(label)
    """
    if image is None:
        raise FileNotFoundError("Input image is None — check the file path.")

    features = _extract_hog_features(image).reshape(1, -1)
    pred_idx = model.predict(features)[0]
    return class_names[int(pred_idx)]


# ── Parameter Comparison ────────────────────────────────────────
def compare_parameters(save: bool = True) -> dict[int, float]:
    """Compare KNN with k=1,3,5,7 and plot accuracy vs k.

    Uses synthetic data so it works without an external dataset.

    Args:
        save: Whether to save the accuracy-vs-k plot.

    Returns:
        Dict mapping k value to accuracy.

    Example:
        >>> results = compare_parameters()
        >>> # {1: 0.96, 3: 0.98, 5: 0.98, 7: 0.96}
    """
    X, y, class_names = _generate_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        accuracies[k] = round(acc, 4)

    print("[KNN Comparison]")
    for k, acc in accuracies.items():
        print(f"  k={k:2d}  -> accuracy={acc:.2%}")

    if save:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(accuracies.keys()), list(accuracies.values()),
                "o-", color="#2196F3", linewidth=2, markersize=8)
        ax.set_xlabel("k (Number of Neighbours)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("KNN: Accuracy vs k", fontsize=14, fontweight="bold")
        ax.set_xticks(k_values)
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = OUTPUT_DIR / "knn_accuracy_vs_k.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[OK] Saved -> {save_path}")
        plt.show()

    return accuracies
