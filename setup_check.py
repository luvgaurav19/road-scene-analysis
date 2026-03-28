"""
setup_check.py -- Environment Verification Script
==================================================
Run this ONCE on a fresh machine to verify everything is ready.
Uses only Python standard library for the check logic.

Usage:  python setup_check.py
"""

import sys
import os
import importlib

# ── Configuration ───────────────────────────────────────────────
MIN_PYTHON = (3, 10)

REQUIRED_LIBS = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("tabulate", "tabulate"),
    ("imutils", "imutils"),
    ("PIL", "Pillow"),
]

REQUIRED_DIRS = [
    "modules",
    "outputs",
    "assets",
    "assets/sample_images",
    "assets/sample_video",
    "notebooks",
]

REQUIRED_FILES = [
    "main.py",
    "app.py",
    "demo.py",
    "requirements.txt",
    "modules/__init__.py",
    "modules/preprocessing.py",
    "modules/edge_detection.py",
    "modules/lane_detection.py",
    "modules/corner_detection.py",
    "modules/object_detector.py",
    "modules/object_tracker.py",
    "modules/classifier.py",
    "modules/metrics.py",
]


def main():
    print("=" * 50)
    print("  Road Scene Analysis -- Environment Check")
    print("=" * 50)
    print()

    issues = 0

    # ── 1. Python version ───────────────────────────────────
    print(f"Python version: {sys.version}")
    v = sys.version_info
    if (v.major, v.minor) >= MIN_PYTHON:
        print(f"  [PASS] Python {v.major}.{v.minor} >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    else:
        print(f"  [FAIL] Python {v.major}.{v.minor} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
        issues += 1
    print()

    # ── 2. Required libraries ──────────────────────────────
    print("Checking required libraries:")
    for import_name, pip_name in REQUIRED_LIBS:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {pip_name:20s} ({ver})")
        except ImportError:
            print(f"  [FAIL] {pip_name:20s} -- not installed")
            issues += 1
    print()

    # ── 3. Required directories ────────────────────────────
    print("Checking project directories:")
    project_root = os.path.dirname(os.path.abspath(__file__))
    for d in REQUIRED_DIRS:
        full = os.path.join(project_root, d)
        if os.path.isdir(full):
            print(f"  [PASS] {d}/")
        else:
            os.makedirs(full, exist_ok=True)
            print(f"  [CREATED] {d}/ (was missing, created now)")
    print()

    # ── 4. Required files ──────────────────────────────────
    print("Checking key files:")
    for f in REQUIRED_FILES:
        full = os.path.join(project_root, f)
        if os.path.isfile(full):
            print(f"  [PASS] {f}")
        else:
            print(f"  [WARN] {f} -- not found")
            issues += 1
    print()

    # ── 5. Summary ─────────────────────────────────────────
    print("=" * 50)
    if issues == 0:
        print("  Environment ready. Run: python demo.py")
    else:
        print(f"  {issues} issue(s) found.")
        print("  Run: pip install -r requirements.txt")
    print("=" * 50)

    return issues


if __name__ == "__main__":
    sys.exit(main())
