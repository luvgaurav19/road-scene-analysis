"""
app.py -- Streamlit Dashboard for Road Scene Analysis
=====================================================
Interactive web UI with sidebar navigation, image/video upload,
real-time parameter controls, and metrics display.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Ensure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import preprocessing, edge_detection, lane_detection
from modules import corner_detection, object_detector, classifier
from main import generate_synthetic_road_image

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Scene Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ─────────────────────────────────────────────────────
def _load_uploaded_image(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile to an OpenCV BGR image."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for Streamlit display."""
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _img_download_button(img: np.ndarray, filename: str, label: str = "Download Image"):
    """Provide a download button for a CV image."""
    success, buf = cv2.imencode(".png", img)
    if success:
        st.download_button(label, buf.tobytes(), file_name=filename, mime="image/png")


# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.title("🚗 Road Scene Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🖼️ Preprocessing",
        "📐 Edge Detection",
        "🛣️ Lane Detection",
        "📌 Corner Detection",
        "🔍 Object Detection",
        "🎯 KNN Classifier",
        "📊 Comparison Analysis",
        "🚀 Full Pipeline",
        "📚 Course Reference",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Image Source:**")
source = st.sidebar.radio("Choose input", ["Upload Image", "Synthetic Demo"], index=1)

uploaded_img = None
if source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        try:
            uploaded_img = _load_uploaded_image(uploaded_file)
            if uploaded_img is None:
                st.sidebar.error("Could not decode the uploaded file. Please upload a valid image (jpg, png, bmp).")
        except Exception:
            st.sidebar.error("Please upload a valid image (jpg, png) or video (mp4, avi).")

# Get working image (persist in session state)
if uploaded_img is not None:
    st.session_state["work_img"] = uploaded_img

if "work_img" not in st.session_state or source == "Synthetic Demo":
    st.session_state["work_img"] = generate_synthetic_road_image(640, 480)

work_img = st.session_state["work_img"]

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Road Scene Analysis System**  
VIT Bhopal -- BYOP Project  
Computer Vision Course  
B.Tech CSE | 2025-26
""")


# ── HOME ────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🚗 Real-Time Road Scene Analysis System")
    st.markdown("""
    > **A modular OpenCV pipeline for road/traffic image analysis**
    
    **Features:**
    - ✅ 5 Modules Covered
    - ✅ 13+ CV Techniques  
    - ✅ Real-time Processing
    - ✅ Interactive Dashboard
    
    Use the **sidebar** to navigate between modules. Upload your own image
    or use the built-in synthetic road scene.
    """)
    st.image(_to_rgb(work_img), caption="Current Working Image", use_container_width=True)


# ── PREPROCESSING ───────────────────────────────────────────────
elif page == "🖼️ Preprocessing":
    st.title("🖼️ Image Preprocessing")
    st.markdown("Toggle each preprocessing step and see before/after results.")

    col_opts, _ = st.columns([1, 2])
    with col_opts:
        do_noise = st.toggle("Noise Removal", value=True)
        do_clahe = st.toggle("CLAHE Histogram Equalization", value=True)
        do_morph = st.toggle("Morphological Operations", value=True)
        do_log = st.toggle("Log Transform", value=True)
        do_flip = st.toggle("Flip & Contrast", value=True)

    processed = work_img.copy()

    if do_noise:
        t0 = time.perf_counter()
        processed = preprocessing.remove_noise(processed, save=False)
        ms = (time.perf_counter() - t0) * 1000
        st.subheader("Noise Removal")
        c1, c2 = st.columns(2)
        c1.image(_to_rgb(work_img), caption="Before", use_container_width=True)
        c2.image(_to_rgb(processed), caption=f"After ({ms:.1f}ms)", use_container_width=True)
        _img_download_button(processed, "denoised.png")

    if do_clahe:
        t0 = time.perf_counter()
        clahe_result = preprocessing.apply_histogram_equalization(processed, save=False)
        ms = (time.perf_counter() - t0) * 1000
        st.subheader("CLAHE Equalization")
        c1, c2 = st.columns(2)
        c1.image(_to_rgb(processed), caption="Before", use_container_width=True)
        c2.image(_to_rgb(clahe_result), caption=f"After ({ms:.1f}ms)", use_container_width=True)
        processed = clahe_result
        _img_download_button(processed, "clahe.png")

    if do_morph:
        st.subheader("Morphological Operations")
        morph = preprocessing.apply_morphological_ops(processed, save=False)
        cols = st.columns(4)
        for col, (name, img) in zip(cols, morph.items()):
            col.image(img, caption=name.title(), use_container_width=True, clamp=True)

    if do_log:
        t0 = time.perf_counter()
        log_result = preprocessing.apply_log_transform(processed, save=False)
        ms = (time.perf_counter() - t0) * 1000
        st.subheader("Log Transform")
        c1, c2 = st.columns(2)
        c1.image(_to_rgb(processed), caption="Before", use_container_width=True)
        c2.image(_to_rgb(log_result), caption=f"After ({ms:.1f}ms)", use_container_width=True)
        _img_download_button(log_result, "log_transform.png")

    if do_flip:
        t0 = time.perf_counter()
        flip_result = preprocessing.flip_and_contrast(processed, save=False)
        ms = (time.perf_counter() - t0) * 1000
        st.subheader("Flip & Contrast")
        c1, c2 = st.columns(2)
        c1.image(_to_rgb(processed), caption="Before", use_container_width=True)
        c2.image(_to_rgb(flip_result), caption=f"After ({ms:.1f}ms)", use_container_width=True)
        _img_download_button(flip_result, "flip_contrast.png")


# ── EDGE DETECTION ──────────────────────────────────────────────
elif page == "📐 Edge Detection":
    st.title("📐 Edge Detection")

    st.sidebar.markdown("### Canny Parameters")
    canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
    canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)

    # Sobel
    st.subheader("Sobel Edge Detection")
    t0 = time.perf_counter()
    sobel = edge_detection.sobel_edge_detection(work_img, save=False)
    ms = (time.perf_counter() - t0) * 1000
    edge_count_sobel = int(np.count_nonzero(sobel))
    c1, c2 = st.columns(2)
    c1.image(_to_rgb(work_img), caption="Original", use_container_width=True)
    c2.image(sobel, caption=f"Sobel ({ms:.1f}ms)", use_container_width=True, clamp=True)
    st.metric("Edge Pixels (Sobel)", f"{edge_count_sobel:,}")
    _img_download_button(sobel, "sobel_edges.png")

    # Canny
    st.subheader("Canny Edge Detection")
    t0 = time.perf_counter()
    canny = edge_detection.canny_edge_detection(work_img, low=canny_low, high=canny_high, save=False)
    ms = (time.perf_counter() - t0) * 1000
    edge_count_canny = int(np.count_nonzero(canny))
    c1, c2 = st.columns(2)
    c1.image(_to_rgb(work_img), caption="Original", use_container_width=True)
    c2.image(canny, caption=f"Canny ({ms:.1f}ms)", use_container_width=True, clamp=True)
    st.metric("Edge Pixels (Canny)", f"{edge_count_canny:,}")
    _img_download_button(canny, "canny_edges.png")


# ── LANE DETECTION ──────────────────────────────────────────────
elif page == "🛣️ Lane Detection":
    st.title("🛣️ Lane Detection")

    st.sidebar.markdown("### Hough Parameters")
    hough_threshold = st.sidebar.slider("Hough Threshold", 10, 200, 50)
    min_line_len = st.sidebar.slider("Min Line Length", 10, 200, 50)

    t0 = time.perf_counter()
    lane_result = lane_detection.detect_lanes(work_img, save=False)
    ms = (time.perf_counter() - t0) * 1000

    # Count detected lines
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, minLineLength=min_line_len, maxLineGap=150)
    num_lanes = len(lines) if lines is not None else 0

    c1, c2 = st.columns(2)
    c1.image(_to_rgb(work_img), caption="Original", use_container_width=True)
    c2.image(_to_rgb(lane_result), caption=f"Lane Detection ({ms:.1f}ms)", use_container_width=True)

    st.metric("Lanes Detected", num_lanes)
    _img_download_button(lane_result, "lane_detection.png")


# ── CORNER DETECTION ────────────────────────────────────────────
elif page == "📌 Corner Detection":
    st.title("📌 Corner Detection")

    st.sidebar.markdown("### Harris Parameters")
    harris_k = st.sidebar.slider("Harris k parameter", 0.01, 0.10, 0.04, step=0.01)

    # Harris
    st.subheader("Harris Corner Detection")
    t0 = time.perf_counter()
    harris = corner_detection.harris_corner_detection(work_img, k=harris_k, save=False)
    ms_harris = (time.perf_counter() - t0) * 1000

    gray_f = np.float32(cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY))
    resp = cv2.cornerHarris(gray_f, 2, 3, harris_k)
    harris_count = int(np.count_nonzero(cv2.dilate(resp, None) > 0.01 * resp.max()))

    c1, c2 = st.columns(2)
    c1.image(_to_rgb(work_img), caption="Original", use_container_width=True)
    c2.image(_to_rgb(harris), caption=f"Harris ({ms_harris:.1f}ms)", use_container_width=True)
    st.metric("Harris Corners", harris_count)
    _img_download_button(harris, "harris_corners.png")

    # Shi-Tomasi
    st.subheader("Shi-Tomasi Corners")
    t0 = time.perf_counter()
    shi = corner_detection.shi_tomasi_corners(work_img, save=False)
    ms_shi = (time.perf_counter() - t0) * 1000

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    shi_count = len(corners) if corners is not None else 0

    c1, c2 = st.columns(2)
    c1.image(_to_rgb(work_img), caption="Original", use_container_width=True)
    c2.image(_to_rgb(shi), caption=f"Shi-Tomasi ({ms_shi:.1f}ms)", use_container_width=True)
    st.metric("Shi-Tomasi Corners", shi_count)
    _img_download_button(shi, "shi_tomasi_corners.png")


# ── OBJECT DETECTION ────────────────────────────────────────────
elif page == "🔍 Object Detection":
    st.title("🔍 Object Detection")

    t0 = time.perf_counter()
    det_img, detections = object_detector.detect_objects(work_img, confidence_threshold=0.5, save=False)
    ms = (time.perf_counter() - t0) * 1000

    c1, c2 = st.columns(2)
    c1.image(_to_rgb(work_img), caption="Original", use_container_width=True)
    c2.image(_to_rgb(det_img), caption=f"Detections ({ms:.1f}ms)", use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Objects Detected", len(detections))
    col2.metric("Processing Time", f"{ms:.1f} ms")

    if detections:
        st.subheader("Detection Details")
        for i, d in enumerate(detections, 1):
            st.write(f"**{i}.** {d['label']} (confidence: {d['confidence']}) -- box: {d['box']}")

    _img_download_button(det_img, "object_detection.png")


# ── KNN CLASSIFIER ──────────────────────────────────────────────
elif page == "🎯 KNN Classifier":
    st.title("🎯 KNN Image Classifier")

    st.sidebar.markdown("### KNN Parameters")
    knn_k = st.sidebar.slider("k (neighbors)", 1, 15, 5, step=2)

    with st.spinner("Training KNN classifier..."):
        model, acc, names = classifier.train_knn_classifier(n_neighbors=knn_k, save=False)

    st.metric("Accuracy", f"{acc:.2%}")
    label = classifier.predict(work_img, model, names)
    st.success(f"Predicted class for current image: **{label}**")

    st.subheader("Accuracy vs k Comparison")
    with st.spinner("Running k comparison..."):
        k_results = classifier.compare_parameters(save=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_results.keys()), list(k_results.values()), "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN: Accuracy vs k")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ── COMPARISON ANALYSIS ─────────────────────────────────────────
elif page == "📊 Comparison Analysis":
    st.title("📊 Parameter Comparison Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Filters (PSNR)", "Edge Density", "Corner Detectors", "Hough Transform"])

    with tab1:
        st.subheader("Gaussian vs Median vs Bilateral (PSNR)")
        results = preprocessing.compare_parameters(work_img, save=False)
        cols = st.columns(len(results))
        for col, (name, val) in zip(cols, results.items()):
            col.metric(name, f"{val:.2f} dB")

    with tab2:
        st.subheader("Sobel vs Canny vs Laplacian (Edge Density)")
        results = edge_detection.compare_parameters(work_img, save=False)
        cols = st.columns(len(results))
        for col, (name, val) in zip(cols, results.items()):
            col.metric(name, f"{val:.4f}")

    with tab3:
        st.subheader("Harris vs Shi-Tomasi vs FAST")
        results = corner_detection.compare_parameters(work_img, save=False)
        cols = st.columns(len(results))
        for col, (name, data) in zip(cols, results.items()):
            col.metric(name, f"{data['count']} keypoints")

    with tab4:
        st.subheader("Standard vs Probabilistic Hough")
        results = lane_detection.compare_parameters(work_img, save=False)
        cols = st.columns(len(results))
        for col, (name, data) in zip(cols, results.items()):
            col.metric(f"{name} Lines", data["line_count"])
            col.caption(f"{data['time_ms']:.2f} ms")


# ── FULL PIPELINE ───────────────────────────────────────────────
elif page == "🚀 Full Pipeline":
    st.title("🚀 Run Full Pipeline")
    st.markdown("Chains every module sequentially on the current image.")

    if st.button("Run Full Pipeline", type="primary"):
        progress = st.progress(0)
        metrics = {}

        # 1. Preprocessing
        st.subheader("1. Preprocessing")
        t0 = time.perf_counter()
        denoised = preprocessing.remove_noise(work_img, save=False)
        clahe = preprocessing.apply_histogram_equalization(denoised, save=False)
        ms = (time.perf_counter() - t0) * 1000
        st.image(_to_rgb(clahe), caption=f"Preprocessed ({ms:.1f}ms)", use_container_width=True)
        progress.progress(15)

        # 2. Edges
        st.subheader("2. Edge Detection")
        t0 = time.perf_counter()
        canny = edge_detection.canny_edge_detection(clahe, save=False)
        ms = (time.perf_counter() - t0) * 1000
        edge_count = int(np.count_nonzero(canny))
        metrics["Edge Pixels"] = edge_count
        st.image(canny, caption=f"Canny Edges ({ms:.1f}ms) -- {edge_count:,} edge pixels", use_container_width=True, clamp=True)
        progress.progress(30)

        # 3. Lane Detection
        st.subheader("3. Lane Detection")
        t0 = time.perf_counter()
        lanes = lane_detection.detect_lanes(work_img, save=False)
        ms = (time.perf_counter() - t0) * 1000
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        bl = cv2.GaussianBlur(gray, (5, 5), 0)
        ed = cv2.Canny(bl, 50, 150)
        ln = cv2.HoughLinesP(ed, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
        num_lanes = len(ln) if ln is not None else 0
        metrics["Lanes Detected"] = num_lanes
        st.image(_to_rgb(lanes), caption=f"Lanes ({ms:.1f}ms) -- {num_lanes} lines", use_container_width=True)
        progress.progress(50)

        # 4. Corners
        st.subheader("4. Corner Detection")
        t0 = time.perf_counter()
        corners = corner_detection.harris_corner_detection(work_img, save=False)
        ms = (time.perf_counter() - t0) * 1000
        gf = np.float32(cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY))
        hr = cv2.cornerHarris(gf, 2, 3, 0.04)
        corner_count = int(np.count_nonzero(cv2.dilate(hr, None) > 0.01 * hr.max()))
        metrics["Corners Detected"] = corner_count
        st.image(_to_rgb(corners), caption=f"Corners ({ms:.1f}ms) -- {corner_count} corners", use_container_width=True)
        progress.progress(65)

        # 5. Object Detection
        st.subheader("5. Object Detection")
        t0 = time.perf_counter()
        det_img, dets = object_detector.detect_objects(work_img, save=False)
        ms = (time.perf_counter() - t0) * 1000
        metrics["Objects Detected"] = len(dets)
        st.image(_to_rgb(det_img), caption=f"Objects ({ms:.1f}ms) -- {len(dets)} objects", use_container_width=True)
        progress.progress(80)

        # 6. Classification
        st.subheader("6. KNN Classification")
        t0 = time.perf_counter()
        model, acc, names = classifier.train_knn_classifier(save=False)
        label = classifier.predict(work_img, model, names)
        ms = (time.perf_counter() - t0) * 1000
        metrics["KNN Accuracy"] = f"{acc:.2%}"
        metrics["Predicted Class"] = label
        st.success(f"Predicted: **{label}** (accuracy: {acc:.2%}, {ms:.1f}ms)")
        progress.progress(100)

        # Metrics Panel
        st.subheader("📊 Metrics Summary")
        cols = st.columns(len(metrics))
        for col, (name, val) in zip(cols, metrics.items()):
            col.metric(name, val)


# ── COURSE REFERENCE ────────────────────────────────────────────
elif page == "📚 Course Reference":
    st.title("📚 Course Module Reference")
    st.markdown("Which VIT module/session each technique belongs to:")
    st.markdown("""
| VIT Module | Session | Technique | File |
|---|---|---|---|
| Module 1 | Session 1 | Image Formation & Basics | preprocessing.py |
| Module 1 | Session 3 | Flipping, Contrast Reduction | preprocessing.py |
| Module 2 | Session 1 | Noise Removal (Gaussian, Median) | preprocessing.py |
| Module 2 | Session 3 | Log Transformation | preprocessing.py |
| Module 2 | Session 4 | Morphological Operations | preprocessing.py |
| Module 2 | Session 5 | CLAHE Histogram Equalization | preprocessing.py |
| Module 3 | Session 1 | Sobel Edge Detection | edge_detection.py |
| Module 3 | Session 2 | Canny Edge Detection | edge_detection.py |
| Module 3 | Session 3-4 | Harris Corner Detection | corner_detection.py |
| Module 3 | Session 3-4 | Shi-Tomasi Corner Detection | corner_detection.py |
| Module 4 | Session 1-2 | Hough Line Transform (Lanes) | lane_detection.py |
| Module 5 | Session 1-2 | Object Detection (DNN/Contour) | object_detector.py |
| Module 5 | Session 3-4 | Object Tracking (CSRT/KCF) | object_tracker.py |
| Module 5 | Session 5 | KNN Image Classification | classifier.py |
    """)

# ── FOOTER ──────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with OpenCV + Streamlit + scikit-learn | VIT Bhopal Computer Vision BYOP 2026")
