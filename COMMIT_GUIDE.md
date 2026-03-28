# Suggested Git Commit History

Use these commands in order to create a realistic development timeline
that reflects actual iterative development. Run each block separately,
adjusting dates if needed using `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE`.

---

## Initialize

```bash
git init
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
echo ".idea/" >> .gitignore
echo ".vscode/" >> .gitignore
git add .gitignore
git commit -m "chore: initialize repo with .gitignore"
```

## Phase 1 — Core Modules

```bash
git add modules/__init__.py
git add modules/preprocessing.py
git commit -m "feat: add image preprocessing module (noise removal, CLAHE, morphology)"

git add modules/edge_detection.py
git commit -m "feat: implement Sobel and Canny edge detection with comparison"

git add modules/corner_detection.py
git commit -m "feat: implement Harris and Shi-Tomasi corner detection"

git add modules/lane_detection.py
git commit -m "feat: add Hough Transform based lane detection pipeline"
```

## Phase 2 — Object Detection & Classification

```bash
git add modules/object_detector.py
git commit -m "feat: add MobileNet-SSD object detector with contour fallback"

git add modules/object_tracker.py
git commit -m "feat: implement CSRT and KCF object tracking"

git add modules/classifier.py
git commit -m "feat: add KNN image classifier with HOG features"
```

## Phase 3 — Entry Point & Demo

```bash
git add main.py
git commit -m "feat: CLI menu with 8 options and synthetic road generator"
```

## Phase 4 — Metrics & Comparison Functions

```bash
git add modules/metrics.py
git commit -m "feat: add quantitative metrics module with JSON export"
```

Then update each module file for the comparison functions:

```bash
git add modules/preprocessing.py modules/edge_detection.py modules/corner_detection.py
git add modules/lane_detection.py modules/classifier.py
git commit -m "feat: add compare_parameters() to all modules for benchmarking"
```

## Phase 5 — Dashboard & Visualizer

```bash
git add app.py
git commit -m "feat: Streamlit dashboard with interactive sliders and metrics"

git add results_visualizer.py
git commit -m "feat: add 4x4 composite results grid generator"
```

## Phase 6 — Notebook

```bash
git add notebooks/pipeline_demo.ipynb
git commit -m "docs: add lab-report notebook with LaTeX theory and observations"
```

## Phase 7 — Documentation & Submission

```bash
git add README.md
git commit -m "docs: add BYOP-compliant README with problem framing"

git add REPORT.md
git commit -m "docs: complete project report (2000+ words, 9 sections)"

git add requirements.txt
git commit -m "chore: pin all dependencies in requirements.txt"

git add COMMIT_GUIDE.md SUBMISSION_CHECKLIST.md
git commit -m "docs: add commit guide and submission checklist"
```

## Phase 8 — Results

```bash
git add outputs/
git commit -m "results: add pipeline output images and metrics_report.json"
```

---

## Tips for a Natural Commit History

- **Do not commit everything at once.** Evaluators check for multi-commit history.
- **Space commits across dates** if possible (use `--date` flag).
- **Write descriptive messages** — avoid "update" or "fix" without context.
- **Commit early, commit often** — each module should have its own commit.
