# BYOP Submission Checklist

Verify every item before submitting on VITyarthi.

---

## GitHub Repository

- [ ] Repository is **PUBLIC** (not private)
- [ ] All files are committed via `git add` + `git commit` (not uploaded as a zip via the GitHub UI)
- [ ] Commit history shows iterative development — at least 10 commits, not one bulk upload
- [ ] `.gitignore` is present and excludes `__pycache__/`, `.env`, `*.pyc`, `venv/`
- [ ] No API keys, passwords, or credentials in any file
- [ ] Repository name is clean and descriptive (e.g., `road-scene-analysis`)

## README.md

- [ ] Problem statement is in the first section — explains **why** this project matters
- [ ] Setup instructions (`pip install -r requirements.txt`) are tested on a fresh environment
- [ ] Course coverage table maps every technique to its VIT module and session
- [ ] Project structure tree is present and accurate
- [ ] Author details (name, roll number, course, year) are filled in — not placeholders
- [ ] Limitations and future work sections are present

## Project Report (REPORT.md)

- [ ] All 9 sections are present and filled in:
  - Abstract, Problem, Objectives, Approach, Implementation, Results, Challenges, Learnings, Conclusion
- [ ] Word count is **>1800 words** (check with: `(Get-Content REPORT.md | Measure-Object -Word).Words`)
- [ ] Section 1 (Problem) and Section 7 (Learnings) use first-person voice
- [ ] Metrics table in Section 5 is populated with actual values from `metrics_report.json`
- [ ] Challenges section (Section 6) is honest and describes real issues faced
- [ ] Section 9 has 6 references in IEEE format
- [ ] Student details table at the top is filled in — not placeholders

## Code Quality

- [ ] All modules in `modules/` have docstrings on every public function
- [ ] `python main.py` runs without errors — option 8 completes the full synthetic demo
- [ ] `streamlit run app.py` launches the dashboard without errors
- [ ] `jupyter notebook notebooks/pipeline_demo.ipynb` opens; all cells run top-to-bottom
- [ ] `python results_visualizer.py` generates `outputs/FULL_PIPELINE_RESULTS.png`
- [ ] `outputs/` folder contains all generated result images (16+ files)
- [ ] `outputs/metrics_report.json` exists and contains actual metric values
- [ ] No hardcoded absolute paths in any Python file (all paths use `Path(__file__)`)

## VITyarthi Platform

- [ ] GitHub repository URL is submitted on VITyarthi
- [ ] REPORT.md (or PDF export of it) is uploaded to VITyarthi
- [ ] README.md is confirmed accessible from the repository root page
- [ ] Notebook (`.ipynb`) is viewable directly on GitHub (renders LaTeX)

---

## Quick Verification Commands

```bash
# Check REPORT.md word count (PowerShell)
(Get-Content REPORT.md | Measure-Object -Word).Words

# Run full synthetic demo
python main.py
# -> Select option 8

# Generate metrics report
python -c "import matplotlib; matplotlib.use('Agg'); import sys; sys.path.insert(0,'.'); from main import generate_synthetic_road_image; from modules.metrics import generate_metrics_report; import cv2; img = generate_synthetic_road_image(640,480); generate_metrics_report(img)"

# Generate composite results image
python results_visualizer.py

# Count output files
(Get-ChildItem outputs/).Count

# Test Streamlit launch (Ctrl+C to exit)
streamlit run app.py
```
