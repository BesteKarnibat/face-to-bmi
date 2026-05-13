# Repository Structure

This repo is organized by workflow area so each team member can find their files without searching through the root directory.

## Root

- `README.md` - project overview, setup notes, and the current repository map.
- `requirements.txt` - main Python dependencies.
- `requirements_api.txt` - smaller dependency set for the API backend.
- `split_ids.csv` - canonical train/test split used by the whole team.
- `pipeline_v1.py` - end-to-end inference pipeline.
- `1_Upload.py` - Streamlit upload page.
- `api.py` - Flask API backend.
- `client.py` - API client / demo helper.
- `test_api.py` - API smoke tests.

## App

- `pages/2_Webcam.py` - Streamlit webcam page.

Run the app with:

```bash
streamlit run 1_Upload.py
```

## Source Code

- `src/dataset.py` - dataset utilities for model training.
- `src/inference_preprocess.py` - preprocessing helpers used during inference.
- `src/__init__.py` - package marker.

## Data

- `data/raw/` - raw VisualBMI images and labels when available locally.
- `data/processed/` - standardized or aligned face outputs when available locally.

The `.gitkeep` files keep these folders visible even when the data itself is not present.

## Notebooks

- `notebooks/data/` - data audit, face standardization, and face alignment notebooks.
- `models/ryan/` - Ryan's feature extraction and verification notebooks.
- `models/beste/` - Beste's model exploration, feature extraction, SVR, SVM, and MLP notebooks.

## Model Artifacts

- `models/beste/features/` - extracted feature arrays and labels.
- `models/beste/models/` - trained SVR model files.

These are left in place for the current team workflow.

## Docs

- `docs/HANDOFF_inference_preprocess.md` - handoff notes for inference preprocessing.
- `docs/trial_and_error_w1.md` - Week 1 engineering log.
- `docs/repository_structure.md` - this file.

## Archive

- `archive/app_backups/` - old app backup files.
- `archive/update_scripts/` - old one-off repo update scripts.
