# Repository Structure

This repo is organized by workflow area so each team member can find their files without searching through the root directory.

## Root

- `README.md` - project overview, setup notes, and the current repository map.
- `requirements.txt` - main Python dependencies.
- `requirements_api.txt` - smaller dependency set for the API backend. Currently stale because `api.py` uses FastAPI.
- `split_ids.csv` - Wade's cleaned split reference. Current modeling notebooks may use smaller filtered feature subsets.
- `pipeline_v1.py` - end-to-end inference pipeline.
- `1_Upload.py` - Streamlit upload page.
- `api.py` - FastAPI backend.
- `client.py` - API client / demo helper. May target an older API contract.
- `test_api.py` - API smoke tests. May target an older API contract.

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
- `models/beste/` - Beste's model exploration, feature extraction, SVR, SVM, MLP, ArcFace ensemble, and sample-weighting notebooks.

## Model Artifacts

- `models/beste/features/` - extracted feature arrays and labels.
- `models/beste/models/` - trained SVR model files.

The committed model artifacts currently cover the VGG-Face and FaceNet path. ArcFace experiment notebooks are present, but the ArcFace model and feature artifacts are not committed locally at this time.

## Docs

- `docs/HANDOFF_inference_preprocess.md` - handoff notes for inference preprocessing.
- `docs/model_decision_log.md` - current model-selection and integration decision log.
- `docs/e2e_inference_flow.md` - end-to-end inference flow chart and integration notes.
- `docs/trial_and_error_w1.md` - Week 1 engineering log.
- `docs/repository_structure.md` - this file.

## Archive

- `archive/app_backups/` - old app backup files.
- `archive/update_scripts/` - old one-off repo update scripts.
