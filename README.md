# Face-to-BMI

Replication and improvement of **Kocabey et al. 2017, _Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media_** (ICWSM 2017).

ADSP 31018 — Machine Learning II · Final Project · Team 2

We predict a person's BMI from a single face image, use pre-trained face-recognition backbones as feature extractors, and ship the model behind a Streamlit web API with optional webcam input. **Goal: beat the paper's overall Pearson r = 0.65.**

## Current Status

The latest pulled repository includes Beste's ArcFace optimization notebooks. The current best reported experiment is a **VGG-Face + ArcFace SVR ensemble** with overall Pearson **r = 0.7180**, which beats the paper's overall `r = 0.65`.

Important integration note: the local `pipeline_v1.py` demo path still loads the committed VGG-Face and FaceNet models only. It reproduces the older VGG-Face + FaceNet ensemble at **r = 0.6469**. ArcFace should be treated as the current best model direction, but not yet as the wired local demo pipeline until the ArcFace `.joblib` model and inference path are added.


---


## Repository Layout

```
face-to-bmi/
├── README.md                 ← this file
├── requirements.txt          ← unified Python dependencies
├── requirements_api.txt      ← lightweight API dependencies
├── split_ids.csv             ← Wade's canonical cleaned split reference
├── pipeline_v1.py            ← W2: end-to-end inference pipeline (Role 1)
├── 1_Upload.py               ← Streamlit upload workflow (Role 5)
├── api.py                    ← FastAPI backend (Role 5)
├── client.py                 ← API client / demo helper
├── test_api.py               ← API smoke tests
│
├── data/
│   ├── raw/                  ← original VisualBMI images (gitignored — pulled from Google Drive)
│   │   ├── images/           ← 4206 raw .bmp files (3962 actually present)
│   │   └── data.csv          ← original paper labels: name, bmi, gender, is_training
│   └── processed/            ← Edward's outputs (gitignored)
│       ├── faces_aligned/        ← MTCNN + eye alignment, 224×224 RGB BMP
│       ├── faces_standardized/   ← simple pad/resize, 224×224 RGB BMP  ← USE THIS for now
│       ├── audit_log.csv         ← per-image dimensions / mode / aspect ratio / exists
│       └── standardization_log.csv
│
├── pages/
│   └── 2_Webcam.py           ← Streamlit webcam workflow
│
├── src/
│   ├── dataset.py            ← PyTorch/TF dataset utilities
│   └── inference_preprocess.py ← inference-time preprocessing helpers
│
├── notebooks/
│   └── data/                 ← Role 2: data preparation pipeline
│       ├── 01_audit_and_standardize.ipynb
│       └── 01b_audit_and_align_mtcnn.ipynb
│
├── models/
│   ├── beste/                ← Role 4: baseline SVR + tuning
│   │   ├── 01_data_exploration.ipynb
│   │   ├── VGG 02_feature_extraction.ipynb
│   │   ├── Facenet512 02_feature_extraction.ipynb
│   │   ├── 03_svr_baseline.ipynb
│   │   ├── SVM_MLP_v3.ipynb
│   │   ├── v4_arcface_ensemble.ipynb
│   │   ├── sample_weighted_optimization.ipynb
│   │   ├── features/         ← X/y/gender/names .npy splits
│   │   └── models/           ← tuned .joblib SVRs
│   └── ryan/                 ← Role 3: feature extraction iterations
│       ├── feature-extraction-v1.ipynb
│       ├── feature-extraction-v2.ipynb
│       └── feature-verification-v1.ipynb
│
├── docs/
│   ├── HANDOFF_inference_preprocess.md
│   ├── model_decision_log.md
│   ├── e2e_inference_flow.md
│   └── trial_and_error_w1.md ← team-level weekly engineering log (Role 6)
│
├── archive/
│   ├── app_backups/          ← old app backups kept out of the root
│   └── update_scripts/       ← old one-off repo update scripts
│
└── .gitignore                ← keeps images, .npy, .pth, .joblib out of git
```

---

## Data & Splits

`split_ids.csv` is Wade's cleaned split reference based on the available images and the original `is_training` flag. Some modeling notebooks use a smaller feature subset after additional filtering or feature-handoff alignment. For final model metrics, use the sample counts reported by the model notebook being cited.

| | Count |
|---|---|
| Total images on disk | 3,962 (4,206 in CSV − 244 missing) |
| `split_ids.csv` train | 3,210 |
| `split_ids.csv` test | 752 |
| Current committed VGG/FaceNet feature train | 3,123 |
| Current committed VGG/FaceNet feature test | 741 |
| Unique persons | 2,003 |
| Pair-level isolation (same person not in both splits) | ✅ verified |

### Schema of `split_ids.csv`

| Column | Type | Description |
|---|---|---|
| `image_id` | int | Original index (0–4205) — matches paper data.csv ordering |
| `name` | str | `img_<image_id>.bmp` |
| `person_id` | int | `image_id // 2` — pairs share the same id (before/after) |
| `is_before` | bool | `True` for the heavier "before" photo, `False` for "after" |
| `gender` | {Male, Female} | |
| `bmi` | float | Target — kg/m² |
| `split` | {train, test} | Locked, derived from paper's `is_training` column |
| `pair_complete` | bool | `True` if both before+after images exist on disk |
| `width`, `height`, `mode` | int/str | From `audit_log.csv` |

### Loading example

```python
import pandas as pd
splits = pd.read_csv("split_ids.csv")
train = splits[splits.split == "train"]   # 3,210 rows
test  = splits[splits.split == "test"]    # 752 rows
```

---

## Pipeline — End-to-End Data Flow

```
┌──────────────────────┐
│ data/raw/images/     │   raw 4,206 .bmp images
│ + data.csv (labels)  │   from VisualBMI / progresspics
└─────────┬────────────┘
          │ Role 2 (Edward)
          ▼
┌────────────────────────────────────┐
│ data/processed/faces_standardized/ │   224×224 RGB BMP
│ + audit_log.csv                    │   3,962 images (244 missing)
└─────────┬──────────────────────────┘
          │   ← Role 1 (Wade) consolidates → split_ids.csv
          ▼
┌────────────────────────────┐
│ models/<role>/features/    │   N×4096 (VGG-Face) / N×512 (Facenet512)
│   X_train_*.npy y_train_*  │   extracted via DeepFace / facenet-pytorch
│   X_test_*.npy  y_test_*   │
└─────────┬──────────────────┘
          │ Role 4 (Beste) — SVR, MLP, fine-tune
          ▼
┌──────────────────────────────────────────┐
│ models/beste/models/svr_*.joblib         │   Pipeline(StandardScaler + SVR)
│   svr_vgg_tuned.joblib       (4096→BMI)  │   current local demo path
│   svr_facenet_tuned.joblib   (512→BMI)   │   local ensemble: r = 0.6469
└─────────┬────────────────────────────────┘
          │ Role 1 (Wade) — pipeline_v1.py
          ▼
┌──────────────────────────────────────────┐
│ pipeline_v1.py                           │   image → {bmi, confidence, backbone}
│   FaceToBMIPipeline.predict()            │   MTCNN crop → DeepFace embed → SVR
└─────────┬────────────────────────────────┘
          │ Role 5 (Dhruvi)
          ▼
┌──────────────────────┐
│ Streamlit app        │   1_Upload.py + pages/2_Webcam.py
└──────────────────────┘
```

---

## Interface Contracts (Don't break these without team agreement)

| Stage | Input | Output | Owner |
|---|---|---|---|
| Face crop / standardize | raw .bmp | 224×224 RGB BMP, filename = `img_<id>.bmp` | Role 2 |
| Feature extraction | 224×224 RGB | `features.npy` shape `(N, D)`, indexed by `image_id` order | Role 3 |
| Regression | `(N, D)` features + BMI labels | scalar BMI prediction | Role 4 |
| Inference | image (path / bytes / PIL / ndarray) | `{ "bmi": float, "confidence": float, "backbone": str }` | Role 1 (`pipeline_v1.py`) |
| API | image bytes | calls `FaceToBMIPipeline.predict()` and renders result | Role 5 |

**Random seed convention:** `random_state = 42` everywhere (sklearn, numpy, torch).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/BesteKarnibat/face-to-bmi.git
cd face-to-bmi

# 2. Install
pip install -r requirements.txt

# 3. Pull raw + processed data from the team Google Drive
#    Place under: data/raw/  and  data/processed/

# 4. Verify the canonical split
python -c "import pandas as pd; print(pd.read_csv('split_ids.csv').groupby('split').size())"

# 5. Sanity-check the inference pipeline (loads Beste's joblibs + features)
python pipeline_v1.py

# 6. Run the web app
streamlit run 1_Upload.py
```

---

## Inference Pipeline (`pipeline_v1.py`)

End-to-end: image in, BMI out. Use this anywhere we need a single prediction
(Streamlit, REST API, batch eval, ad-hoc scripts). Current local support is VGG-Face, FaceNet512, and their 2-way ensemble.

```python
from pipeline_v1 import FaceToBMIPipeline

pipe = FaceToBMIPipeline()                       # backbone='ensemble' (default)
result = pipe.predict("photo.jpg")
# {'bmi': 25.4, 'confidence': 0.82, 'backbone': 'ensemble'}
```

### Backbone options

| Backbone | What it loads | Reported r (test) |
|---|---|---|
| `vgg` | `svr_vgg_tuned.joblib` only | 0.6261 (Beste) |
| `facenet` | `svr_facenet_tuned.joblib` only | 0.6144 (Beste) |
| `ensemble` *(default)* | both, averages predictions | **0.6469** (Beste) |

### Current best experiment not yet wired

Beste's `models/beste/v4_arcface_ensemble.ipynb` reports a VGG-Face + ArcFace ensemble at **r = 0.7180** on the 741-row aligned test set. That result is the current best model direction, but it requires ArcFace artifacts and an ArcFace inference path before it becomes the app default.

### Accepted input types

`predict()` accepts: file path (`str` / `Path`), raw image `bytes`, `PIL.Image`,
HxWx3 `np.ndarray`, or a file-like object (e.g. Streamlit's `UploadedFile`).

### Reproducibility

Inference replicates Beste's training-time DeepFace call exactly:
`model_name='VGG-Face' | 'Facenet512'`, `enforce_detection=False`,
`detector_backend='skip'`. Beste's joblib is a `Pipeline(StandardScaler + SVR)`
— scaling is applied internally; do **not** pre-scale.

### Integration sanity test

`python pipeline_v1.py` loads Beste's saved test features and verifies the
regressor produces her reported Pearson r values. Run after pulling new
`.joblib` or `.npy` files from Drive.

---

## Roadmap

| Week | Phase | Headline goal | Status |
|---|---|---|---|
| W1 | Baseline replication | Match paper's r ≈ 0.65 with VGG-Face + SVR | Done |
| W2 | Inference + UI integration | `pipeline_v1.py` end-to-end; Streamlit calls into it | Done for VGG/FaceNet path |
| W3 | Model optimization | Add stronger backbone and beat 0.65 | Best reported: ArcFace ensemble r = 0.7180 |
| W4 | System API & inference | RESTful API + webcam streaming | Initial FastAPI + Streamlit pages present |
| W5 | Final delivery | Cloud deploy + 10-page report + 3-min demo video | |

## Known Gaps

- ArcFace is the current best reported model direction, but ArcFace model artifacts are not committed locally yet.
- `pipeline_v1.py` still needs an ArcFace-backed option before the demo can reproduce `r = 0.7180`.
- `client.py` and `test_api.py` appear to target an older API contract; the current demo path is `api.py` + `1_Upload.py` + `pages/2_Webcam.py`.
- `requirements_api.txt` is stale relative to `api.py`; `api.py` uses FastAPI, not Flask.

---

## References

Kocabey, E., Camurcu, M., Ofli, F., Aytar, Y., Marin, J., Torralba, A., Weber, I. (2017). _Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media._ ICWSM 2017.

Original data: [VisualBMI](http://www.visualbmi.com/) (sourced from Reddit r/progresspics).
