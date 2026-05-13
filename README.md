# Face-to-BMI

Replication and improvement of **Kocabey et al. 2017, _Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media_** (ICWSM 2017).

ADSP 31018 — Machine Learning II · Final Project · Team 2

We predict a person's BMI from a single face image, fine-tune a pre-trained face recognition backbone, and ship the model behind a Streamlit web API with optional webcam input. **Goal: beat the paper's overall Pearson r = 0.65.**

---

## Team & Roles

| # | Role | Member | W1 deliverable |
|---|------|--------|----------------|
| 1 | Team Lead & Integration | **Wade Chen** | `README.md`, `requirements.txt`, `split_ids.csv` |
| 2 | Data Engineer | Zihao Huang (Edward) | Face detection + standardization (`notebooks/data/`) |
| 3 | ML Researcher A — Feature Architect | Ryan (Jung) Chen | Feature extraction notebooks (`models/ryan/`) |
| 4 | ML Researcher B — Optimization | Beste Karnibat | Baseline SVR + tuning (`models/beste/`) |
| 5 | Full-Stack & API Developer | Dhruvi Gandhi | `1_Upload.py`, `pages/2_Webcam.py` (Streamlit) |
| 6 | Technical Writer & Analyst | Junny Choi | `docs/trial_and_error_w1.md` |

---

## Repository Layout

```
face-to-bmi/
├── README.md                 ← this file
├── requirements.txt          ← unified Python dependencies
├── requirements_api.txt      ← lightweight API dependencies
├── split_ids.csv             ← CANONICAL train/test split — use this everywhere
├── pipeline_v1.py            ← W2: end-to-end inference pipeline (Role 1)
├── 1_Upload.py               ← Streamlit upload workflow (Role 5)
├── api.py                    ← Flask API backend (Role 5)
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
│   │   ├── features/         ← X/y/gender/names .npy splits
│   │   └── models/           ← tuned .joblib SVRs
│   └── ryan/                 ← Role 3: feature extraction iterations
│       ├── feature-extraction-v1.ipynb
│       ├── feature-extraction-v2.ipynb
│       └── feature-verification-v1.ipynb
│
├── docs/
│   ├── HANDOFF_inference_preprocess.md
│   └── trial_and_error_w1.md ← team-level weekly engineering log (Role 6)
│
├── archive/
│   ├── app_backups/          ← old app backups kept out of the root
│   └── update_scripts/       ← old one-off repo update scripts
│
└── .gitignore                ← keeps images, .npy, .pth, .joblib out of git
```

---

## Data & Canonical Split

The **only** train/test split everyone uses is **`split_ids.csv`**. Do not resplit.

| | Count |
|---|---|
| Total images on disk | 3,962 (4,206 in CSV − 244 missing) |
| Train | 3,210 |
| Test | 752 |
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
│   svr_vgg_tuned.joblib       (4096→BMI)  │   ensemble best so far: r ≈ 0.65
│   svr_facenet_tuned.joblib   (512→BMI)   │
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
(Streamlit, REST API, batch eval, ad-hoc scripts).

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
| `facenet` | `svr_facenet_tuned.joblib` only | — |
| `ensemble` *(default)* | both, averages predictions | **0.6469** (Beste) |

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
| W1 | Baseline replication | Match paper's r ≈ 0.65 with VGG-Face + SVR | ✅ done (ensemble: 0.6469) |
| **W2** | Inference + UI integration | `pipeline_v1.py` end-to-end; Streamlit calls into it | 🟡 in progress |
| W3 | Model optimization | Swap backbone (ResNet50 / ArcFace / fine-tune) — beat 0.65 | |
| W4 | System API & inference | RESTful API + webcam streaming | |
| W5 | Final delivery | Cloud deploy + 10-page report + 3-min demo video | |

---

## References

Kocabey, E., Camurcu, M., Ofli, F., Aytar, Y., Marin, J., Torralba, A., Weber, I. (2017). _Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media._ ICWSM 2017.

Original data: [VisualBMI](http://www.visualbmi.com/) (sourced from Reddit r/progresspics).
