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
| 5 | Full-Stack & API Developer | Dhruvi Gandhi | `app.py` (Streamlit) |
| 6 | Technical Writer & Analyst | Junny Choi | `trial_and_error_w1.md` |

---

## Repository Layout

```
face-to-bmi/
├── README.md                 ← this file
├── requirements.txt          ← unified Python dependencies
├── split_ids.csv             ← CANONICAL train/test split — use this everywhere
├── app.py                    ← Streamlit web app (Role 5)
├── trial_and_error_w1.md     ← team-level weekly engineering log (Role 6)
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
│   │   ├── features/         ← X/y/gender/names .npy splits (gitignored)
│   │   └── models/           ← tuned .joblib SVRs (gitignored)
│   └── ryan/                 ← Role 3: feature extraction iterations
│       ├── feature-extraction-v1.ipynb
│       ├── feature-extraction-v2.ipynb
│       └── feature-verification-v1.ipynb
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
┌──────────────────────┐
│ models/final_model.* │   target Pearson r > 0.65
└─────────┬────────────┘
          │ Role 5 (Dhruvi)
          ▼
┌──────────────────────┐
│ app.py (Streamlit)   │   upload / webcam → BMI prediction
└──────────────────────┘
```

---

## Interface Contracts (Don't break these without team agreement)

| Stage | Input | Output | Owner |
|---|---|---|---|
| Face crop / standardize | raw .bmp | 224×224 RGB BMP, filename = `img_<id>.bmp` | Role 2 |
| Feature extraction | 224×224 RGB | `features.npy` shape `(N, D)`, indexed by `image_id` order | Role 3 |
| Regression | `(N, D)` features + BMI labels | scalar BMI prediction | Role 4 |
| API | image bytes | `{ "bmi": float, "confidence": float }` (TBD W3) | Role 5 |

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

# 5. Run the web app (W1 version: upload + preview only)
streamlit run app.py
```

---

## Roadmap

| Week | Phase | Headline goal |
|---|---|---|
| **W1** | Baseline replication | Match paper's r ≈ 0.65 with VGG-Face + SVR |
| W2 | Data & UI integration | Augmentation; UI shows face-crop preview |
| W3 | Model optimization | Swap backbone (ResNet50 / ArcFace / fine-tune) — beat 0.65 |
| W4 | System API & inference | RESTful API + webcam streaming |
| W5 | Final delivery | Cloud deploy + 10-page report + 3-min demo video |

---

## References

Kocabey, E., Camurcu, M., Ofli, F., Aytar, Y., Marin, J., Torralba, A., Weber, I. (2017). _Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media._ ICWSM 2017.

Original data: [VisualBMI](http://www.visualbmi.com/) (sourced from Reddit r/progresspics).
