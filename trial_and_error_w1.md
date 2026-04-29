## Week 1 Trial & Error Log

### 1. Initial Data Audit

**Goal:** Understand raw dataset quality before preprocessing.

**Findings:**
- CSV records: 4206
- Actual images: 3962 → **244 missing**
- Image dimensions vary widely: 41×52 → 900×1222
- File sizes: 6 KB → 3 MB
- Most images are portrait (aspect ratio ≈ 0.8)
- Faces occupy ~40% of image area → significant background noise
- 8 images in palette (P) mode, rest RGB

**Conclusion:**  
Dataset is highly inconsistent → requires standardization before feature extraction.

---

### 2. Image Preprocessing Approaches

#### Approach 1: MTCNN Alignment (`faces_aligned/`)
- Face detection + eye alignment (horizontal leveling)
- Cropped around face and resized to 224×224

**Pros:**
- Better face focus
- Eye alignment improves consistency

**Cons:**
- ~6% failure rate (fallback needed)
- Rotation introduces black padding (corner artifacts)
- More complex pipeline

---

#### Approach 2: Simple Pad + Resize (`faces_standardized/`)
- No face detection
- Pad to square with black borders → resize to 224×224

**Pros:**
- 0% failure rate
- Preserves face proportions
- Faster and simpler
- Less severe padding artifacts than alignment method

**Cons:**
- No facial alignment

---

### Decision (End of Week 1)

- Selected **faces_standardized** as the default pipeline  
- Reason: stability > complexity for baseline reproducibility  
- faces_aligned kept as backup for experimentation  

---

### 3. Data Consistency Issue

**Problem Identified:**
- Mismatch between CSV (4206) and available images (3962)

**Impact:**
- Incorrect expected shapes in feature extraction
- Potential misalignment between features and labels

**Resolution:**
- Filtered dataset to valid images only
- Final split:
  - Train: 3210 samples
  - Test: 752 samples
- Standardized across team using `split_ids.csv`

---

### 4. Feature Extraction + Early Modeling Issues

**Initial Goal:** Match paper baseline performance

**Observations:**
- VGG-Face and FaceNet tested
- Could not reach paper performance
- Ensemble (VGG + FaceNet) reached **r ≈ 0.6469** (close but not exceeding)

---

### 5. Critical Bug Identified (Normalization)

**Problem:**
- Used ImageNet normalization for VGG-Face

**Why this is wrong:**
- VGG-Face is trained on face data with **different input distribution**
- ImageNet normalization shifts pixel values incorrectly
- Result: distorted 4096D feature representations

**Impact:**
- Models underperform despite correct pipeline structure
- Features lose semantic meaning

---

### 6. Fix Applied

- Switched to **VGG-Face specific preprocessing (BGR mean subtraction)**
- Re-extracted features
- Released corrected feature set (`handoff_v3`)

---

### 7. Pipeline + Reproducibility Decisions

- Set `random_state = 42` for consistent splits
- Ensured pipeline is reusable for future augmentation
- Saved filenames alongside features → guarantees row alignment
- Introduced standardized train/test split (`3210 / 752`)

---

### 8. Additional Improvements

**Landmark-based Alignment (faces_aligned_v3):**
- Replaced bbox cropping with landmark-driven alignment
- Rotates face to align eyes
- Uses eye-to-mouth distance for scaling
- FFHQ-style cropping

**Results:**
- Significant reduction in padding
- Black border issue largely removed
- No distortion in aspect ratio
- Visually cleaner than previous methods

---

### Key Takeaways

- Raw dataset inconsistency was a major challenge
- Simple preprocessing (faces_standardized) provided the most reliable baseline
- Alignment methods improve quality but introduce trade-offs
- Data mismatch (CSV vs images) required correction before modeling
- **Incorrect normalization was the biggest hidden issue affecting performance**
- Reproducibility (splits, seeds, pipelines) is critical for team coordination
