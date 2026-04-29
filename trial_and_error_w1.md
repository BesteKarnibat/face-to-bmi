## Week 1 Trial & Error Log

### Image Preprocessing

**Problem:**  
Images were inconsistent in size and format, with ~244 missing images and significant background noise.

---

### Tried

**1. MTCNN Alignment (faces_aligned)**  
- Improved face focus  
- ~6% failure rate  
- Introduced padding issues and black borders  

**2. Simple Pad/Resize (faces_standardized)**  
- No failures  
- Faster processing  
- Preserved proportions  
- Selected as the stable baseline for Week 1  

**3. Landmark-based Alignment (faces_aligned_v3)**  
- Replaced bbox + margin method with landmark-driven alignment  
- Rotates faces to align eyes and uses eye-to-mouth distance for scaling  
- FFHQ-style cropping applied  

**Results:**  
- Padding significantly reduced  
- Black border issue largely eliminated  
- No aspect ratio distortion  
- Visually cleaner and more consistent faces  

**Conclusion:**  
faces_aligned_v3 shows clear improvement and should be considered for future experiments.

---

### Feature Extraction Issues

**Problem:**  
Model performance was significantly below the paper benchmark.

**Root Cause:**  
- ImageNet normalization was applied to VGG-Face  
- VGG-Face expects BGR mean subtraction  
- This mismatch distorted the 4096D feature representations  

**Fix:**  
- Re-extracted features using correct VGG-Face preprocessing  
- Released updated dataset: `handoff_v3`

---

### Data Consistency Issue

**Problem:**  
- CSV contained 4206 records  
- Only 3962 images available  

**Resolution:**  
- Final aligned dataset:  
  - Train: 3210 samples  
  - Test: 752 samples  
- Confirmed consistency across all team pipelines  

---

### Modeling Observations

- SVR on initial features performed poorly (r ≈ 0.35–0.40)  
- Ensemble (VGG + FaceNet) achieved r = 0.6469 (close to paper but not exceeding it)  
- Indicates feature quality and preprocessing are critical  
- Classical ML models show limitations on these embeddings  

---

### Key Takeaways

- faces_standardized provided a reliable baseline for Week 1  
- Incorrect normalization was a major hidden issue affecting performance  
- handoff_v3 resolves feature extraction inconsistencies  
- Consistent dataset alignment and reproducibility are critical  
- Likely need to move toward deep learning or fine-tuning approaches  

---

### Next Steps

- Compare faces_standardized vs faces_aligned_v3 for feature extraction  
- Use only `handoff_v3` features for modeling  
- Test MLP on 4096D features  
- Move to end-to-end CNN fine-tuning if performance remains below benchmark  
- Continue tracking preprocessing impact on model performance  
