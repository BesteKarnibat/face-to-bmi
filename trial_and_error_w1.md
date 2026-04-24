# Week 1 Trial & Error Log

## Image Preprocessing

- Problem: Images inconsistent (size, format, missing ~244 images, lots of background)

- Tried:
  - MTCNN alignment (faces_aligned)
    - Better face focus
    - ~6% failures + padding issues
  - Simple pad/resize (faces_standardized)
    - No failures, faster, preserves proportions

- Suggestion: Use faces_standardized for now (more stable)

- Next: run feature extraction + compare to baseline
