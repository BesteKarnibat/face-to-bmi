# Week 1 Trial & Error Log

## Image Preprocessing

Tried two approaches for standardizing images:

- MTCNN alignment
  - Some images failed (~6%)
  - Added black padding after rotation

- Simple resize/pad
  - No failures
  - Faster and cleaner

Decided to use simple resize for now.

Also noticed dataset issues:
- Missing ~244 images
- Sizes and formats all over the place

Next: run model on standardized images
