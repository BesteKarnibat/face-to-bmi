# Model Decision Log

Owner: Wade Chen (Role 1 - Team Lead & Integration)

Purpose: record the current integration decision based on the model artifacts and notebooks available in the repository. This is a living document. If the modeling team changes the final model, update this file and the README together.

## Current Integration Decision

As of the latest pulled repository state, the strongest reported experiment is Beste's ArcFace update in `models/beste/v4_arcface_ensemble.ipynb`.

Current best research result:

| Candidate | Overall r | Male r | Female r | MAE | Status |
|---|---:|---:|---:|---:|---|
| 2-way ensemble: VGG-Face + ArcFace | 0.7180 | 0.7021 | 0.7438 | 4.6848 | Best reported experiment |
| ArcFace SVR | 0.7153 | 0.6966 | 0.7419 | 4.6434 | Beats paper baseline |
| 3-way weighted ensemble: VGG-Face + FaceNet + ArcFace | 0.7127 | 0.7054 | 0.7291 | 4.7899 | Beats paper baseline |
| 3-way equal ensemble: VGG-Face + FaceNet + ArcFace | 0.7089 | 0.7028 | 0.7241 | 4.8141 | Beats paper baseline |
| Previous 2-way ensemble: VGG-Face + FaceNet | 0.6469 | 0.6555 | 0.6427 | 5.1134 | Currently wired in local pipeline |
| Paper baseline | 0.6500 | 0.7100 | 0.5700 | N/A | Target reference |

Working decision:

- Treat `VGG-Face + ArcFace` as the current best model direction.
- Keep `VGG-Face + FaceNet` as the currently wired demo pipeline until ArcFace feature extraction and model artifacts are committed or synchronized locally.
- Do not describe the production/demo pipeline as ArcFace-ready until `pipeline_v1.py` can load the ArcFace model and reproduce the reported `r = 0.7180` path.

## Why This Decision

The original project target is to beat the paper's overall Pearson correlation of `r = 0.65`. The older VGG-Face + FaceNet ensemble reached `r = 0.6469`, which was close but still below the target. The new ArcFace experiment reports `r = 0.7180`, which clears the target by `0.0680`.

The ArcFace result also uses the same 741-image test subset as Beste's previous feature set after aligning Ryan's ArcFace filenames to Beste's VGG/FaceNet filename order. The notebook explicitly checks that all 3123 training rows and 741 test rows in Beste's subset can be aligned to Ryan's ArcFace handoff.

## Current Artifact Status

Available in this repository:

- `models/beste/models/svr_vgg_tuned.joblib`
- `models/beste/models/svr_facenet_tuned.joblib`
- VGG-Face and FaceNet feature arrays in `models/beste/features/`
- Notebooks documenting ArcFace experiments:
  - `models/beste/v4_arcface_ensemble.ipynb`
  - `models/beste/sample_weighted_optimization.ipynb`

Not currently available in this repository:

- `svr_arcface_tuned.joblib`
- `svr_arcface_weighted_C10.0.joblib`
- Ryan's ArcFace feature handoff arrays
- `winner_info.json`
- The saved `w3_results.csv` / `sample_weight_experiments.csv` outputs referenced by the notebooks

## Integration Implications

The local `pipeline_v1.py` currently supports:

- `vgg`
- `facenet`
- `ensemble` = average of VGG-Face and FaceNet predictions

To make the ArcFace result part of the demo pipeline, the team needs to add:

1. ArcFace feature extraction at inference time.
2. Loading for the ArcFace SVR model artifact.
3. A new ensemble option that averages VGG-Face and ArcFace predictions.
4. An integration sanity test that reproduces the reported ArcFace/VGG result on the 741-row test set.

## Open Questions

- Will the final demo use the highest-r model (`VGG-Face + ArcFace`) or the sample-weighted ArcFace model that reduces regression-to-mean behavior?
- Will ArcFace artifacts be committed to GitHub, stored in Google Drive, or loaded from another shared location?
- Should `pipeline_v1.py` keep the older VGG-Face + FaceNet path as a fallback after ArcFace is integrated?
- Who owns the final update to `pipeline_v1.py` once ArcFace artifacts are ready?

## Next Update Trigger

Update this log when any of the following happens:

- ArcFace `.joblib` and feature arrays are added locally.
- `pipeline_v1.py` gains an ArcFace inference path.
- The team chooses a final model for the report and presentation.
- A new model beats `r = 0.7180` or materially improves calibration/MAE.
