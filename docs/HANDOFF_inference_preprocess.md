# Handoff: `inference_preprocess.py` → API & Streamlit

**From:** Zihao Huang (Edward) — Data Engineer (Role 2)
**To:** Dhruvi Gandhi — Full-Stack & API Developer (Role 5)
**Module:** [`src/inference_preprocess.py`](src/inference_preprocess.py)
**Status:** ready to integrate into `api.py` and `app.py` (W3)

---

## TL;DR

You can replace the inline image-decoding + Haar-cascade + resize blocks in
`api.py` and `app.py` with a single call:

```python
from src.inference_preprocess import preprocess_for_inference

face_arr = preprocess_for_inference(payload)   # (224, 224, 3) uint8 RGB
```

`payload` can be anything a web client sends — file path, raw bytes, base64
string (with or without `data:image/...;base64,` prefix), `PIL.Image`,
`np.ndarray`, or a file-like object such as Streamlit's `UploadedFile`.

The returned array is byte-for-byte compatible with what Beste's SVRs were
trained on (224×224 RGB, aspect-preserving pad-to-square with black bars,
matching `data/processed/faces_standardized/`). Pass it straight into
`FaceToBMIPipeline.predict(...)` — see "Wiring with `pipeline_v1.py`" below.

---

## Why a separate preprocessing module?

`pipeline_v1.py` (Wade) bundles MTCNN + DeepFace + sklearn end-to-end. That's
correct for batch evaluation, but heavy for the Web API hot path:

| Concern | `pipeline_v1.py` raw | `inference_preprocess.py` |
|---|---|---|
| First-call cold start (face detect) | ~1.5 s (MTCNN + torch) | ~50 ms (Haar) |
| RAM cost (face detect) | ~200 MB | ~5 MB |
| EXIF rotation | not handled | handled |
| Input validation (size / dims / bytes) | minimal | rejects bad uploads early |
| Error types | generic `ValueError` | typed: `ImageDecodeError`, `ImageTooLargeError`, `ImageTooSmallError` |

Net effect: faster first request, cleaner 4xx error mapping, same regression
result.

---

## Quick start

### 1. Flask (`api.py`)

```python
from src.inference_preprocess import (
    preprocess_for_inference,
    ImageDecodeError, ImageTooLargeError, ImageTooSmallError,
)
from pipeline_v1 import FaceToBMIPipeline

# Build the pipeline ONCE at process start. auto_detect_face=False because
# inference_preprocess already cropped the face — don't run MTCNN again.
pipe = FaceToBMIPipeline(backbone="ensemble", auto_detect_face=False)

@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")               # data URL or raw base64
    if not image_b64:
        return jsonify({"error": "missing 'image' field"}), 400

    try:
        face_arr, diag = preprocess_for_inference(
            image_b64, return_diagnostics=True
        )
    except ImageTooLargeError as e:
        return jsonify({"error": str(e)}), 413     # Payload Too Large
    except ImageTooSmallError as e:
        return jsonify({"error": str(e)}), 422     # Unprocessable Entity
    except ImageDecodeError as e:
        return jsonify({"error": str(e)}), 400     # Bad Request

    result = pipe.predict(face_arr)                # {"bmi": ..., "confidence": ..., "backbone": ...}
    result["preprocess"] = diag.to_dict()
    return jsonify(result), 200
```

### 2. Streamlit (`app.py`)

```python
from src.inference_preprocess import InferencePreprocessor
from pipeline_v1 import FaceToBMIPipeline

@st.cache_resource
def load_preprocessor():
    return InferencePreprocessor(detect_face=True)

@st.cache_resource
def load_pipeline():
    return FaceToBMIPipeline(backbone="ensemble", auto_detect_face=False)

uploaded = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is not None:
    pre = load_preprocessor()
    pipe = load_pipeline()
    face_arr, diag = pre(uploaded, return_diagnostics=True)   # UploadedFile is file-like
    result = pipe.predict(face_arr)
    st.metric("Predicted BMI", f"{result['bmi']:.1f}")
    st.caption(f"confidence {result['confidence']:.2f}, "
               f"face detected: {diag.face_detected}, "
               f"preprocess {sum(diag.timings_ms.values()):.1f} ms")
```

For the webcam path, `st.camera_input(...)` returns a file-like object too —
same call, no special-casing.

---

## Wiring with `pipeline_v1.py` — important

`FaceToBMIPipeline.predict()` runs its own MTCNN + standardize internally.
If `inference_preprocess` already did face detection + standardization, that
work would be duplicated, which:

- adds ~1.5 s of latency on the first request (MTCNN cold start), and
- makes the inference path inconsistent with the lightweight one we just built.

**Construct the pipeline with `auto_detect_face=False`:**

```python
pipe = FaceToBMIPipeline(backbone="ensemble", auto_detect_face=False)
```

This still calls `_standardize_face` inside `predict()`, but that step is
idempotent on an already-224×224 image — the resize is a no-op, the canvas
copy is microseconds. Safe.

If you want to bypass `predict()` entirely and call into the regressors, use
`pipe.predict_from_features(...)` — but that requires extracting embeddings
yourself (Wade's path is the supported one).

---

## Public API reference

### `preprocess_for_inference(image, *, detect_face=True, return_diagnostics=False, max_bytes=10*1024*1024)`

One-shot preprocessing. Reuses a cached `InferencePreprocessor` when args
match, so back-to-back calls share the loaded Haar cascade.

- **Returns:** `np.ndarray` shape `(224, 224, 3)` dtype `uint8` (RGB).
  If `return_diagnostics=True`, returns `(arr, PreprocessDiagnostics)`.

### `class InferencePreprocessor(*, detect_face=True, max_bytes=10*1024*1024, min_side=32, max_side=8192)`

Long-lived instance for the Flask process or Streamlit app. The constructor
warms the Haar cascade so the first request doesn't pay the load cost.

- `__call__(image, *, return_diagnostics=False)` — same return type as the
  free function above.

### `decode_image(payload, *, max_bytes=...)` → `PIL.Image.Image` (RGB)

Lower-level helper. Use it if you want a `PIL.Image` instead of an array.

### `class PreprocessDiagnostics`

Returned when `return_diagnostics=True`. Fields:

| Field | Type | Meaning |
|---|---|---|
| `original_size` | `(w, h)` | Input image size before EXIF / detect / resize |
| `output_size` | `(224, 224)` | Always 224×224 on success |
| `face_detected` | `bool` | True if Haar found a face |
| `face_box` | `(x1, y1, x2, y2)` or `None` | Bbox in original-image coordinates |
| `timings_ms` | `dict` | Per-stage latency: `decode`, `detect`, `standardize` |

Call `diag.to_dict()` for a JSON-serializable copy.

### Exceptions (subclasses of `PreprocessError` → `ValueError`)

| Class | When raised | Suggested HTTP status |
|---|---|---|
| `ImageDecodeError` | bytes / base64 / file did not decode | **400** Bad Request |
| `ImageTooLargeError` | payload exceeds `max_bytes` *or* image side > `max_side` | **413** Payload Too Large |
| `ImageTooSmallError` | image side < `min_side` (default 32 px) | **422** Unprocessable Entity |

---

## Accepted input formats

| Input type | Example | Notes |
|---|---|---|
| File path | `"/path/to/photo.jpg"` or `Path(...)` | Read from disk; useful in tests |
| Raw bytes | `request.files["file"].read()` | Most common API path |
| Base64 string | `"iVBORw0KGgoAAA..."` | Tolerates `\n` whitespace |
| Data URL | `"data:image/png;base64,iVBOR..."` | Prefix is stripped automatically |
| `PIL.Image` | already-decoded image | Converted to RGB if needed |
| `np.ndarray` | HxW or HxWx3 uint8 | Other dtypes are clipped to uint8 |
| File-like | Streamlit `UploadedFile`, `BytesIO`, `request.files["file"]` | Anything with `.read()` returning bytes |

EXIF rotation (`Orientation` tag) is honoured automatically — portrait phone
photos land upright without extra code on your side.

---

## Output format (the contract you're consuming)

- Shape: `(224, 224, 3)`
- Dtype: `uint8`
- Channel order: **RGB** (not BGR — do **not** `cv2.cvtColor` it)
- Aspect-preserving pad-to-square with black bars, then resize to 224×224
- Bytewise compatible with `data/processed/faces_standardized/img_*.bmp`
- Safe to feed directly into `pipe.predict(arr)` (with `auto_detect_face=False`)

If you ever need to display the array in Streamlit / OpenCV, remember:

```python
st.image(face_arr)                    # OK — Streamlit treats arrays as RGB
cv2.imshow("face", face_arr[..., ::-1])  # cv2 wants BGR
```

---

## Performance notes

Measured on an M-series MacBook with `detect_face=True`:

| Stage | Typical | Notes |
|---|---|---|
| `decode` | 1–5 ms | PNG/JPEG bytes → PIL |
| `detect` | 20–60 ms | Haar cascade on full-res image |
| `standardize` | <1 ms | pad + bilinear resize |

Cold-start cost lives in the constructor (cascade load, ~30 ms once). After
that, throughput is bottlenecked by DeepFace inside `pipe.predict()`, not by
preprocessing.

If you want to skip face detection entirely (e.g., the client already cropped
a face), pass `detect_face=False`. That avoids the cv2 import altogether on
that code path.

---

## Common pitfalls

1. **Don't run MTCNN twice.** Construct the pipeline with
   `auto_detect_face=False` whenever you preprocess upstream.
2. **Don't pre-scale features.** Beste's `.joblib` is `Pipeline(StandardScaler, SVR)`
   — scaling is internal. Just pass raw embeddings (this is `pipe.predict()`'s job).
3. **Don't BGR-convert the output.** It's already RGB, which is what DeepFace
   and PIL expect.
4. **Don't forget EXIF.** The module handles it; if you're decoding manually
   anywhere else, also call `PIL.ImageOps.exif_transpose(img)`.
5. **Catch the typed exceptions.** Mapping them to 400 / 413 / 422 keeps the
   API contract predictable for the frontend.
6. **Cache the preprocessor and pipeline.** In Flask, build them at module
   import. In Streamlit, use `@st.cache_resource`. Loading them per request
   would re-download / re-deserialize the SVRs.

---

## Smoke check

After wiring, run a sanity request to verify the integration:

```bash
# 1. Make sure the regressor side still works
python pipeline_v1.py     # should print Beste's reported r values

# 2. Quick manual test of the preprocessor
python -m src.inference_preprocess /path/to/any/photo.jpg
# expected: "output array: shape=(224, 224, 3) dtype=uint8" + diagnostics
```

Ping me on Slack if anything in this doc doesn't match what you see in the
module — I'll update both. — Edward
