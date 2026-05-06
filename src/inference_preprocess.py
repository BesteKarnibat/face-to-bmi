"""inference_preprocess.py - Lightweight single-image preprocessing for the Web API.

Owner: Zihao Huang (Edward), Data Engineer (Role 2) - W3 deliverable.

Why this module exists
----------------------
`pipeline_v1.py` (Wade) bundles MTCNN + DeepFace + sklearn into the inference
path. That is correct for batch evaluation but heavy for a Web API:
    - facenet-pytorch / torch import: ~1.5 s cold start, ~200 MB RAM
    - DeepFace pulls TensorFlow + 4096-D backbone weights
    - First-call latency dominates the user-perceived response time

This module is the lightweight front of the API. It stays in PIL + numpy + cv2
(all already pinned in `requirements_api.txt`) and produces an array byte-for-byte
compatible with what Beste's SVRs were trained on:

    bytes / path / base64 / PIL / ndarray / file-like
        -> validation (size, format, dims)
        -> EXIF rotation
        -> optional face crop (OpenCV Haar; falls back to no-crop)
        -> pad-to-square + resize 224x224 RGB
        -> uint8 ndarray, shape (224, 224, 3)

The pad-to-square + resize step is a direct port of the procedure in
`notebooks/data/01_audit_and_standardize.ipynb`, so inference inputs match
the `data/processed/faces_standardized/` distribution.

Public API
----------
    preprocess_for_inference(image, detect_face=True, return_diagnostics=False)
    decode_image(payload) -> PIL.Image.Image
    InferencePreprocessor(detect_face=True, max_bytes=10 * 1024 * 1024)

Run as a script
---------------
    python -m src.inference_preprocess <path-to-image>
        prints output array shape and per-stage timing.
"""

from __future__ import annotations

import base64
import binascii
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration - keep aligned with Edward's training-time standardization.
# ---------------------------------------------------------------------------

TARGET_SIZE: Tuple[int, int] = (224, 224)        # matches faces_standardized/
PAD_COLOR: Tuple[int, int, int] = (0, 0, 0)      # black bars, matches notebook
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024        # 10 MB upload cap
DEFAULT_MIN_SIDE: int = 32                       # reject anything smaller
DEFAULT_MAX_SIDE: int = 8192                     # reject absurd uploads early
HAAR_MARGIN: float = 0.20                        # extra context around face bbox

ImageInput = Union[str, Path, bytes, bytearray, memoryview, np.ndarray, Image.Image, io.IOBase]


# ---------------------------------------------------------------------------
# Errors - distinct types so the API layer can map to HTTP status codes.
# ---------------------------------------------------------------------------

class PreprocessError(ValueError):
    """Base class for any preprocessing failure that should surface to the client."""


class ImageDecodeError(PreprocessError):
    """Bytes / base64 / file did not decode into a valid image."""


class ImageTooLargeError(PreprocessError):
    """Payload exceeded the configured byte or pixel limits."""


class ImageTooSmallError(PreprocessError):
    """Image is below the minimum usable size."""


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class PreprocessDiagnostics:
    original_size: Tuple[int, int]                  # (width, height) before any processing
    output_size: Tuple[int, int] = TARGET_SIZE      # always 224x224 on success
    face_detected: bool = False
    face_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in original coords
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original_size": list(self.original_size),
            "output_size": list(self.output_size),
            "face_detected": self.face_detected,
            "face_box": list(self.face_box) if self.face_box else None,
            "timings_ms": self.timings_ms,
        }


# ---------------------------------------------------------------------------
# Decoding - accept the variety of payloads a Web API actually receives.
# ---------------------------------------------------------------------------

_DATA_URL_PREFIX = "data:"


def _strip_data_url(s: str) -> str:
    if s.startswith(_DATA_URL_PREFIX) and "," in s:
        return s.split(",", 1)[1]
    return s


def _bytes_from_base64(s: str) -> bytes:
    try:
        return base64.b64decode(_strip_data_url(s), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageDecodeError(f"invalid base64 payload: {e}") from e


def _looks_like_base64(s: str) -> bool:
    if s.startswith(_DATA_URL_PREFIX):
        return True
    head = s.lstrip()[:32]
    return bool(head) and all(c.isalnum() or c in "+/=\n\r " for c in head)


def _is_existing_file(s: str) -> bool:
    try:
        return Path(s).expanduser().is_file()
    except (OSError, ValueError):
        return False


def decode_image(payload: ImageInput, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Image.Image:
    """Coerce any supported payload into a PIL.Image in RGB mode.

    Accepted inputs:
        str | Path                 - filesystem path
        str (base64 / data URL)    - decoded as image bytes
        bytes | bytearray          - raw image bytes
        memoryview                 - raw image bytes
        np.ndarray (HxW or HxWx3)  - uint8 array (other dtypes get clipped)
        PIL.Image.Image            - returned as-is (after RGB convert)
        file-like with .read()     - read once, decoded as bytes

    Raises:
        ImageDecodeError if the payload is unrecognisable.
        ImageTooLargeError if raw bytes exceed `max_bytes`.
    """
    if isinstance(payload, Image.Image):
        return payload.convert("RGB") if payload.mode != "RGB" else payload

    if isinstance(payload, np.ndarray):
        arr = payload
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return Image.fromarray(arr[..., :3], mode="RGB")
        raise ImageDecodeError(f"unsupported ndarray shape {arr.shape}")

    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
    elif isinstance(payload, (str, Path)):
        # Disambiguate: data-URL > short existing path > generic base64.
        # Skip is_file() on long strings - the OS rejects them with ENAMETOOLONG.
        s = str(payload)
        if isinstance(payload, Path):
            raw = payload.expanduser().read_bytes()
        elif s.startswith(_DATA_URL_PREFIX):
            raw = _bytes_from_base64(s)
        elif len(s) <= 4096 and _is_existing_file(s):
            raw = Path(s).expanduser().read_bytes()
        elif _looks_like_base64(s):
            raw = _bytes_from_base64(s)
        else:
            raise ImageDecodeError(
                "string is neither an existing file path nor a base64 / data-URL payload"
            )
    elif hasattr(payload, "read"):
        raw = payload.read()
        if not isinstance(raw, (bytes, bytearray)):
            raise ImageDecodeError(
                f"file-like .read() returned {type(raw).__name__}, expected bytes"
            )
    else:
        raise ImageDecodeError(f"unsupported input type: {type(payload).__name__}")

    if len(raw) > max_bytes:
        raise ImageTooLargeError(f"payload {len(raw)} bytes exceeds limit {max_bytes}")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ImageDecodeError(f"could not decode image bytes: {e}") from e

    return img.convert("RGB") if img.mode != "RGB" else img


# ---------------------------------------------------------------------------
# Optional face detection - OpenCV Haar (lightweight, lazy-loaded).
# ---------------------------------------------------------------------------

_haar_cascade = None


def _get_haar_cascade():
    """Lazy-load and cache the frontal-face Haar cascade."""
    global _haar_cascade
    if _haar_cascade is not None:
        return _haar_cascade
    import cv2  # imported lazily so callers that pass detect_face=False stay cv2-free

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    _haar_cascade = cascade
    return cascade


def _detect_largest_face(
    pil_img: Image.Image,
    margin: float = HAAR_MARGIN,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) for the largest detected face, or None.

    Uses Haar cascade because it is already a hard dependency of the API and
    runs in tens of milliseconds on CPU. Margin expands the bbox so the crop
    keeps the same loose framing Edward's standardized images have.
    """
    import cv2

    arr = np.asarray(pil_img)                # HxWx3 RGB
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade = _get_haar_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return None

    # Pick the largest box - most likely the subject in single-face uploads.
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(margin * max(w, h))
    H, W = arr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Standardization - aspect-preserving pad to square + resize to 224x224.
# ---------------------------------------------------------------------------

def _pad_and_resize(pil_img: Image.Image, target: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Aspect-preserving resize + black pad. Mirrors `01_audit_and_standardize.ipynb`."""
    w, h = pil_img.size
    if w == 0 or h == 0:
        raise ImageTooSmallError(f"image has zero-size axis: {pil_img.size}")

    scale = min(target[0] / w, target[1] / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", target, color=PAD_COLOR)
    canvas.paste(resized, ((target[0] - new_w) // 2, (target[1] - new_h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

class InferencePreprocessor:
    """Reusable single-image preprocessor for the Web API hot path.

    A long-lived instance avoids reloading the Haar cascade per request.

    Parameters
    ----------
    detect_face : bool, default True
        Run Haar face detection + crop with margin. Falls back to no-crop if
        no face is found (the SVRs are tolerant; standardization still runs).
    max_bytes : int, default 10 MB
        Reject raw payloads larger than this before attempting to decode.
    min_side, max_side : int
        Reject decoded images whose smaller / larger side is outside this band.
    """

    def __init__(
        self,
        *,
        detect_face: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        min_side: int = DEFAULT_MIN_SIDE,
        max_side: int = DEFAULT_MAX_SIDE,
    ) -> None:
        self.detect_face = detect_face
        self.max_bytes = max_bytes
        self.min_side = min_side
        self.max_side = max_side
        if detect_face:
            # Warm the cascade now so first request does not pay the load cost.
            _get_haar_cascade()

    def __call__(
        self,
        image: ImageInput,
        *,
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, PreprocessDiagnostics]]:
        timings: dict = {}

        t0 = time.perf_counter()
        pil = decode_image(image, max_bytes=self.max_bytes)
        # Honour camera EXIF rotation so portrait phone uploads land upright.
        pil = ImageOps.exif_transpose(pil)
        timings["decode"] = round((time.perf_counter() - t0) * 1000, 2)

        w, h = pil.size
        if min(w, h) < self.min_side:
            raise ImageTooSmallError(
                f"image {w}x{h} smaller than min_side={self.min_side}"
            )
        if max(w, h) > self.max_side:
            raise ImageTooLargeError(
                f"image {w}x{h} larger than max_side={self.max_side}"
            )

        diag = PreprocessDiagnostics(original_size=(w, h))

        if self.detect_face:
            t0 = time.perf_counter()
            box = _detect_largest_face(pil)
            timings["detect"] = round((time.perf_counter() - t0) * 1000, 2)
            if box is not None:
                pil = pil.crop(box)
                diag.face_detected = True
                diag.face_box = box

        t0 = time.perf_counter()
        out_pil = _pad_and_resize(pil)
        arr = np.asarray(out_pil, dtype=np.uint8)
        timings["standardize"] = round((time.perf_counter() - t0) * 1000, 2)

        diag.timings_ms = timings

        if return_diagnostics:
            return arr, diag
        return arr


# Module-level shortcut for callers that do not want to manage an instance.
_default_preprocessor: Optional[InferencePreprocessor] = None


def preprocess_for_inference(
    image: ImageInput,
    *,
    detect_face: bool = True,
    return_diagnostics: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Union[np.ndarray, Tuple[np.ndarray, PreprocessDiagnostics]]:
    """One-shot preprocessing. Reuses a cached preprocessor when defaults match."""
    global _default_preprocessor
    if (
        _default_preprocessor is None
        or _default_preprocessor.detect_face != detect_face
        or _default_preprocessor.max_bytes != max_bytes
    ):
        _default_preprocessor = InferencePreprocessor(
            detect_face=detect_face, max_bytes=max_bytes
        )
    return _default_preprocessor(image, return_diagnostics=return_diagnostics)


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run inference preprocessing on a single image.")
    parser.add_argument("path", type=str, help="Path to an image file.")
    parser.add_argument("--no-detect", action="store_true", help="Skip Haar face detection.")
    args = parser.parse_args()

    arr, diag = preprocess_for_inference(
        args.path,
        detect_face=not args.no_detect,
        return_diagnostics=True,
    )
    print(f"output array: shape={arr.shape} dtype={arr.dtype}")
    print(f"diagnostics : {diag.to_dict()}")


if __name__ == "__main__":
    _main()
