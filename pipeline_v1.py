"""
pipeline_v1.py — Face-to-BMI End-to-End Inference Pipeline (Week 2)
====================================================================

Owner: Wade Chen (Role 1 — Team Lead & Integration)
Scope: Glue layer between teammates' artifacts. Image in → BMI out.

Producers (whose outputs we consume)
------------------------------------
    Edward (Role 2):
        Pre-processing standard — 224×224 RGB, aspect-preserving pad/resize.
        We replicate that target shape inside `_standardize_face()`.

    Beste (Role 4):
        Trained regressors at `models/beste/models/*.joblib`. Each is a
        sklearn `Pipeline([StandardScaler(), SVR(...)])` — i.e., scaling is
        embedded. Pass raw embeddings; do NOT pre-scale.
            • svr_vgg_tuned.joblib       expects (N, 4096)   VGG-Face
            • svr_facenet_tuned.joblib   expects (N, 512)    Facenet512

    Ryan (Role 3):
        Ryan ships PCA-reduced features (1071D) + a saved scaler/PCA pair.
        Empirical result on his features so far: r=0.40 (Beste, SVR_handoff_v2).
        Beste's direct DeepFace path (this module's default) hits r=0.6469
        ensemble. We therefore default to Beste's path. A `ryan_pca` backbone
        can be added later by extending `__init__` and adding load logic.

Consumers
---------
    Dhruvi (Role 5):
        `app.py` imports `FaceToBMIPipeline` for the Streamlit predict button.

Public Interface (frozen — coordinate before changing)
------------------------------------------------------
    >>> from pipeline_v1 import FaceToBMIPipeline
    >>> pipe = FaceToBMIPipeline()                 # backbone='ensemble'
    >>> result = pipe.predict("path/to/photo.jpg")
    >>> result
    {"bmi": 24.7, "confidence": 0.82, "backbone": "ensemble"}

    >>> # Bypass image extraction (for testing against teammates' saved features)
    >>> bmis = pipe.predict_from_features(X_test_4096, backbone="vgg")

Reproducibility note
--------------------
Inference replicates Beste's training-time extraction call exactly:

    DeepFace.represent(
        img_path        = <H×W×3 uint8 RGB ndarray>,
        model_name      = 'VGG-Face' | 'Facenet512',
        enforce_detection = False,
        detector_backend  = 'skip',
    )

`detector_backend='skip'` is correct because we hand DeepFace an already-
face-cropped image. For raw user uploads we run MTCNN ourselves first
(`auto_detect_face=True`, the default).

Run as a script
---------------
    python pipeline_v1.py
        ↳ runs the integration sanity test against Beste's saved test features.
"""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_VGG_MODEL_PATH     = REPO_ROOT / "models" / "beste" / "models" / "svr_vgg_tuned.joblib"
DEFAULT_FACENET_MODEL_PATH = REPO_ROOT / "models" / "beste" / "models" / "svr_facenet_tuned.joblib"

# Standardization target — matches Edward's faces_standardized/
TARGET_SIZE: Tuple[int, int] = (224, 224)

# DeepFace extraction kwargs — must match Beste's training-time call exactly.
DEEPFACE_KWARGS = dict(enforce_detection=False, detector_backend="skip")

# Plausible BMI range. Predictions outside this band get clipped + flagged.
BMI_CLIP_MIN = 12.0
BMI_CLIP_MAX = 70.0

# Expected feature dimensions from each backbone.
VGG_EMBED_DIM     = 4096
FACENET_EMBED_DIM = 512


# --------------------------------------------------------------------------
# 2. Result container
# --------------------------------------------------------------------------

@dataclass
class PredictionResult:
    bmi: float
    confidence: float
    backbone: str
    diagnostics: dict

    def to_dict(self, include_diagnostics: bool = False) -> dict:
        out = {"bmi": self.bmi, "confidence": self.confidence, "backbone": self.backbone}
        if include_diagnostics:
            out["diagnostics"] = self.diagnostics
        return out


# --------------------------------------------------------------------------
# 3. Image normalization helpers (any input → PIL RGB → 224×224 ndarray)
# --------------------------------------------------------------------------

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray, Image.Image, io.IOBase]


def _to_pil_rgb(image: ImageInput) -> Image.Image:
    """Coerce arbitrary image input (path, bytes, ndarray, file-like, PIL) into PIL RGB."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        img = Image.fromarray(image)
    elif hasattr(image, "read"):
        img = Image.open(image)
    else:
        raise TypeError(f"Unsupported image input type: {type(image).__name__}")

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _center_crop_square(pil_img: Image.Image) -> Image.Image:
    """Largest centered square crop. Used as MTCNN fallback."""
    w, h = pil_img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return pil_img.crop((left, top, left + s, top + s))


# Lazy-init MTCNN (heavy import; only loaded when needed)
_mtcnn_singleton = None


def _detect_and_crop_face(pil_img: Image.Image, margin: float = 0.20) -> Image.Image:
    """
    MTCNN face detect + crop with margin. Falls back to center-square on failure.

    `margin` adds extra context around the bbox so the face isn't tightly cropped
    (matches the looser framing that Edward's faces_standardized/ produces).
    """
    global _mtcnn_singleton
    try:
        if _mtcnn_singleton is None:
            from facenet_pytorch import MTCNN  # type: ignore
            _mtcnn_singleton = MTCNN(keep_all=False, device="cpu", post_process=False)
    except Exception as e:
        warnings.warn(f"MTCNN unavailable ({e}); falling back to center crop", stacklevel=2)
        return _center_crop_square(pil_img)

    boxes, probs = _mtcnn_singleton.detect(pil_img)
    if boxes is None or len(boxes) == 0 or probs[0] is None:
        logger.warning("No face detected — falling back to center crop")
        return _center_crop_square(pil_img)

    idx = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[idx]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(pil_img.width,  int(cx + side / 2))
    ny2 = min(pil_img.height, int(cy + side / 2))
    return pil_img.crop((nx1, ny1, nx2, ny2))


def _standardize_face(pil_img: Image.Image, target: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Aspect-preserving resize + black pad to `target`. Returns HWC uint8 RGB ndarray.

    Mirrors Edward's faces_standardized/ procedure so that inference-time
    images match the distribution Beste's SVR was trained on.
    """
    w, h = pil_img.size
    scale = min(target[0] / w, target[1] / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", target, color=(0, 0, 0))
    px = (target[0] - new_w) // 2
    py = (target[1] - new_h) // 2
    canvas.paste(resized, (px, py))
    return np.asarray(canvas)


# --------------------------------------------------------------------------
# 4. Feature extraction (DeepFace wrappers — match Beste's exact call)
# --------------------------------------------------------------------------

def _extract_embedding(face_arr_rgb: np.ndarray, model_name: str, expected_dim: int) -> np.ndarray:
    from deepface import DeepFace  # type: ignore
    rep = DeepFace.represent(
        img_path=face_arr_rgb,
        model_name=model_name,
        **DEEPFACE_KWARGS,
    )
    if not rep:
        raise RuntimeError(f"DeepFace returned empty for model {model_name!r}")
    emb = np.asarray(rep[0]["embedding"], dtype=np.float64)
    if emb.shape != (expected_dim,):
        raise ValueError(
            f"{model_name} embedding shape {emb.shape} != expected ({expected_dim},). "
            f"This breaks the SVR contract; check DeepFace version pin."
        )
    return emb


def _extract_vgg(face_arr_rgb: np.ndarray) -> np.ndarray:
    return _extract_embedding(face_arr_rgb, "VGG-Face", VGG_EMBED_DIM)


def _extract_facenet(face_arr_rgb: np.ndarray) -> np.ndarray:
    return _extract_embedding(face_arr_rgb, "Facenet512", FACENET_EMBED_DIM)


# --------------------------------------------------------------------------
# 5. Main pipeline class
# --------------------------------------------------------------------------

class FaceToBMIPipeline:
    """
    End-to-end Face-to-BMI inference.

    Parameters
    ----------
    backbone : {"vgg", "facenet", "ensemble"}, default "ensemble"
        Regressor configuration.
            "vgg"      — VGG-Face 4096D + svr_vgg_tuned.joblib only
            "facenet"  — Facenet512 512D + svr_facenet_tuned.joblib only
            "ensemble" — average of the two predictions (current best, r=0.6469)
    auto_detect_face : bool, default True
        If True, run MTCNN to crop a face from raw photos before standardization.
        Set False if the input is already a pre-cropped face (e.g., images
        from data/processed/faces_standardized/).
    vgg_model_path, facenet_model_path : Path, optional
        Override default joblib paths.

    Examples
    --------
    >>> pipe = FaceToBMIPipeline()
    >>> pipe.predict("photo.jpg")
    {'bmi': 25.4, 'confidence': 0.81, 'backbone': 'ensemble'}

    >>> # Use only the VGG backbone, skip MTCNN (input is already a face crop)
    >>> pipe = FaceToBMIPipeline(backbone="vgg", auto_detect_face=False)
    """

    VALID_BACKBONES = {"vgg", "facenet", "ensemble"}

    def __init__(
        self,
        backbone: str = "ensemble",
        auto_detect_face: bool = True,
        vgg_model_path: Optional[Path] = None,
        facenet_model_path: Optional[Path] = None,
    ):
        if backbone not in self.VALID_BACKBONES:
            raise ValueError(f"backbone must be one of {self.VALID_BACKBONES}, got {backbone!r}")
        self.backbone = backbone
        self.auto_detect_face = auto_detect_face

        self.vgg_model = None
        self.facenet_model = None

        if backbone in {"vgg", "ensemble"}:
            p = Path(vgg_model_path or DEFAULT_VGG_MODEL_PATH)
            if not p.exists():
                raise FileNotFoundError(f"VGG SVR model not found at {p}")
            self.vgg_model = joblib.load(p)
            logger.info("Loaded VGG SVR from %s", p)

        if backbone in {"facenet", "ensemble"}:
            p = Path(facenet_model_path or DEFAULT_FACENET_MODEL_PATH)
            if not p.exists():
                raise FileNotFoundError(f"Facenet SVR model not found at {p}")
            self.facenet_model = joblib.load(p)
            logger.info("Loaded Facenet SVR from %s", p)

    # ---- public API ------------------------------------------------------

    def predict(self, image: ImageInput, *, return_diagnostics: bool = False) -> dict:
        """Single-image inference. Returns dict with bmi/confidence/backbone."""
        face_arr = self._prepare_face(image)
        diag: dict = {"face_shape": face_arr.shape}
        vgg_pred = facenet_pred = None

        if self.vgg_model is not None:
            emb = _extract_vgg(face_arr)
            vgg_pred = float(self.vgg_model.predict(emb.reshape(1, -1))[0])
            diag["vgg_pred"] = vgg_pred
            diag["vgg_emb_norm"] = float(np.linalg.norm(emb))

        if self.facenet_model is not None:
            emb = _extract_facenet(face_arr)
            facenet_pred = float(self.facenet_model.predict(emb.reshape(1, -1))[0])
            diag["facenet_pred"] = facenet_pred
            diag["facenet_emb_norm"] = float(np.linalg.norm(emb))

        bmi, confidence = self._combine(vgg_pred, facenet_pred)
        bmi_clipped = float(np.clip(bmi, BMI_CLIP_MIN, BMI_CLIP_MAX))
        if bmi_clipped != bmi:
            diag["bmi_clipped"] = True
            diag["bmi_raw"] = bmi

        return PredictionResult(
            bmi=bmi_clipped,
            confidence=confidence,
            backbone=self.backbone,
            diagnostics=diag,
        ).to_dict(include_diagnostics=return_diagnostics)

    def predict_batch(self, images: list, *, return_diagnostics: bool = False) -> list:
        """Convenience wrapper. Calls predict() per image; no batching optimisation yet."""
        return [self.predict(img, return_diagnostics=return_diagnostics) for img in images]

    def predict_from_features(self, features: np.ndarray, *, backbone: str) -> np.ndarray:
        """
        Skip image preprocessing + extraction; predict directly from precomputed features.

        Use this for integration tests against Ryan's / Beste's saved .npy files.
        Returns a 1D ndarray of BMI predictions.
        """
        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D (N, D), got shape {features.shape}")

        if backbone == "vgg":
            if self.vgg_model is None:
                raise RuntimeError("VGG model not loaded; init with backbone='vgg' or 'ensemble'")
            if features.shape[1] != VGG_EMBED_DIM:
                raise ValueError(f"VGG expects {VGG_EMBED_DIM} dims, got {features.shape[1]}")
            return self.vgg_model.predict(features)

        if backbone == "facenet":
            if self.facenet_model is None:
                raise RuntimeError("Facenet model not loaded; init with backbone='facenet' or 'ensemble'")
            if features.shape[1] != FACENET_EMBED_DIM:
                raise ValueError(f"Facenet expects {FACENET_EMBED_DIM} dims, got {features.shape[1]}")
            return self.facenet_model.predict(features)

        raise ValueError(f"backbone must be 'vgg' or 'facenet', got {backbone!r}")

    # ---- internals ------------------------------------------------------

    def _prepare_face(self, image: ImageInput) -> np.ndarray:
        pil = _to_pil_rgb(image)
        if self.auto_detect_face:
            pil = _detect_and_crop_face(pil)
        return _standardize_face(pil)

    def _combine(self, vgg_pred: Optional[float], facenet_pred: Optional[float]) -> Tuple[float, float]:
        if self.backbone == "vgg":
            return float(vgg_pred), self._confidence_single(vgg_pred)
        if self.backbone == "facenet":
            return float(facenet_pred), self._confidence_single(facenet_pred)
        # ensemble
        avg = 0.5 * (vgg_pred + facenet_pred)
        # Confidence ≈ agreement between the two backbones.
        # 0 BMI disagreement → 1.0 confidence; ≥10 BMI disagreement → 0.0
        disagreement = abs(vgg_pred - facenet_pred)
        confidence = float(np.clip(1.0 - disagreement / 10.0, 0.0, 1.0))
        return float(avg), confidence

    @staticmethod
    def _confidence_single(pred: Optional[float]) -> float:
        # Heuristic placeholder — single-backbone confidence has no model-internal signal.
        if pred is None:
            return 0.0
        if 18 <= pred <= 45:
            return 0.7
        if 15 <= pred <= 55:
            return 0.5
        return 0.3


# --------------------------------------------------------------------------
# 6. Integration sanity test  (run: `python pipeline_v1.py`)
# --------------------------------------------------------------------------

def _integration_test() -> int:
    """
    Validate the regressor side of the pipeline against Beste's saved test features.
    Skips raw-image extraction entirely (that's tested separately by Streamlit).

    Returns 0 on success, non-zero on failure.
    """
    try:
        from scipy.stats import pearsonr
    except ImportError:
        print("scipy not installed — install requirements.txt first")
        return 2

    feats_dir = REPO_ROOT / "models" / "beste" / "features"
    if not feats_dir.exists():
        print(f"Feature dir missing: {feats_dir}")
        print("Pull from team Google Drive before running this test.")
        return 2

    print("=" * 64)
    print("Face-to-BMI Pipeline — Integration Sanity Test")
    print("=" * 64)

    # ---- 1. VGG backbone -------------------------------------------------
    print("\n[1/3] VGG-Face SVR")
    pipe_v = FaceToBMIPipeline(backbone="vgg", auto_detect_face=False)
    X_v = np.load(feats_dir / "X_test_VGG_Face.npy")
    y_v = np.load(feats_dir / "y_test_VGG_Face.npy")
    pred_v = pipe_v.predict_from_features(X_v, backbone="vgg")
    r_v, _ = pearsonr(pred_v, y_v)
    print(f"      X_test {X_v.shape}, y_test {y_v.shape}")
    print(f"      Pearson r = {r_v:.4f}    [Beste reported 0.6261]")

    # ---- 2. Facenet backbone --------------------------------------------
    print("\n[2/3] Facenet512 SVR")
    pipe_f = FaceToBMIPipeline(backbone="facenet", auto_detect_face=False)
    X_f = np.load(feats_dir / "X_test_Facenet512.npy")
    y_f = np.load(feats_dir / "y_test_Facenet512.npy")
    pred_f = pipe_f.predict_from_features(X_f, backbone="facenet")
    r_f, _ = pearsonr(pred_f, y_f)
    print(f"      X_test {X_f.shape}, y_test {y_f.shape}")
    print(f"      Pearson r = {r_f:.4f}")

    # ---- 3. Ensemble -----------------------------------------------------
    print("\n[3/3] Ensemble (average of VGG + Facenet)")
    if X_v.shape[0] != X_f.shape[0]:
        print(f"      SKIP — row counts differ ({X_v.shape[0]} vs {X_f.shape[0]}).")
        print("      Beste's two .npy splits are not row-aligned; ensemble check needs a")
        print("      shared index. Add image_id columns to align before averaging.")
    else:
        pipe_e = FaceToBMIPipeline(backbone="ensemble", auto_detect_face=False)
        pv = pipe_e.predict_from_features(X_v, backbone="vgg")
        pf = pipe_e.predict_from_features(X_f, backbone="facenet")
        pred_e = 0.5 * (pv + pf)
        r_e, _ = pearsonr(pred_e, y_v)
        print(f"      Pearson r = {r_e:.4f}    [Beste reported 0.6469]")

    print("\nDone. If r values match Beste's reported numbers, the regressor")
    print("integration is sound and Streamlit can wire to FaceToBMIPipeline.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(_integration_test())
