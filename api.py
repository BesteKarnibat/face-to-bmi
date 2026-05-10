"""
api.py — Face-to-BMI Backend API
=================================
FastAPI server that wraps FaceToBMIPipeline and exposes BMI prediction
over HTTP. Accepts images either as a multipart file upload or as a
base64-encoded string inside a JSON body.

Endpoints
---------
GET  /              → API info
GET  /health        → health check (model loaded?)
POST /predict       → multipart file upload  (field name: "file")
POST /predict/json  → base64 JSON body       (field name: "image_b64")

Run
---
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Interactive docs → http://localhost:8000/docs
"""

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from pipeline_v1 import FaceToBMIPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger("face-to-bmi-api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_VERSION   = "1.0.0"
MAX_FILE_SIZE = 10 * 1024 * 1024          # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# BMI category thresholds (WHO)
BMI_CATEGORIES = [
    (18.5, "Underweight"),
    (25.0, "Normal weight"),
    (30.0, "Overweight"),
    (float("inf"), "Obese"),
]


def bmi_category(bmi: float) -> str:
    for threshold, label in BMI_CATEGORIES:
        if bmi < threshold:
            return label
    return "Obese"


# ---------------------------------------------------------------------------
# Application state  (pipeline loaded once at startup)
# ---------------------------------------------------------------------------
class AppState:
    pipeline: FaceToBMIPipeline | None = None
    startup_time: float = 0.0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model pipeline when the server starts; release on shutdown."""
    log.info("Loading FaceToBMIPipeline …")
    t0 = time.time()
    state.pipeline = FaceToBMIPipeline()        # backbone='ensemble' by default
    state.startup_time = time.time()
    log.info(f"Pipeline ready in {time.time() - t0:.2f}s")
    yield
    log.info("Shutting down — releasing pipeline")
    state.pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Face-to-BMI API",
    description=(
        "Predict Body Mass Index from a facial image using a VGG-Face / "
        "Facenet512 feature extractor + SVR ensemble. "
        "Images can be uploaded as multipart/form-data **or** sent as a "
        "base64-encoded string in a JSON body."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Successful BMI prediction."""
    bmi:          float  = Field(...,  description="Predicted BMI (kg/m²)", examples=[24.7])
    confidence:   float  = Field(...,  description="Model confidence score [0, 1]", examples=[0.82])
    bmi_category: str    = Field(...,  description="WHO BMI category", examples=["Normal weight"])
    backbone:     str    = Field(...,  description="Backbone used for prediction", examples=["ensemble"])
    latency_ms:   float  = Field(...,  description="Server-side inference time (ms)", examples=[120.5])


class Base64Request(BaseModel):
    """JSON body for base64-encoded image upload."""
    image_b64: str = Field(
        ...,
        description=(
            "Base64-encoded image string. Standard or data-URI prefix accepted. "
            "E.g.  'data:image/jpeg;base64,/9j/4AAQ…'  or just  '/9j/4AAQ…'"
        ),
    )
    backbone: str = Field(
        default="ensemble",
        description="Which backbone to use: 'vgg', 'facenet', or 'ensemble'",
    )

    @field_validator("backbone")
    @classmethod
    def check_backbone(cls, v: str) -> str:
        allowed = {"vgg", "facenet", "ensemble"}
        if v not in allowed:
            raise ValueError(f"backbone must be one of {allowed}")
        return v


class ErrorResponse(BaseModel):
    """Returned on any error."""
    error:   str = Field(..., description="Short error code")
    detail:  str = Field(..., description="Human-readable description")


class HealthResponse(BaseModel):
    status:       str   = Field(..., examples=["ok"])
    model_loaded: bool  = Field(..., examples=[True])
    uptime_s:     float = Field(..., description="Seconds since startup")
    version:      str   = Field(..., examples=[API_VERSION])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pipeline_or_503() -> FaceToBMIPipeline:
    """Return the loaded pipeline or raise 503 if not ready."""
    if state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded yet. Try again in a moment.",
        )
    return state.pipeline


def _bytes_to_pil(data: bytes, source: str = "upload") -> Image.Image:
    """Convert raw image bytes → PIL Image, with size guard."""
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large ({len(data)/1e6:.1f} MB). Limit is 10 MB.",
        )
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image from {source}: {exc}",
        )
    return img


def _run_prediction(
    pipeline: FaceToBMIPipeline,
    image: Image.Image,
    backbone: str = "ensemble",
) -> PredictionResponse:
    """
    Call the pipeline and build the response.
    Catches prediction errors and converts them to clean HTTP errors.
    """
    t0 = time.perf_counter()
    try:
        result: dict = pipeline.predict(image)
    except Exception as exc:
        log.exception("Pipeline prediction failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {exc}",
        )
    latency = (time.perf_counter() - t0) * 1000

    bmi = float(result["bmi"])
    return PredictionResponse(
        bmi          = round(bmi, 2),
        confidence   = round(float(result.get("confidence", 0.0)), 4),
        bmi_category = bmi_category(bmi),
        backbone     = result.get("backbone", backbone),
        latency_ms   = round(latency, 1),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="API info",
    tags=["General"],
)
def root():
    """Returns basic API metadata and available endpoints."""
    return {
        "name":    "Face-to-BMI API",
        "version": API_VERSION,
        "endpoints": {
            "GET  /health":        "Health check",
            "POST /predict":       "Predict BMI from multipart image upload",
            "POST /predict/json":  "Predict BMI from base64 JSON body",
            "GET  /docs":          "Interactive Swagger UI",
            "GET  /redoc":         "ReDoc documentation",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["General"],
)
def health():
    """Returns server health and whether the model pipeline is loaded."""
    return HealthResponse(
        status       = "ok" if state.pipeline is not None else "degraded",
        model_loaded = state.pipeline is not None,
        uptime_s     = round(time.time() - state.startup_time, 1),
        version      = API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Image too large (> 10 MB)"},
        422: {"model": ErrorResponse, "description": "Invalid image or prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
    summary="Predict BMI — multipart file upload",
    tags=["Prediction"],
)
async def predict_upload(
    file:     Annotated[UploadFile, File(description="Image file (JPEG / PNG / BMP / WebP)")],
    backbone: str = "ensemble",
):
    """
    Upload an image as **multipart/form-data** and get a BMI prediction.

    **Request**
    - `file`     — image file (JPEG, PNG, BMP, WebP), max 10 MB
    - `backbone` — optional query param: `vgg` | `facenet` | `ensemble` (default)

    **Response**
    ```json
    {
      "bmi": 24.7,
      "confidence": 0.82,
      "bmi_category": "Normal weight",
      "backbone": "ensemble",
      "latency_ms": 118.3
    }
    ```

    **Example (curl)**
    ```bash
    curl -X POST http://localhost:8000/predict \\
         -F "file=@photo.jpg"
    ```
    """
    # Validate content type
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}"
            ),
        )

    # Validate backbone query param
    if backbone not in {"vgg", "facenet", "ensemble"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="backbone must be 'vgg', 'facenet', or 'ensemble'",
        )

    pipeline = _pipeline_or_503()
    raw      = await file.read()
    image    = _bytes_to_pil(raw, source=file.filename or "upload")

    log.info(
        f"[/predict] file={file.filename!r}  "
        f"size={len(raw)/1024:.1f}KB  backbone={backbone}"
    )
    return _run_prediction(pipeline, image, backbone)


@app.post(
    "/predict/json",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed base64 string"},
        413: {"model": ErrorResponse, "description": "Decoded image too large"},
        422: {"model": ErrorResponse, "description": "Invalid image or prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
    summary="Predict BMI — base64 JSON body",
    tags=["Prediction"],
)
def predict_json(body: Base64Request):
    """
    Send an image as a **base64-encoded string** inside a JSON body.

    Accepts both plain base64 and the data-URI format:
    - `"/9j/4AAQSkZJRg…"`
    - `"data:image/jpeg;base64,/9j/4AAQSkZJRg…"`

    **Request body**
    ```json
    {
      "image_b64": "/9j/4AAQSkZJRg...",
      "backbone":  "ensemble"
    }
    ```

    **Response**
    ```json
    {
      "bmi": 24.7,
      "confidence": 0.82,
      "bmi_category": "Normal weight",
      "backbone": "ensemble",
      "latency_ms": 118.3
    }
    ```

    **Example (curl)**
    ```bash
    B64=$(base64 -i photo.jpg)
    curl -X POST http://localhost:8000/predict/json \\
         -H "Content-Type: application/json" \\
         -d "{\"image_b64\": \"$B64\", \"backbone\": \"ensemble\"}"
    ```
    """
    pipeline = _pipeline_or_503()

    # Strip data-URI prefix if present  ("data:image/jpeg;base64,…")
    b64_str = body.image_b64
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode base64 string. Make sure it is valid base64.",
        )

    image = _bytes_to_pil(raw, source="base64 payload")
    log.info(
        f"[/predict/json] size={len(raw)/1024:.1f}KB  backbone={body.backbone}"
    )
    return _run_prediction(pipeline, image, body.backbone)


# ---------------------------------------------------------------------------
# Global exception handler — always returns JSON, never HTML
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.exception(f"Unhandled error on {request.url}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # set False in production
        log_level="info",
    )
