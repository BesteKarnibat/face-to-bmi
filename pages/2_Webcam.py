"""
webcam.py — Live Webcam BMI Prediction Page
=============================================
Streamlit page that:
  1. Shows a live webcam feed via streamlit-webrtc (works locally + deployed)
  2. Lets the user press "Capture & Predict" to grab the current frame
  3. Detects faces with OpenCV, sends each crop to the FastAPI backend
  4. Displays BMI results with the WHO reference bar

Run standalone:
    uvicorn api:app --port 8001 --reload   # terminal 1
    streamlit run webcam.py                # terminal 2

Or as part of multipage app:
    Move this file into pages/2_Webcam.py — Streamlit picks it up automatically.

Dependencies:
    pip install streamlit-webrtc aiortc av opencv-python-headless
"""

import io
import os
import threading
import time

import av
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ============================================================================
# CONFIG
# ============================================================================
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
PREDICT_URL  = f"{API_BASE_URL}/predict"
HEALTH_URL   = f"{API_BASE_URL}/health"

# Works both locally and on deployed servers.
# Twilio TURN credentials can be injected via env vars for production.
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

BMI_RANGES = [
    (0,    18.5, "Underweight",  "#2196F3"),
    (18.5, 25.0, "Normal weight","#4CAF50"),
    (25.0, 30.0, "Overweight",   "#FF9800"),
    (30.0, 999,  "Obese",        "#F44336"),
]

def bmi_color(bmi: float) -> str:
    for lo, hi, _, color in BMI_RANGES:
        if lo <= bmi < hi:
            return color
    return "#F44336"

def bmi_label(bmi: float) -> str:
    for lo, hi, label, _ in BMI_RANGES:
        if lo <= bmi < hi:
            return label
    return "Obese"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Webcam BMI — Face to BMI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.bmi-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
}
.api-status-ok   { color: #4CAF50; font-weight: bold; }
.api-status-fail { color: #F44336; font-weight: bold; }
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1976d2;
    margin-bottom: 1rem;
    color: #000;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API HELPERS
# ============================================================================
def check_api_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def predict_bmi(image_bytes: bytes) -> dict | None:
    try:
        r = requests.post(
            PREDICT_URL,
            files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach API at {API_BASE_URL}. "
            "Run: `uvicorn api:app --port 8001 --reload`"
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None

# ============================================================================
# FACE DETECTION
# ============================================================================
@st.cache_resource
def load_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def detect_faces(frame_bgr: np.ndarray, min_neighbors: int = 5, min_size: int = 60):
    """Returns (crops_bgr, boxes) from a BGR frame."""
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = load_cascade()
    faces   = cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    crops, boxes = [], []
    for (x, y, w, h) in faces:
        pad  = int(0.1 * min(w, h))
        x0   = max(0, x - pad)
        y0   = max(0, y - pad)
        x1   = min(frame_bgr.shape[1], x + w + pad)
        y1   = min(frame_bgr.shape[0], y + h + pad)
        crops.append(frame_bgr[y0:y1, x0:x1])
        boxes.append((x0, y0, x1, y1))
    return crops, boxes


def draw_boxes(frame_bgr: np.ndarray, boxes: list) -> np.ndarray:
    out = frame_bgr.copy()
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return out


def bgr_to_bytes(crop_bgr: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ============================================================================
# WebRTC VIDEO PROCESSOR
# Runs in a background thread — just annotates live frames with face boxes.
# Captured frame is stored in session state when user clicks the button.
# ============================================================================
class FaceBoxProcessor(VideoProcessorBase):
    """
    Draws green face-detection boxes on the live stream.
    Stores the latest raw frame so the capture button can grab it.
    """

    def __init__(self):
        self.min_neighbors: int = 5
        self.min_size: int      = 60
        self._lock              = threading.Lock()
        self._latest_frame: np.ndarray | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Store a copy of the raw frame (thread-safe)
        with self._lock:
            self._latest_frame = img.copy()

        # Detect and annotate
        _, boxes = detect_faces(img, self.min_neighbors, self.min_size)
        annotated = draw_boxes(img, boxes)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

# ============================================================================
# BMI RESULT RENDERER
# ============================================================================
def render_bmi_card(result: dict, face_index: int):
    bmi      = result["bmi"]
    conf     = result["confidence"]
    category = result["bmi_category"]
    backbone = result["backbone"]
    latency  = result["latency_ms"]
    color    = bmi_color(bmi)

    st.markdown(f"""
    <div class="bmi-card" style="background:{color}22; border:2px solid {color};">
        <h2 style="color:{color}; margin:0;">BMI {bmi:.1f}</h2>
        <p style="font-size:18px; margin:0.25rem 0; color:#333;">
            <strong>{category}</strong>
        </p>
        <p style="color:#555; margin:0; font-size:13px;">
            Confidence: {conf:.0%} &nbsp;|&nbsp;
            Backbone: {backbone} &nbsp;|&nbsp;
            {latency:.0f} ms
        </p>
    </div>
    """, unsafe_allow_html=True)

    # WHO reference bar
    st.markdown("**WHO BMI reference**")
    cols = st.columns(4)
    for col, (lo, hi, label, c) in zip(cols, BMI_RANGES):
        active = lo <= bmi < hi
        with col:
            st.markdown(
                f"<div style='background:{c}; padding:6px; border-radius:6px; "
                f"text-align:center; opacity:{'1.0' if active else '0.35'};'>"
                f"<span style='color:white; font-size:12px;'>"
                f"{label}{'<br>▲ you' if active else ''}</span></div>",
                unsafe_allow_html=True,
            )

# ============================================================================
# MAIN PAGE
# ============================================================================
def main():
    st.title("🎥 Webcam BMI Prediction")
    st.markdown(
        "Point your webcam at a face → press **Capture & Predict** → get a BMI estimate."
    )

    # ── API status ────────────────────────────────────────────────────────────
    api_ok = check_api_health()
    if api_ok:
        st.markdown('<span class="api-status-ok">● API online</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="api-status-fail">● API offline</span> — '
            "start with: `uvicorn api:app --port 8001 --reload`",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong>
    The live feed shows green boxes around detected faces in real time.
    Press <strong>Capture &amp; Predict</strong> to freeze the current frame,
    crop each face, and send it to the backend for BMI prediction.
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Detection settings")
    min_neighbors = st.sidebar.slider(
        "Detection confidence (neighbors)", 3, 10, 5,
        help="Higher = stricter, fewer false positives",
    )
    min_size = st.sidebar.slider(
        "Minimum face size (px)", 30, 150, 60,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** For best results, ensure good lighting and "
        "face the camera directly."
    )
    st.sidebar.markdown(
        f"**API endpoint:** `{API_BASE_URL}`"
    )

    # ── WebRTC stream ─────────────────────────────────────────────────────────
    st.subheader("📹 Live Feed")

    ctx = webrtc_streamer(
        key="webcam-bmi",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FaceBoxProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )

    # Push detection settings into the processor whenever they change
    if ctx.video_processor:
        ctx.video_processor.min_neighbors = min_neighbors
        ctx.video_processor.min_size      = min_size

    # ── Capture button ────────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        capture = st.button(
            "📸 Capture & Predict",
            disabled=not (ctx.state.playing and api_ok),
            use_container_width=True,
        )
    with col_hint:
        if not ctx.state.playing:
            st.info("▶ Start the webcam above first.")
        elif not api_ok:
            st.warning("API is offline — cannot predict.")

    # ── Process captured frame ────────────────────────────────────────────────
    if capture and ctx.video_processor:
        frame = ctx.video_processor.get_latest_frame()

        if frame is None:
            st.warning("No frame captured yet — wait a moment and try again.")
            return

        st.markdown("---")
        st.subheader("📸 Captured Frame & Results")

        # Detect faces in the captured frame
        crops, boxes = detect_faces(frame, min_neighbors, min_size)
        annotated    = draw_boxes(frame, boxes)

        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original capture**")
            st.image(bgr_to_rgb(frame), use_container_width=True)
        with col_ann:
            st.markdown("**Detected faces**")
            st.image(bgr_to_rgb(annotated), use_container_width=True)

        if not crops:
            st.warning(
                "No faces detected in the captured frame. "
                "Try adjusting the detection settings in the sidebar."
            )
            return

        st.markdown("---")
        st.subheader(f"🔬 BMI Predictions ({len(crops)} face{'s' if len(crops) > 1 else ''} detected)")

        for i, crop in enumerate(crops):
            col_face, col_result = st.columns([1, 2])

            with col_face:
                st.image(
                    bgr_to_rgb(crop),
                    caption=f"Face #{i+1}  ({crop.shape[1]}×{crop.shape[0]}px)",
                    use_container_width=True,
                )

            with col_result:
                with st.spinner(f"Predicting BMI for face #{i+1}…"):
                    result = predict_bmi(bgr_to_bytes(crop))
                if result:
                    render_bmi_card(result, i)

            st.markdown("---")

    # ── Fallback: st.camera_input if WebRTC is unavailable ───────────────────
    with st.expander("📷 Alternative: Snapshot mode (no WebRTC required)"):
        st.markdown(
            "If the live feed above doesn't work (e.g. browser blocks WebRTC), "
            "use this snapshot camera instead."
        )
        snapshot = st.camera_input("Take a snapshot")

        if snapshot and api_ok:
            img_bytes = snapshot.getvalue()
            img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame_rgb = np.array(img_pil)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            crops, boxes = detect_faces(frame_bgr, min_neighbors, min_size)
            annotated    = draw_boxes(frame_bgr, boxes)

            st.image(bgr_to_rgb(annotated),
                     caption="Detected faces", use_container_width=True)

            if not crops:
                st.warning("No faces detected in snapshot.")
            else:
                st.subheader(f"BMI Predictions ({len(crops)} face detected)")
                for i, crop in enumerate(crops):
                    col_f, col_r = st.columns([1, 2])
                    with col_f:
                        st.image(bgr_to_rgb(crop),
                                 caption=f"Face #{i+1}", use_container_width=True)
                    with col_r:
                        with st.spinner(f"Predicting…"):
                            result = predict_bmi(bgr_to_bytes(crop))
                        if result:
                            render_bmi_card(result, i)


if __name__ == "__main__":
    main()
