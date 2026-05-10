"""
app.py — Face-to-BMI Streamlit Frontend
========================================
Keeps the existing face detection / crop UI and adds BMI prediction
by calling the running api.py FastAPI backend.

Run order:
  1. uvicorn api:app --host 0.0.0.0 --port 8001 --reload
  2. streamlit run app.py
"""

import io
import os
import tempfile
import zipfile

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ============================================================================
# CONFIG
# ============================================================================
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
PREDICT_URL  = f"{API_BASE_URL}/predict"
HEALTH_URL   = f"{API_BASE_URL}/health"

BMI_RANGES = [
    (0,    18.5, "Underweight",   "#2196F3"),
    (18.5, 25.0, "Normal weight", "#4CAF50"),
    (25.0, 30.0, "Overweight",    "#FF9800"),
    (30.0, 999,  "Obese",         "#F44336"),
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
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Face to BMI",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { padding: 2rem; }
.stColumns { gap: 2rem; }
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1976d2;
    margin-bottom: 1rem;
    color: #000000;
}
.metric-box {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}
.bmi-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
}
.api-status-ok   { color: #4CAF50; font-weight: bold; }
.api-status-fail { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API HELPERS
# ============================================================================
def check_api_health() -> dict | None:
    """Returns health dict or None if the API is unreachable."""
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.ConnectionError:
        pass
    return None


def predict_bmi(image_bytes: bytes, backbone: str = "ensemble") -> dict | None:
    """
    POST image bytes to /predict and return the JSON response dict.
    Returns None on any error and surfaces the message via st.error.
    """
    try:
        r = requests.post(
            PREDICT_URL,
            files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            params={"backbone": backbone},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        # Surface API error messages directly
        detail = r.json().get("detail", r.text)
        st.error(f"API error {r.status_code}: {detail}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach the API at {API_BASE_URL}. "
            "Make sure `uvicorn api:app --port 8001` is running."
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error calling API: {exc}")
        return None


# ============================================================================
# FACE DETECTION (unchanged from original)
# ============================================================================
@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_and_crop_faces(image_path, min_neighbors: int = 5, min_size: int = 30):
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, [], [], 0

    gray         = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    face_cascade = load_face_cascade()
    faces        = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        maxSize=(500, 500),
    )

    face_crops, face_boxes = [], []
    for (x, y, w, h) in faces:
        padding = int(0.1 * min(w, h))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(original_img.shape[1], x + w + padding)
        y1 = min(original_img.shape[0], y + h + padding)
        face_crops.append(original_img[y0:y1, x0:x1])
        face_boxes.append((x0, y0, x1, y1))

    return original_img, face_crops, face_boxes, len(faces)


def draw_face_boxes(image, face_boxes):
    out = image.copy()
    for (x0, y0, x1, y1) in face_boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return out


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def face_to_bytes(face_bgr) -> bytes:
    """Convert an OpenCV BGR crop to JPEG bytes for the API."""
    face_rgb = bgr_to_rgb(face_bgr)
    pil      = Image.fromarray(face_rgb)
    buf      = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_bmi_card(result: dict, face_index: int):
    bmi        = result["bmi"]
    confidence = result["confidence"]
    category   = result["bmi_category"]
    backbone   = result["backbone"]
    latency    = result["latency_ms"]
    color      = bmi_color(bmi)

    st.markdown(f"""
    <div class="bmi-card" style="background-color: {color}22; border: 2px solid {color};">
        <h2 style="color: {color}; margin: 0;">BMI {bmi:.1f}</h2>
        <p style="font-size: 18px; margin: 0.25rem 0; color: #333;">
            <strong>{category}</strong>
        </p>
        <p style="color: #555; margin: 0; font-size: 13px;">
            Confidence: {confidence:.0%} &nbsp;|&nbsp;
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
                f"text-align:center; opacity:{'1' if active else '0.35'};'>"
                f"<span style='color:white; font-size:12px;'>{label}<br>"
                f"{'▲ you' if active else ''}</span></div>",
                unsafe_allow_html=True,
            )


# ============================================================================
# MAIN
# ============================================================================
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("📸 Face-to-BMI")
    st.markdown("Upload a photo → detect faces → predict BMI via the API")

    # ── API status badge ─────────────────────────────────────────────────────
    health = check_api_health()
    if health and health.get("model_loaded"):
        st.markdown(
            f'<span class="api-status-ok">● API online</span> '
            f'(uptime {health["uptime_s"]:.0f}s)',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="api-status-fail">● API offline</span> — '
            "start the backend with: `uvicorn api:app --port 8001 --reload`",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong>
    Upload an image → faces are detected with OpenCV →
    each crop is sent to the FastAPI backend →
    the SVR ensemble (VGG-Face + FaceNet512) returns a BMI prediction.
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("---")

    backbone = st.sidebar.selectbox(
        "Prediction backbone",
        options=["ensemble", "vgg", "facenet"],
        index=0,
        help="ensemble averages VGG-Face and FaceNet512 predictions",
    )

    auto_predict = st.sidebar.checkbox(
        "Auto-predict on upload",
        value=True,
        help="Run BMI prediction automatically for every detected face",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Detection settings")
    min_neighbors = st.sidebar.slider(
        "Detection confidence (neighbors)", 3, 10, 5,
        help="Higher = stricter, fewer false positives",
    )
    min_face_size = st.sidebar.slider(
        "Minimum face size (px)", 10, 100, 30,
    )

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported: JPG, PNG, BMP",
    )

    # ── Main flow ─────────────────────────────────────────────────────────────
    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem;">
        <h2>👋 Welcome to Face-to-BMI</h2>
        <p style="font-size:16px; color:#666;">
            Upload an image using the sidebar to get started.
        </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Save to temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        # ── Face detection ───────────────────────────────────────────────────
        with st.spinner("🔍 Detecting faces…"):
            original_img, face_crops, face_boxes, face_count = detect_and_crop_faces(
                temp_path, min_neighbors=min_neighbors, min_size=min_face_size
            )

        if original_img is None:
            st.error("Could not load the image. Please try a different file.")
            return

        # ── Metrics row ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
            <h3 style="margin:0;color:#000;">Faces Detected</h3>
            <p style="font-size:32px;margin:0.5rem 0;color:#000;"><strong>{face_count}</strong></p>
            </div>""", unsafe_allow_html=True)
        with c2:
            status_icon = "✓" if face_count > 0 else "—"
            st.markdown(f"""
            <div class="metric-box">
            <h3 style="margin:0;color:#000;">Crops Ready</h3>
            <p style="font-size:32px;margin:0.5rem 0;color:#000;"><strong>{status_icon}</strong></p>
            </div>""", unsafe_allow_html=True)
        with c3:
            img_kb = os.path.getsize(temp_path) / 1024
            st.markdown(f"""
            <div class="metric-box">
            <h3 style="margin:0;color:#000;">Image Size</h3>
            <p style="font-size:14px;margin:0.5rem 0;color:#000;"><strong>{img_kb:.1f} KB</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        if face_count == 0:
            st.warning("⚠️ No faces detected. Try a clearer image or lower the detection confidence.")
            # Still show the original
            st.subheader("📷 Uploaded Image")
            st.image(bgr_to_rgb(original_img), use_container_width=True)
            return

        # ── Original + annotated side by side ────────────────────────────────
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.subheader("📷 Original Image")
            st.image(bgr_to_rgb(original_img), use_container_width=True)
        with col_ann:
            st.subheader("🎯 Detected Faces")
            img_boxes = draw_face_boxes(original_img, face_boxes)
            st.image(bgr_to_rgb(img_boxes), use_container_width=True)

        st.markdown("---")

        # ── Per-face: crop + BMI prediction ──────────────────────────────────
        st.subheader(f"✂️ Cropped Faces & BMI Predictions ({face_count} detected)")

        for i, face_crop in enumerate(face_crops):
            with st.container():
                col_img, col_bmi = st.columns([1, 2])

                with col_img:
                    face_pil = Image.fromarray(bgr_to_rgb(face_crop))
                    st.image(face_pil, caption=f"Face #{i+1}  "
                             f"({face_crop.shape[1]}×{face_crop.shape[0]}px)",
                             use_container_width=True)

                with col_bmi:
                    if auto_predict:
                        with st.spinner(f"Predicting BMI for face #{i+1}…"):
                            result = predict_bmi(face_to_bytes(face_crop), backbone)
                        if result:
                            render_bmi_card(result, i)
                    else:
                        if st.button(f"🔍 Predict BMI for face #{i+1}", key=f"btn_{i}"):
                            with st.spinner("Calling API…"):
                                result = predict_bmi(face_to_bytes(face_crop), backbone)
                            if result:
                                render_bmi_card(result, i)

            st.markdown("---")

        # ── Download crops ────────────────────────────────────────────────────
        st.subheader("💾 Download Cropped Faces")
        if st.button("📥 Generate ZIP with all crops"):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, face_crop in enumerate(face_crops):
                    img_buf = io.BytesIO()
                    Image.fromarray(bgr_to_rgb(face_crop)).save(img_buf, format="PNG")
                    zf.writestr(f"face_{idx+1:03d}.png", img_buf.getvalue())
            zip_buf.seek(0)
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buf.getvalue(),
                file_name="cropped_faces.zip",
                mime="application/zip",
            )

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    main()
