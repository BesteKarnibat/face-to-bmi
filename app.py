import streamlit as st
from PIL import Image
import io

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face-to-BMI Predictor",
    page_icon="🧠",
    layout="centered",
)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.title("🧠 Face-to-BMI Predictor")
st.markdown(
    "Upload a **frontal face image** and the model will predict an estimated BMI. "
    "Accepted formats: `.jpg`, `.jpeg`, `.png`"
)
st.divider()

# ─── Upload Section ──────────────────────────────────────────────────────────────
st.subheader("📤 Upload Face Image")
uploaded_file = st.file_uploader(
    label="Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, frontal face photo for best results.",
)

# ─── Image Preview Section ───────────────────────────────────────────────────────
if uploaded_file is not None:
    st.divider()
    st.subheader("🖼️ Image Preview")

    # Read and display the image
    image = Image.open(io.BytesIO(uploaded_file.read()))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

    with col2:
        st.markdown("**File Details**")
        st.write(f"- **Name:** {uploaded_file.name}")
        st.write(f"- **Format:** {image.format if image.format else uploaded_file.type}")
        st.write(f"- **Size:** {image.size[0]} × {image.size[1]} px")
        st.write(f"- **Mode:** {image.mode}")
        file_size_kb = uploaded_file.size / 1024
        st.write(f"- **File size:** {file_size_kb:.1f} KB")

    st.divider()

    # ─── Placeholder: Prediction Section (W2+) ──────────────────────────────────
    st.subheader("📊 BMI Prediction")
    st.info(
        "🚧 Model inference coming in Week 2. "
        "The uploaded image has been received and is ready for preprocessing.",
        icon="🔬",
    )

    # Placeholder predict button (wired up but no model yet)
    if st.button("🔍 Predict BMI", type="primary", disabled=True):
        pass  # Will be replaced with model inference in W2

else:
    # Empty state guidance
    st.info("👆 Upload a face image above to get started.", icon="📁")

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Face-to-BMI · Week 1 · Computer Vision · UChicago Applied Data Science")
