import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Face to BMI - Face Crop Display",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stColumns {
        gap: 2rem;
    }
    .crop-container {
        border: 2px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f0f2f6;
    }
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
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FACE DETECTION & CROPPING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_face_cascade():
    """Load pre-trained Haar Cascade for face detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def detect_and_crop_faces(image_path_or_array):
    """
    Detect faces in an image and return cropped faces with bounding boxes.
    
    Args:
        image_path_or_array: Path to image file or numpy array
    
    Returns:
        tuple: (original_image, face_crops, face_boxes, face_count)
    """
    # Load image
    if isinstance(image_path_or_array, str):
        original_img = cv2.imread(image_path_or_array)
    else:
        original_img = image_path_or_array
    
    if original_img is None:
        return None, [], [], 0
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Load cascade classifier
    face_cascade = load_face_cascade()
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(500, 500)
    )
    
    face_crops = []
    face_boxes = []
    
    # Extract and crop faces with padding
    for (x, y, w, h) in faces:
        # Add padding to crop (10% on each side)
        padding = int(0.1 * min(w, h))
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(original_img.shape[1], x + w + padding)
        y_end = min(original_img.shape[0], y + h + padding)
        
        # Crop face region
        face_crop = original_img[y_start:y_end, x_start:x_end]
        face_crops.append(face_crop)
        face_boxes.append((x_start, y_start, x_end, y_end))
    
    return original_img, face_crops, face_boxes, len(faces)

def draw_face_boxes(image, face_boxes):
    """Draw bounding boxes around detected faces."""
    image_with_boxes = image.copy()
    for (x_start, y_start, x_end, y_end) in face_boxes:
        cv2.rectangle(image_with_boxes, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    return image_with_boxes

def convert_bgr_to_rgb(image):
    """Convert OpenCV BGR image to RGB for display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("📸 Face Detection & Crop Display")
    st.markdown("**Upload an image → Instantly see detected faces cropped**")
    
    # Info box
    st.markdown("""
        <div class="info-box">
        <strong>How it works:</strong> Upload an image, and the app will automatically detect faces using 
        computer vision, crop them, and display both the original image with face boxes and individual face crops.
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("---")
    
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence (Neighbors)",
        min_value=3,
        max_value=10,
        value=5,
        help="Higher = stricter detection (fewer false positives)"
    )
    
    min_face_size = st.sidebar.slider(
        "Minimum Face Size (pixels)",
        min_value=10,
        max_value=100,
        value=30,
        help="Faces smaller than this will be ignored"
    )
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        help="Supported formats: JPG, PNG, BMP, GIF"
    )
    
    # Process uploaded image
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Load and detect faces
        with st.spinner("🔍 Detecting faces..."):
            original_img, face_crops, face_boxes, face_count = detect_and_crop_faces(temp_path)
        
        # Display results
        if original_img is not None:
            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-box">
                    <h3 style="margin: 0; color: #000000;">Faces Detected</h3>
                    <p style="font-size: 32px; margin: 0.5rem 0; color: #000000;"><strong>{face_count}</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if face_crops:
                    st.markdown(f"""
                        <div class="metric-box">
                        <h3 style="margin: 0; color: #000000;">Crops Ready</h3>
                        <p style="font-size: 32px; margin: 0.5rem 0; color: #000000;"><strong>✓</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                img_size = os.path.getsize(temp_path)
                st.markdown(f"""
                    <div class="metric-box">
                    <h3 style="margin: 0; color: #000000;">Image Size</h3>
                    <p style="font-size: 14px; margin: 0.5rem 0; color: #000000;"><strong>{img_size / 1024:.1f} KB</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display original with bounding boxes
            if face_count > 0:
                col_orig, col_annotated = st.columns(2)
                
                with col_orig:
                    st.subheader("📷 Original Image")
                    st.image(convert_bgr_to_rgb(original_img), use_column_width=True)
                
                with col_annotated:
                    st.subheader("🎯 Detected Faces (with boxes)")
                    image_with_boxes = draw_face_boxes(original_img, face_boxes)
                    st.image(convert_bgr_to_rgb(image_with_boxes), use_column_width=True)
            else:
                st.warning("⚠️ No faces detected in the image. Try uploading a clearer image.")
            
            # Display individual face crops
            if face_crops:
                st.markdown("---")
                st.subheader(f"✂️ Cropped Faces ({len(face_crops)} detected)")
                
                # Create columns for face crops
                crops_per_row = 3
                for i in range(0, len(face_crops), crops_per_row):
                    cols = st.columns(crops_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(face_crops):
                            with col:
                                face_pil = Image.fromarray(
                                    convert_bgr_to_rgb(face_crops[i + j])
                                )
                                st.image(face_pil, use_column_width=True)
                                
                                # Display crop info
                                crop = face_crops[i + j]
                                st.caption(
                                    f"Face #{i + j + 1}\n"
                                    f"{crop.shape[1]}×{crop.shape[0]}px"
                                )
                
                # Download option
                st.markdown("---")
                st.subheader("💾 Download Cropped Faces")
                
                if st.button("📥 Generate ZIP with all crops"):
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for idx, face_crop in enumerate(face_crops):
                            face_pil = Image.fromarray(
                                convert_bgr_to_rgb(face_crop)
                            )
                            img_buffer = io.BytesIO()
                            face_pil.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            zip_file.writestr(f"face_{idx + 1:03d}.png", img_buffer.getvalue())
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="cropped_faces.zip",
                        mime="application/zip"
                    )
        
        # Cleanup
        os.unlink(temp_path)
    
    else:
        # Empty state
        st.markdown("""
            <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to Face Crop Display</h2>
            <p style="font-size: 16px; color: #666;">
            Upload an image using the sidebar to get started.
            </p>
            <p style="color: #999; font-size: 14px;">
            Supported formats: JPG, PNG, BMP, GIF
            </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
