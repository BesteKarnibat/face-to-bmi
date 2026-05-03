"""
Face-to-BMI Backend API
Flask API for receiving images, detecting faces, extracting features, and predicting BMI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
from datetime import datetime
import logging
from functools import wraps
import traceback

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_VERSION = "1.0.0"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FORMATS = {'image/jpeg', 'image/png', 'image/bmp', 'image/gif'}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_face_cascade():
    """Load pre-trained Haar Cascade for face detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def load_eye_cascade():
    """Load pre-trained Haar Cascade for eye detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    return cv2.CascadeClassifier(cascade_path)

def decode_image_from_base64(base64_string):
    """
    Decode base64 image string to numpy array.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        tuple: (image_array, success, error_message)
    """
    try:
        # Handle data URL format (data:image/jpeg;base64,...)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image_array = cv2.imdecode(
            np.frombuffer(image_data, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            return None, False, "Failed to decode image"
        
        return image_array, True, None
    except Exception as e:
        return None, False, f"Image decoding error: {str(e)}"

def encode_image_to_base64(image_array):
    """
    Encode numpy array image to base64 string.
    
    Args:
        image_array: Numpy array image
    
    Returns:
        str: Base64 encoded image string
    """
    try:
        _, buffer = cv2.imencode('.png', image_array)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error: {str(e)}")
        return None

def detect_faces(image_array):
    """
    Detect faces in an image using Haar Cascade.
    
    Args:
        image_array: Numpy array image (BGR format)
    
    Returns:
        tuple: (face_crops, face_boxes, face_count)
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        face_cascade = load_face_cascade()
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(500, 500)
        )
        
        face_crops = []
        face_boxes = []
        
        for (x, y, w, h) in faces:
            # Add 10% padding
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image_array.shape[1], x + w + padding)
            y_end = min(image_array.shape[0], y + h + padding)
            
            face_crop = image_array[y_start:y_end, x_start:x_end]
            face_crops.append(face_crop)
            face_boxes.append({
                "x_start": int(x_start),
                "y_start": int(y_start),
                "x_end": int(x_end),
                "y_end": int(y_end),
                "width": int(x_end - x_start),
                "height": int(y_end - y_start)
            })
        
        return face_crops, face_boxes, len(faces)
    except Exception as e:
        logger.error(f"Face detection error: {str(e)}")
        return [], [], 0

def extract_facial_features(face_crop):
    """
    Extract facial features from a cropped face image.
    
    Args:
        face_crop: Cropped face image (numpy array)
    
    Returns:
        dict: Extracted features
    """
    try:
        features = {}
        
        # Basic dimensions
        features['height'] = face_crop.shape[0]
        features['width'] = face_crop.shape[1]
        features['aspect_ratio'] = face_crop.shape[1] / face_crop.shape[0]
        
        # Face area
        features['area'] = face_crop.shape[0] * face_crop.shape[1]
        
        # Color statistics (average RGB values)
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        features['avg_r'] = float(np.mean(rgb_face[:, :, 0]))
        features['avg_g'] = float(np.mean(rgb_face[:, :, 1]))
        features['avg_b'] = float(np.mean(rgb_face[:, :, 2]))
        
        # Edge detection (Canny edge detection)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = float(np.sum(edges) / (face_crop.shape[0] * face_crop.shape[1]))
        
        # Texture (standard deviation of gray values)
        features['texture_std'] = float(np.std(gray))
        
        return features
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        return {}

def predict_bmi(facial_features):
    """
    Predict BMI from facial features.
    
    NOTE: This is a placeholder. In production, you would:
    1. Train a real ML model on labeled data
    2. Use the trained model to make predictions
    3. Include confidence scores
    
    Args:
        facial_features: Dictionary of extracted features
    
    Returns:
        dict: BMI prediction with confidence
    """
    try:
        # Placeholder prediction logic
        # In real implementation, load and use a trained model
        
        if not facial_features:
            return {
                "bmi": None,
                "confidence": 0.0,
                "category": "unknown",
                "error": "No features extracted"
            }
        
        # Simple heuristic-based prediction (for demonstration)
        # In production: use actual ML model (Random Forest, Neural Network, etc.)
        
        # Extract key features
        aspect_ratio = facial_features.get('aspect_ratio', 1.0)
        edge_density = facial_features.get('edge_density', 0.0)
        texture_std = facial_features.get('texture_std', 0.0)
        
        # Simple weighted calculation (placeholder)
        # These weights should come from a trained model
        bmi_estimate = (
            20 +  # Base BMI
            (aspect_ratio - 1.0) * 5 +  # Face shape contribution
            edge_density * 10 +  # Edge features
            texture_std * 0.01  # Texture contribution
        )
        
        # Clamp to realistic BMI range
        bmi_estimate = max(15.0, min(50.0, bmi_estimate))
        
        # Determine BMI category
        if bmi_estimate < 18.5:
            category = "underweight"
        elif bmi_estimate < 25.0:
            category = "normal"
        elif bmi_estimate < 30.0:
            category = "overweight"
        else:
            category = "obese"
        
        return {
            "bmi": round(bmi_estimate, 2),
            "confidence": 0.65,  # Placeholder confidence
            "category": category,
            "error": None
        }
    except Exception as e:
        logger.error(f"BMI prediction error: {str(e)}")
        return {
            "bmi": None,
            "confidence": 0.0,
            "category": "unknown",
            "error": str(e)
        }

def error_handler(f):
    """Decorator for consistent error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }), 500
    return decorated_function

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "api_version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route('/api/detect-faces', methods=['POST'])
@error_handler
def detect_faces_endpoint():
    """
    Endpoint to detect faces in an image.
    
    Request body (JSON):
    {
        "image": "<base64_encoded_image>",
        "format": "base64"  # or "file"
    }
    
    Response (JSON):
    {
        "success": true,
        "faces_detected": 2,
        "face_crops": [
            {
                "crop_base64": "<base64_encoded_crop>",
                "bounding_box": {"x_start": 100, "y_start": 150, ...},
                "face_id": 0
            },
            ...
        ],
        "original_image": "<base64_encoded_original_with_boxes>",
        "timestamp": "2026-05-03T..."
    }
    """
    data = request.get_json()
    
    # Validate request
    if not data or 'image' not in data:
        return jsonify({
            "success": False,
            "error": "Missing 'image' field in request"
        }), 400
    
    # Decode image
    image_array, success, error = decode_image_from_base64(data['image'])
    if not success:
        return jsonify({
            "success": False,
            "error": error
        }), 400
    
    # Detect faces
    face_crops, face_boxes, face_count = detect_faces(image_array)
    
    # Prepare response
    response_data = {
        "success": True,
        "faces_detected": face_count,
        "face_crops": [],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add face crops
    for idx, (face_crop, face_box) in enumerate(zip(face_crops, face_boxes)):
        crop_base64 = encode_image_to_base64(face_crop)
        response_data["face_crops"].append({
            "face_id": idx,
            "crop_base64": crop_base64,
            "bounding_box": face_box,
            "dimensions": {
                "width": face_box["width"],
                "height": face_box["height"]
            }
        })
    
    # Draw bounding boxes on original
    image_with_boxes = image_array.copy()
    for face_box in face_boxes:
        cv2.rectangle(
            image_with_boxes,
            (face_box["x_start"], face_box["y_start"]),
            (face_box["x_end"], face_box["y_end"]),
            (0, 255, 0),
            2
        )
    
    response_data["annotated_image"] = encode_image_to_base64(image_with_boxes)
    
    return jsonify(response_data), 200

@app.route('/api/predict-bmi', methods=['POST'])
@error_handler
def predict_bmi_endpoint():
    """
    Endpoint to predict BMI from a face image.
    
    Request body (JSON):
    {
        "image": "<base64_encoded_face_crop>",
        "face_id": 0
    }
    
    Response (JSON):
    {
        "success": true,
        "face_id": 0,
        "bmi": 23.5,
        "confidence": 0.75,
        "category": "normal",
        "features": {
            "height": 150,
            "width": 120,
            "aspect_ratio": 0.8,
            ...
        },
        "timestamp": "2026-05-03T..."
    }
    """
    data = request.get_json()
    
    # Validate request
    if not data or 'image' not in data:
        return jsonify({
            "success": False,
            "error": "Missing 'image' field in request"
        }), 400
    
    # Decode image
    face_crop, success, error = decode_image_from_base64(data['image'])
    if not success:
        return jsonify({
            "success": False,
            "error": error
        }), 400
    
    # Extract features
    features = extract_facial_features(face_crop)
    
    # Predict BMI
    bmi_prediction = predict_bmi(features)
    
    # Prepare response
    response_data = {
        "success": bmi_prediction["error"] is None,
        "face_id": data.get('face_id', 0),
        "bmi": bmi_prediction["bmi"],
        "confidence": bmi_prediction["confidence"],
        "category": bmi_prediction["category"],
        "features": features,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if bmi_prediction["error"]:
        response_data["error"] = bmi_prediction["error"]
    
    return jsonify(response_data), 200

@app.route('/api/process-image', methods=['POST'])
@error_handler
def process_image_endpoint():
    """
    End-to-end endpoint: detect faces and predict BMI for all faces.
    
    Request body (JSON):
    {
        "image": "<base64_encoded_image>"
    }
    
    Response (JSON):
    {
        "success": true,
        "faces_detected": 2,
        "results": [
            {
                "face_id": 0,
                "bmi": 23.5,
                "confidence": 0.75,
                "category": "normal",
                "bounding_box": {...},
                "crop_base64": "..."
            },
            ...
        ],
        "annotated_image": "<base64>",
        "timestamp": "2026-05-03T..."
    }
    """
    data = request.get_json()
    
    # Validate request
    if not data or 'image' not in data:
        return jsonify({
            "success": False,
            "error": "Missing 'image' field in request"
        }), 400
    
    # Decode image
    image_array, success, error = decode_image_from_base64(data['image'])
    if not success:
        return jsonify({
            "success": False,
            "error": error
        }), 400
    
    # Detect faces
    face_crops, face_boxes, face_count = detect_faces(image_array)
    
    # Process each face
    results = []
    for idx, (face_crop, face_box) in enumerate(zip(face_crops, face_boxes)):
        # Extract features
        features = extract_facial_features(face_crop)
        
        # Predict BMI
        bmi_prediction = predict_bmi(features)
        
        # Add to results
        result = {
            "face_id": idx,
            "bmi": bmi_prediction["bmi"],
            "confidence": bmi_prediction["confidence"],
            "category": bmi_prediction["category"],
            "bounding_box": face_box,
            "crop_base64": encode_image_to_base64(face_crop),
            "features": features
        }
        
        if bmi_prediction["error"]:
            result["error"] = bmi_prediction["error"]
        
        results.append(result)
    
    # Draw bounding boxes on original
    image_with_boxes = image_array.copy()
    for face_box in face_boxes:
        cv2.rectangle(
            image_with_boxes,
            (face_box["x_start"], face_box["y_start"]),
            (face_box["x_end"], face_box["y_end"]),
            (0, 255, 0),
            2
        )
    
    # Prepare response
    response_data = {
        "success": True,
        "faces_detected": face_count,
        "results": results,
        "annotated_image": encode_image_to_base64(image_with_boxes),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return jsonify(response_data), 200

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get API information."""
    return jsonify({
        "name": "Face-to-BMI API",
        "version": API_VERSION,
        "description": "Backend API for face detection and BMI prediction",
        "endpoints": {
            "health": "/api/health",
            "detect_faces": "/api/detect-faces (POST)",
            "predict_bmi": "/api/predict-bmi (POST)",
            "process_image": "/api/process-image (POST)",
            "info": "/api/info"
        }
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "timestamp": datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "timestamp": datetime.utcnow().isoformat()
    }), 405

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  🚀 Face-to-BMI Backend API")
    print("=" * 70)
    print(f"\n📌 API Version: {API_VERSION}")
    print("\n📡 Starting Flask server...")
    print("   → Health check: http://localhost:5000/api/health")
    print("   → API Info: http://localhost:5000/api/info")
    print("   → Docs: http://localhost:5000/api/info")
    print("\n✅ Server running on http://localhost:5000\n")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
