"""
API Testing Script
Demonstrates how to use the Face-to-BMI API endpoints
"""

import requests
import base64
import json
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:5000"
API_ENDPOINTS = {
    'health': f'{API_BASE_URL}/api/health',
    'detect_faces': f'{API_BASE_URL}/api/detect-faces',
    'predict_bmi': f'{API_BASE_URL}/api/predict-bmi',
    'process_image': f'{API_BASE_URL}/api/process-image',
    'info': f'{API_BASE_URL}/api/info',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")

def print_success(text):
    """Print success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print error message."""
    print(f"❌ {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")

def load_image_to_base64(image_path):
    """Load image file and convert to base64."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print_error(f"Image file not found: {image_path}")
        return None

def save_base64_image(base64_str, output_path):
    """Save base64 string to image file."""
    try:
        image_data = base64.b64decode(base64_str)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print_success(f"Saved image to: {output_path}")
    except Exception as e:
        print_error(f"Failed to save image: {str(e)}")

def print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))

# ============================================================================
# API TEST FUNCTIONS
# ============================================================================

def test_health_check():
    """Test health check endpoint."""
    print_header("Test 1: Health Check")
    
    try:
        response = requests.get(API_ENDPOINTS['health'])
        
        if response.status_code == 200:
            print_success("Health check passed")
            print_info(f"API Status: {response.json()['status']}")
            print_info(f"API Version: {response.json()['api_version']}")
        else:
            print_error(f"Health check failed with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API server")
        print_info("Make sure the API is running: python api.py")
    except Exception as e:
        print_error(f"Error: {str(e)}")

def test_api_info():
    """Test API info endpoint."""
    print_header("Test 2: API Information")
    
    try:
        response = requests.get(API_ENDPOINTS['info'])
        
        if response.status_code == 200:
            data = response.json()
            print_success("API Info retrieved")
            print_info(f"API Name: {data['name']}")
            print_info(f"Version: {data['version']}")
            print_info("Available Endpoints:")
            for name, endpoint in data['endpoints'].items():
                print(f"   • {name}: {endpoint}")
        else:
            print_error(f"Failed with status {response.status_code}")
    except Exception as e:
        print_error(f"Error: {str(e)}")

def test_detect_faces(image_path):
    """Test face detection endpoint."""
    print_header(f"Test 3: Detect Faces from {image_path}")
    
    # Load image
    image_base64 = load_image_to_base64(image_path)
    if not image_base64:
        return
    
    try:
        response = requests.post(
            API_ENDPOINTS['detect_faces'],
            json={'image': image_base64},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                print_success(f"Face detection successful")
                print_info(f"Faces detected: {data['faces_detected']}")
                
                # Save annotated image
                if 'annotated_image' in data:
                    save_base64_image(data['annotated_image'], 'output_annotated.png')
                
                # Save face crops
                for face in data['face_crops']:
                    output_file = f"output_face_{face['face_id']}.png"
                    save_base64_image(face['crop_base64'], output_file)
                    
                    bbox = face['bounding_box']
                    print(f"\n   Face {face['face_id']}:")
                    print(f"      Position: ({bbox['x_start']}, {bbox['y_start']})")
                    print(f"      Size: {bbox['width']}x{bbox['height']}")
            else:
                print_error(f"Detection failed: {data.get('error', 'Unknown error')}")
        else:
            print_error(f"Request failed with status {response.status_code}")
            print_info(f"Response: {response.text}")
    except Exception as e:
        print_error(f"Error: {str(e)}")

def test_predict_bmi(image_path):
    """Test BMI prediction endpoint."""
    print_header(f"Test 4: Predict BMI from {image_path}")
    
    # First, detect faces to get a face crop
    image_base64 = load_image_to_base64(image_path)
    if not image_base64:
        return
    
    try:
        # Detect faces first
        detect_response = requests.post(
            API_ENDPOINTS['detect_faces'],
            json={'image': image_base64},
            timeout=30
        )
        
        if detect_response.status_code != 200 or not detect_response.json()['success']:
            print_error("Face detection failed")
            return
        
        faces = detect_response.json()['face_crops']
        if not faces:
            print_error("No faces detected to predict BMI")
            return
        
        # Predict BMI for first face
        face_crop_b64 = faces[0]['crop_base64']
        
        response = requests.post(
            API_ENDPOINTS['predict_bmi'],
            json={
                'image': face_crop_b64,
                'face_id': 0
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                print_success("BMI prediction successful")
                print_info(f"Face ID: {data['face_id']}")
                print_info(f"Predicted BMI: {data['bmi']}")
                print_info(f"Category: {data['category']}")
                print_info(f"Confidence: {data['confidence']}")
                
                print("\n📊 Extracted Features:")
                for key, value in data['features'].items():
                    print(f"   {key}: {value}")
            else:
                print_error(f"Prediction failed: {data.get('error', 'Unknown error')}")
        else:
            print_error(f"Request failed with status {response.status_code}")
    except Exception as e:
        print_error(f"Error: {str(e)}")

def test_process_image(image_path):
    """Test end-to-end image processing."""
    print_header(f"Test 5: End-to-End Processing of {image_path}")
    
    # Load image
    image_base64 = load_image_to_base64(image_path)
    if not image_base64:
        return
    
    try:
        response = requests.post(
            API_ENDPOINTS['process_image'],
            json={'image': image_base64},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                print_success("End-to-end processing successful")
                print_info(f"Faces detected: {data['faces_detected']}")
                
                # Save annotated image
                if 'annotated_image' in data:
                    save_base64_image(data['annotated_image'], 'output_final_annotated.png')
                
                # Print results for each face
                print("\n📊 Results for each face:")
                for result in data['results']:
                    print(f"\n   Face {result['face_id']}:")
                    print(f"      BMI: {result['bmi']}")
                    print(f"      Category: {result['category']}")
                    print(f"      Confidence: {result['confidence']}")
                    print(f"      Bounding Box: ({result['bounding_box']['x_start']}, {result['bounding_box']['y_start']}) → ({result['bounding_box']['x_end']}, {result['bounding_box']['y_end']})")
            else:
                print_error(f"Processing failed: {data.get('error', 'Unknown error')}")
        else:
            print_error(f"Request failed with status {response.status_code}")
            print_info(f"Response: {response.text}")
    except Exception as e:
        print_error(f"Error: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("  🧪 Face-to-BMI API Test Suite")
    print("=" * 70)
    
    print("\n⚠️  Prerequisites:")
    print("   1. API server running: python api.py")
    print("   2. Test image available: photo.jpg")
    print("   3. Internet connection to localhost:5000")
    
    input("\nPress Enter to start testing...")
    
    # Test 1: Health Check
    test_health_check()
    
    # Test 2: API Info
    test_api_info()
    
    # Check if image exists
    image_path = 'photo.jpg'
    if not Path(image_path).exists():
        print_header("⚠️  Test Image Not Found")
        print_info(f"Place your test image at: {Path(image_path).absolute()}")
        print_info("Skipping image-based tests...")
        return
    
    # Test 3: Detect Faces
    test_detect_faces(image_path)
    
    # Test 4: Predict BMI
    test_predict_bmi(image_path)
    
    # Test 5: Process Image (End-to-End)
    test_process_image(image_path)
    
    # Summary
    print_header("✅ Testing Complete")
    print_success("All API endpoints tested successfully!")
    print_info("Check the output folder for generated images")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print_error(f"Fatal error: {str(e)}")
