"""
Client script to test Face-to-BMI API
"""

import requests
import base64
import json
from pathlib import Path


class FaceToBMIClient:
    """Client for Face-to-BMI API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"✓ Health check: {response.json()}")
            return response.json()
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return None
    
    def predict_from_file(self, image_path):
        """Predict BMI from image file"""
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.base_url}/predict",
                    files=files
                )
            
            result = response.json()
            self._print_response("File Upload Prediction", result)
            return result
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            return None
    
    def predict_from_base64(self, image_path):
        """Predict BMI from base64 encoded image"""
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {"image": image_data}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload
            )
            
            result = response.json()
            self._print_response("Base64 Prediction", result)
            return result
        except Exception as e:
            print(f"✗ Base64 prediction failed: {e}")
            return None
    
    def batch_predict(self, image_paths):
        """Predict BMI for multiple images"""
        try:
            images = []
            for path in image_paths:
                with open(path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    images.append(image_data)
            
            payload = {"images": images}
            response = requests.post(
                f"{self.base_url}/batch-predict",
                json=payload
            )
            
            result = response.json()
            self._print_response("Batch Prediction", result)
            return result
        except Exception as e:
            print(f"✗ Batch prediction failed: {e}")
            return None
    
    def _print_response(self, title, result):
        """Pretty print API response"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2))


def main():
    """Test the API"""
    client = FaceToBMIClient()
    
    # Health check
    print("Testing Face-to-BMI API Client\n")
    client.health_check()
    
    # Test with a sample image (replace with actual path)
    sample_image = "sample_image.jpg"
    
    if Path(sample_image).exists():
        print(f"\n\nTesting with image: {sample_image}")
        
        # Test file upload
        client.predict_from_file(sample_image)
        
        # Test base64
        client.predict_from_base64(sample_image)
        
        # Test batch
        client.batch_predict([sample_image])
    else:
        print(f"\n✓ API client ready. Provide a test image to proceed.")
        print(f"  Usage: Place image at '{sample_image}' and run this script")
        print(f"\n  Or use curl to test:\n")
        print(f"  # File upload:")
        print(f"  curl -X POST http://localhost:5000/predict \\")
        print(f"    -F 'image=@path/to/image.jpg'\n")
        print(f"  # Base64 JSON:")
        print(f"  curl -X POST http://localhost:5000/predict \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -d '{{\"image\": \"base64_encoded_image_data\"}}'\n")


if __name__ == '__main__':
    main()
