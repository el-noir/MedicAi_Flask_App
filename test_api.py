"""Simple API testing script."""
import requests
import json

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:5000/api"
    
    print("🧪 Testing Medical Diagnosis API...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test symptoms endpoint
    print("\n2. Testing symptoms endpoint...")
    try:
        response = requests.get(f"{base_url}/symptoms")
        print(f"   Status: {response.status_code}")
        print(f"   Available symptoms: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test prediction endpoint
    print("\n3. Testing prediction endpoint...")
    test_symptoms = ["fever", "headache", "cough"]
    try:
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"symptoms": test_symptoms})
        )
        print(f"   Status: {response.status_code}")
        print(f"   Input symptoms: {test_symptoms}")
        print(f"   Prediction: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_api()
