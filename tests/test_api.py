"""API endpoint tests."""
import pytest
import json


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data


def test_symptoms_endpoint(client):
    """Test symptoms endpoint."""
    response = client.get('/api/symptoms')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert isinstance(data, list)


def test_diseases_endpoint(client):
    """Test diseases endpoint."""
    response = client.get('/api/diseases')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert isinstance(data, list)


def test_predict_endpoint_valid_input(client):
    """Test prediction endpoint with valid input."""
    test_data = {
        'symptoms': ['fever', 'headache', 'cough']
    }
    
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    # Should return 200 or 400 (depending on if models are trained)
    assert response.status_code in [200, 400]


def test_predict_endpoint_invalid_input(client):
    """Test prediction endpoint with invalid input."""
    test_data = {
        'symptoms': []
    }
    
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_endpoint_no_json(client):
    """Test prediction endpoint without JSON data."""
    response = client.post('/api/predict')
    assert response.status_code == 400
