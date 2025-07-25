"""Input validation utilities."""
from typing import List, Dict, Any


def validate_symptoms(symptoms) -> Dict[str, Any]:
    """Validate symptoms input."""
    if not symptoms:
        return {
            'valid': False,
            'message': 'No symptoms provided'
        }
    
    # Convert single string to list if needed
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(',')]
    
    # Ensure we have a list
    if not isinstance(symptoms, list):
        return {
            'valid': False,
            'message': 'Symptoms must be a list or comma-separated string'
        }
    
    # Clean symptoms - lowercase and trim
    cleaned_symptoms = [s.strip().lower() for s in symptoms if s.strip()]
    
    if not cleaned_symptoms:
        return {
            'valid': False,
            'message': 'Please provide valid symptoms'
        }
    
    return {
        'valid': True,
        'symptoms': cleaned_symptoms
    }


def validate_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate prediction request data."""
    if not data:
        return {
            'valid': False,
            'message': 'No data provided'
        }
    
    if 'symptoms' not in data:
        return {
            'valid': False,
            'message': 'Symptoms field is required'
        }
    
    return validate_symptoms(data['symptoms'])
