"""API routes."""
from flask import request, jsonify, current_app
from app.api import bp
from app.services.diagnosis_service import DiagnosisService
from app.utils.validators import validate_symptoms
import logging

logger = logging.getLogger(__name__)

# Initialize diagnosis service
diagnosis_service = DiagnosisService()


@bp.route('/predict', methods=['POST'])
def predict():
    """Predict disease from symptoms."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        symptoms = data.get('symptoms', [])
        
        # Validate input
        validation_result = validate_symptoms(symptoms)
        if not validation_result['valid']:
            return jsonify({"error": validation_result['message']}), 400
        
        # Get prediction
        result = diagnosis_service.predict_disease(validation_result['symptoms'])
        
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            return jsonify(result), 400
        
        logger.info(f"Successful prediction for symptoms: {validation_result['symptoms']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@bp.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get all recognized symptoms."""
    try:
        symptoms = diagnosis_service.get_all_symptoms()
        return jsonify(symptoms)
    except Exception as e:
        logger.error(f"Error getting symptoms: {str(e)}")
        return jsonify({"error": "Failed to retrieve symptoms"}), 500


@bp.route('/diseases', methods=['GET'])
def get_diseases():
    """Get all known diseases."""
    try:
        diseases = diagnosis_service.get_all_diseases()
        return jsonify(diseases)
    except Exception as e:
        logger.error(f"Error getting diseases: {str(e)}")
        return jsonify({"error": "Failed to retrieve diseases"}), 500


@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        health_status = diagnosis_service.get_health_status()
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": "Health check failed"
        }), 500
