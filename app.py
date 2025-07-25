from flask import Flask, request, jsonify
from flask_cors import CORS
from medical_diagnosis_system import MedicalDiagnosisSystem
import joblib
import os

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origin": "http://localhost:5173"},
    r"/symptoms": {"origin": "http://localhost:5173"}
})
# Initialize the diagnosis system
diagnosis_system = MedicalDiagnosisSystem()

def initialize_system():
    """Initialize the diagnosis system"""
    if not diagnosis_system.load_data():
        raise RuntimeError("Failed to load data")
    
    # Try to load pre-trained models
    model_files = [
        'rf_disease_predictor.pkl',
        'gb_disease_predictor.pkl',
        'label_encoder.pkl',
        'disease_info.pkl',
        'all_symptoms.pkl',
        'symptom_severity.pkl'
    ]
    
    if all(os.path.exists(f) for f in model_files):
        try:
            diagnosis_system.models['random_forest'] = joblib.load('rf_disease_predictor.pkl')
            diagnosis_system.models['gradient_boosting'] = joblib.load('gb_disease_predictor.pkl')
            diagnosis_system.label_encoder = joblib.load('label_encoder.pkl')
            diagnosis_system.disease_info = joblib.load('disease_info.pkl')
            diagnosis_system.all_symptoms = joblib.load('all_symptoms.pkl')
            diagnosis_system.symptom_severity = joblib.load('symptom_severity.pkl')
            print("Pre-trained models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    else:
        print("Training new models...")
        if diagnosis_system.train_models() and diagnosis_system.save_models():
            return True
        return False

# Initialize on startup
if not initialize_system():
    raise RuntimeError("System initialization failed")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from symptoms"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        # Convert single string to list if needed
        if isinstance(symptoms, str):
            symptoms = [s.strip() for s in symptoms.split(',')]
        
        # Ensure we have a list
        if not isinstance(symptoms, list):
            return jsonify({"error": "Symptoms must be a list"}), 400
            
        # Clean symptoms - lowercase and trim
        symptoms = [s.strip().lower() for s in symptoms if s.strip()]
        
        if not symptoms:
            return jsonify({"error": "Please provide valid symptoms"}), 400
        
        result = diagnosis_system.predict_disease(symptoms)
        
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get all recognized symptoms"""
    return jsonify(diagnosis_system.all_symptoms)

@app.route('/diseases', methods=['GET'])
def get_diseases():
    """Get all known diseases"""
    return jsonify(list(diagnosis_system.disease_info.keys()))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(diagnosis_system.models) > 0,
        "symptoms_count": len(diagnosis_system.all_symptoms),
        "diseases_count": len(diagnosis_system.disease_info)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)