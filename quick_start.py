"""Quick start script for testing the API without full setup."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging

# Simple Flask app for quick testing
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptoms = ['fever', 'headache', 'cough', 'fatigue', 'nausea']
        self.diseases = ['flu', 'cold', 'migraine']
        self._create_sample_model()
    
    def _create_sample_model(self):
        """Create a simple model with sample data."""
        # Create sample training data
        np.random.seed(42)
        X = np.random.randint(0, 2, (100, len(self.symptoms)))
        y = np.random.choice(self.diseases, 100)
        
        # Train model
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y_encoded)
        
        logger.info("Sample model created successfully")
    
    def predict(self, symptoms_input):
        """Make prediction from symptoms."""
        # Create input vector
        input_vector = np.zeros(len(self.symptoms))
        for symptom in symptoms_input:
            if symptom.lower() in self.symptoms:
                idx = self.symptoms.index(symptom.lower())
                input_vector[idx] = 1
        
        # Get prediction
        prediction = self.model.predict([input_vector])[0]
        probabilities = self.model.predict_proba([input_vector])[0]
        
        disease = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        return {
            'disease': disease,
            'confidence': confidence,
            'matched_symptoms': [s for s in symptoms_input if s.lower() in self.symptoms]
        }

# Initialize predictor
predictor = SimplePredictor()

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Quick start API running"})

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify(predictor.symptoms)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400
        
        result = predictor.predict(symptoms)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Quick Start Medical Diagnosis API")
    print("📍 Running on http://localhost:5000")
    print("🧪 This is a simplified version for testing")
    print("\nAvailable endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/symptoms") 
    print("  POST /api/predict")
    print("\nExample request:")
    print('  curl -X POST http://localhost:5000/api/predict \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"symptoms": ["fever", "headache"]}\'')
    
    app.run(host='0.0.0.0', port=5000, debug=True)
