"""Diagnosis service layer."""
import numpy as np
from typing import List, Dict, Any
from flask import current_app
from app.models.diagnosis import MedicalDiagnosisModel
import logging

logger = logging.getLogger(__name__)


class DiagnosisService:
    """Service for medical diagnosis operations."""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the diagnosis model."""
        try:
            model_path = current_app.config.get('MODEL_PATH', 'data/models')
            data_path = current_app.config.get('DATA_PATH', 'data/raw')
            
            self.model = MedicalDiagnosisModel(model_path, data_path)
            
            # Try to load pre-trained models first
            if not self.model.load_models():
                logger.info("Pre-trained models not found, loading data and training...")
                if self.model.load_data():
                    if self.model.train_models():
                        self.model.save_models()
                    else:
                        raise RuntimeError("Failed to train models")
                else:
                    raise RuntimeError("Failed to load data")
            
            logger.info("Diagnosis service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize diagnosis service: {str(e)}")
            raise
    
    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease from symptoms."""
        try:
            # Get configuration
            critical_threshold = current_app.config.get('CRITICAL_THRESHOLD', 6)
            min_confidence = current_app.config.get('MIN_CONFIDENCE', 0.1)
            
            # Clean and validate input symptoms
            cleaned_symptoms = [s.strip().lower() for s in symptoms 
                              if isinstance(s, str) and s.strip()]
            valid_symptoms = [s for s in cleaned_symptoms 
                            if s in self.model.all_symptoms]
            
            if not valid_symptoms:
                return {
                    "error": "No valid symptoms provided",
                    "valid_symptoms": self.model.all_symptoms
                }
            
            # Prepare input data
            input_data = np.zeros(len(self.model.all_symptoms))
            for symptom in valid_symptoms:
                index = self.model.all_symptoms.index(symptom)
                input_data[index] = 1
            input_data = input_data.reshape(1, -1)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.model.models.items():
                proba = model.predict_proba(input_data)[0]
                predictions[name] = {
                    'probabilities': proba,
                    'top_indices': np.argsort(proba)[-3:][::-1]
                }
            
            # Combine predictions (average probabilities)
            combined_proba = np.mean([pred['probabilities'] 
                                    for pred in predictions.values()], axis=0)
            top3_indices = np.argsort(combined_proba)[-3:][::-1]
            top3_confidences = combined_proba[top3_indices]
            
            # Filter predictions by minimum confidence
            valid_predictions = [
                (idx, conf) for idx, conf in zip(top3_indices, top3_confidences)
                if conf >= min_confidence
            ]
            
            if not valid_predictions:
                return {
                    "warning": "No predictions met confidence threshold",
                    "confidence_threshold": min_confidence
                }
            
            # Prepare results
            results = []
            emergency_flag = False
            severe_symptoms = [
                s for s in valid_symptoms
                if self.model.symptom_severity.get(s, 0) >= 5
            ]
            
            for idx, confidence in valid_predictions:
                disease = self.model.label_encoder.classes_[idx]
                disease_data = self.model.disease_info.get(disease.lower(), {})
                
                # Check for emergency conditions
                current_severity = disease_data.get('severity_score', 0)
                critical_symptoms = (set(disease_data.get('critical_symptoms', [])) & 
                                   set(valid_symptoms))
                
                if current_severity >= critical_threshold or critical_symptoms:
                    emergency_flag = True
                
                results.append({
                    'disease': disease,
                    'confidence': float(confidence),
                    'description': disease_data.get('description', 
                                                  'No description available'),
                    'precautions': disease_data.get('precautions', []),
                    'severity_score': disease_data.get('severity_score', 0),
                    'critical_symptoms': list(critical_symptoms),
                    'matched_symptoms': valid_symptoms
                })
            
            # Calculate overall risk score
            severity_scores = [self.model.symptom_severity.get(s, 0) 
                             for s in valid_symptoms]
            risk_score = np.average(
                severity_scores,
                weights=[1 + s/10 for s in severity_scores]
            )
            
            return {
                'predictions': results,
                'emergency': emergency_flag,
                'overall_risk_score': float(risk_score),
                'severe_symptoms_present': severe_symptoms,
                'matched_symptoms': valid_symptoms
            }
            
        except Exception as e:
            logger.error(f"Error in disease prediction: {str(e)}")
            return {"error": str(e)}
    
    def get_all_symptoms(self) -> List[str]:
        """Get all recognized symptoms."""
        return self.model.all_symptoms if self.model else []
    
    def get_all_diseases(self) -> List[str]:
        """Get all known diseases."""
        return (list(self.model.disease_info.keys()) 
                if self.model and self.model.disease_info else [])
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        if not self.model:
            return {
                "status": "unhealthy",
                "error": "Model not initialized"
            }
        
        return {
            "status": "healthy",
            "models_loaded": len(self.model.models) > 0,
            "symptoms_count": len(self.model.all_symptoms),
            "diseases_count": len(self.model.disease_info)
        }
