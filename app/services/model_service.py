"""Model management service."""
from app.models.diagnosis import MedicalDiagnosisModel
from flask import current_app
import logging

logger = logging.getLogger(__name__)


class ModelService:
    """Service for ML model operations."""
    
    @staticmethod
    def retrain_models():
        """Retrain all models with fresh data."""
        try:
            model_path = current_app.config.get('MODEL_PATH', 'data/models')
            data_path = current_app.config.get('DATA_PATH', 'data/raw')
            
            model = MedicalDiagnosisModel(model_path, data_path)
            
            if model.load_data():
                if model.train_models():
                    if model.save_models():
                        logger.info("Models retrained and saved successfully")
                        return True
                    else:
                        logger.error("Failed to save retrained models")
                        return False
                else:
                    logger.error("Failed to train models")
                    return False
            else:
                logger.error("Failed to load data for retraining")
                return False
                
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
            return False
