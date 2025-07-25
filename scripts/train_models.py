"""Script to train and save ML models."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.diagnosis import MedicalDiagnosisModel
from app.utils.helpers import setup_logging
import logging


def main():
    """Train and save models."""
    setup_logging('INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    
    model = MedicalDiagnosisModel()
    
    if model.load_data():
        logger.info("Data loaded successfully")
        
        if model.train_models():
            logger.info("Models trained successfully")
            
            if model.save_models():
                logger.info("Models saved successfully")
                print("✅ Model training completed successfully!")
            else:
                logger.error("Failed to save models")
                sys.exit(1)
        else:
            logger.error("Failed to train models")
            sys.exit(1)
    else:
        logger.error("Failed to load data")
        sys.exit(1)


if __name__ == "__main__":
    main()
