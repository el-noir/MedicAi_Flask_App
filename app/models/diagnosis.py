"""Medical diagnosis models and data structures."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MedicalDiagnosisModel:
    """Medical diagnosis ML model wrapper."""
    
    def __init__(self, model_path: str = "data/models", data_path: str = "data/raw"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.models = {}
        self.label_encoder = None
        self.disease_info = {}
        self.all_symptoms = []
        self.symptom_severity = {}
        self.symptoms_df = None
        
        # Create directories if they don't exist
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> bool:
        """Load and preprocess all required datasets."""
        try:
            # Load datasets
            symptoms_df = pd.read_csv(self.data_path / 'dataset.csv')
            desc_df = pd.read_csv(self.data_path / 'symptom_Description.csv')
            precaution_df = pd.read_csv(self.data_path / 'symptom_precaution.csv')
            severity_df = pd.read_csv(self.data_path / 'Symptom-severity.csv')
            
            # Process data
            self._process_symptoms_data(symptoms_df)
            self._prepare_disease_info(desc_df, precaution_df, severity_df)
            
            logger.info("Data loaded and processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _process_symptoms_data(self, symptoms_df: pd.DataFrame):
        """Process symptoms data and create binary features."""
        # Get all unique symptoms from the dataset
        all_symptoms = set()
        for col in symptoms_df.columns[1:]:  # Skip Disease column
            unique_symptoms = symptoms_df[col].dropna().unique()
            all_symptoms.update(unique_symptoms)
        
        # Clean symptoms
        self.all_symptoms = [s.strip().lower() for s in all_symptoms
                             if isinstance(s, str) and s.strip() != '']
        
        # Create binary features
        symptom_columns = {symptom: 0 for symptom in self.all_symptoms}
        symptoms_binary = pd.DataFrame([symptom_columns] * len(symptoms_df))
        
        # Mark symptoms present in each row
        for index, row in symptoms_df.iterrows():
            for col in symptoms_df.columns[1:17]:  # Symptom columns
                symptom = row[col]
                if pd.notna(symptom) and symptom.strip().lower() in symptom_columns:
                    symptoms_binary.at[index, symptom.strip().lower()] = 1
        
        # Combine with original dataframe
        self.symptoms_df = pd.concat([symptoms_df['Disease'], symptoms_binary], axis=1)
    
    def _prepare_disease_info(self, desc_df: pd.DataFrame,
                             precaution_df: pd.DataFrame,
                             severity_df: pd.DataFrame):
        """Prepare disease information dictionary."""
        # Create symptom severity mapping
        self.symptom_severity = dict(zip(
            severity_df['Symptom'].str.strip().str.lower(),
            severity_df['weight']
        ))
        
        # Initialize disease info with descriptions
        self.disease_info = {}
        for _, row in desc_df.iterrows():
            disease = row['Disease'].strip()
            self.disease_info[disease.lower()] = {
                'description': row['Description'],
                'precautions': [],
                'severity_score': 0,
                'critical_symptoms': []
            }
        
        # Add precautions
        for _, row in precaution_df.iterrows():
            disease = row['Disease'].strip().lower()
            precautions = [row[f'Precaution_{i}'] for i in range(1, 5)
                          if pd.notna(row[f'Precaution_{i}'])]
            if disease in self.disease_info:
                self.disease_info[disease]['precautions'] = precautions
        
        # Calculate severity scores for each disease
        for disease in self.disease_info:
            disease_rows = self.symptoms_df[
                self.symptoms_df['Disease'].str.strip().str.lower() == disease
            ]
            
            disease_symptoms = set()
            for _, row in disease_rows.iterrows():
                for symptom in self.all_symptoms:
                    if row[symptom] == 1:
                        disease_symptoms.add(symptom)
            
            # Calculate average severity and identify critical symptoms
            if disease_symptoms:
                severities = []
                critical_symptoms = []
                
                for symptom in disease_symptoms:
                    severity = self.symptom_severity.get(symptom, 0)
                    severities.append(severity)
                    if severity >= 6:  # Critical threshold
                        critical_symptoms.append(symptom)
                
                self.disease_info[disease]['severity_score'] = np.mean(severities)
                self.disease_info[disease]['critical_symptoms'] = critical_symptoms
    
    def train_models(self) -> bool:
        """Train and evaluate machine learning models."""
        try:
            # Prepare features and target
            X = self.symptoms_df[self.all_symptoms]
            y = self.symptoms_df['Disease'].str.strip().str.lower()
            
            # Encode target labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Create and train models
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=150, random_state=42, class_weight='balanced'
            )
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
            
            # Train models
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
            
            # Evaluate models
            for name, model in self.models.items():
                self._evaluate_model(name, model, X_test, y_test)
            
            logger.info("Models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def _evaluate_model(self, name: str, model, X_test, y_test):
        """Evaluate and log model performance."""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{name} accuracy: {accuracy:.2f}")
    
    def save_models(self) -> bool:
        """Save trained models and data."""
        try:
            joblib.dump(self.models['random_forest'], 
                       self.model_path / 'rf_disease_predictor.pkl')
            joblib.dump(self.models['gradient_boosting'], 
                       self.model_path / 'gb_disease_predictor.pkl')
            joblib.dump(self.label_encoder, 
                       self.model_path / 'label_encoder.pkl')
            joblib.dump(self.disease_info, 
                       self.model_path / 'disease_info.pkl')
            joblib.dump(self.all_symptoms, 
                       self.model_path / 'all_symptoms.pkl')
            joblib.dump(self.symptom_severity, 
                       self.model_path / 'symptom_severity.pkl')
            
            logger.info("Models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self) -> bool:
        """Load pre-trained models."""
        try:
            model_files = [
                'rf_disease_predictor.pkl',
                'gb_disease_predictor.pkl',
                'label_encoder.pkl',
                'disease_info.pkl',
                'all_symptoms.pkl',
                'symptom_severity.pkl'
            ]
            
            # Check if all model files exist
            if not all((self.model_path / f).exists() for f in model_files):
                logger.warning("Some model files are missing")
                return False
            
            # Load models
            self.models['random_forest'] = joblib.load(
                self.model_path / 'rf_disease_predictor.pkl')
            self.models['gradient_boosting'] = joblib.load(
                self.model_path / 'gb_disease_predictor.pkl')
            self.label_encoder = joblib.load(
                self.model_path / 'label_encoder.pkl')
            self.disease_info = joblib.load(
                self.model_path / 'disease_info.pkl')
            self.all_symptoms = joblib.load(
                self.model_path / 'all_symptoms.pkl')
            self.symptom_severity = joblib.load(
                self.model_path / 'symptom_severity.pkl')
            
            logger.info("Pre-trained models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
