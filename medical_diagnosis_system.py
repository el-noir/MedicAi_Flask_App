import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import defaultdict
import pprint
from typing import List, Dict, Any

class MedicalDiagnosisSystem:
    def __init__(self):
        self.models = {}
        self.label_encoder = None
        self.disease_info = {}
        self.all_symptoms = []
        self.symptom_severity = {}
        self.critical_threshold = 6  # Severity score threshold for emergency
        self.min_confidence = 0.1  # Minimum confidence threshold for predictions

    def load_data(self):
        """Load and preprocess all required datasets"""
        try:
            # Load datasets
            symptoms_df = pd.read_csv('dataset.csv')
            desc_df = pd.read_csv('symptom_Description.csv')
            precaution_df = pd.read_csv('symptom_precaution.csv')
            severity_df = pd.read_csv('Symptom-severity.csv')

            # Process symptoms data
            self._process_symptoms_data(symptoms_df)
            
            # Prepare disease information
            self._prepare_disease_info(desc_df, precaution_df, severity_df)

            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def _process_symptoms_data(self, symptoms_df: pd.DataFrame):
        """Process symptoms data and create binary features"""
        # Get all unique symptoms from the dataset
        all_symptoms = set()
        for col in symptoms_df.columns[1:]:  # Skip Disease column
            unique_symptoms = symptoms_df[col].dropna().unique()
            all_symptoms.update(unique_symptoms)

        # Clean symptoms
        self.all_symptoms = [s.strip().lower() for s in all_symptoms 
                            if isinstance(s, str) and s.strip() != '']
        
        # Create binary features efficiently without fragmentation
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
        """Prepare disease information dictionary"""
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
            # Get symptoms for this disease
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
                    if severity >= self.critical_threshold:
                        critical_symptoms.append(symptom)
                
                self.disease_info[disease]['severity_score'] = np.mean(severities)
                self.disease_info[disease]['critical_symptoms'] = critical_symptoms

    def train_models(self):
        """Train and evaluate machine learning models"""
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
                print(f"Training {name}...")
                model.fit(X_train, y_train)

            # Evaluate models
            print("\nModel Evaluation:")
            for name, model in self.models.items():
                self._evaluate_model(name, model, X_test, y_test)

            # Cross-validation
            print("\nCross-validation Scores:")
            for name, model in self.models.items():
                scores = cross_val_score(model, X, y_encoded, cv=5)
                print(f"{name}: {np.mean(scores):.2f} accuracy with std: {np.std(scores):.2f}")

            return True
        except Exception as e:
            print(f"Error training models: {str(e)}")
            return False

    def _evaluate_model(self, name: str, model, X_test, y_test):
        """Evaluate and print model performance"""
        y_pred = model.predict(X_test)
        print(f"\n{name.capitalize()} Evaluation:")
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))

    def save_models(self, path_prefix=''):
        """Save trained models and data"""
        try:
            joblib.dump(self.models['random_forest'], f'{path_prefix}rf_disease_predictor.pkl')
            joblib.dump(self.models['gradient_boosting'], f'{path_prefix}gb_disease_predictor.pkl')
            joblib.dump(self.label_encoder, f'{path_prefix}label_encoder.pkl')
            joblib.dump(self.disease_info, f'{path_prefix}disease_info.pkl')
            joblib.dump(self.all_symptoms, f'{path_prefix}all_symptoms.pkl')
            joblib.dump(self.symptom_severity, f'{path_prefix}symptom_severity.pkl')
            return True
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False

    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease with full details including description, precautions, and severity"""
        try:
            # Clean and validate input symptoms
            cleaned_symptoms = [s.strip().lower() for s in symptoms if isinstance(s, str) and s.strip()]
            valid_symptoms = [s for s in cleaned_symptoms if s in self.all_symptoms]
            
            if not valid_symptoms:
                return {"error": "No valid symptoms provided", "valid_symptoms": self.all_symptoms}

            # Prepare input data
            input_data = np.zeros(len(self.all_symptoms))
            for symptom in valid_symptoms:
                index = self.all_symptoms.index(symptom)
                input_data[index] = 1
            input_data = input_data.reshape(1, -1)

            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                proba = model.predict_proba(input_data)[0]
                predictions[name] = {
                    'probabilities': proba,
                    'top_indices': np.argsort(proba)[-3:][::-1]  # Top 3 predictions
                }

            # Combine predictions (average probabilities)
            combined_proba = np.mean([pred['probabilities'] for pred in predictions.values()], axis=0)
            top3_indices = np.argsort(combined_proba)[-3:][::-1]
            top3_confidences = combined_proba[top3_indices]

            # Filter predictions by minimum confidence
            valid_predictions = [
                (idx, conf) for idx, conf in zip(top3_indices, top3_confidences) 
                if conf >= self.min_confidence
            ]

            if not valid_predictions:
                return {"warning": "No predictions met confidence threshold", "confidence_threshold": self.min_confidence}

            # Prepare results
            results = []
            emergency_flag = False
            severe_symptoms = [
                s for s in valid_symptoms 
                if self.symptom_severity.get(s, 0) >= 5
            ]

            for idx, confidence in valid_predictions:
                disease = self.label_encoder.classes_[idx]
                disease_data = self.disease_info.get(disease.lower(), {})
                
                # Check for emergency conditions
                current_severity = disease_data.get('severity_score', 0)
                critical_symptoms = set(disease_data.get('critical_symptoms', [])) & set(valid_symptoms)
                
                if current_severity >= self.critical_threshold or critical_symptoms:
                    emergency_flag = True
                
                results.append({
                    'disease': disease,
                    'confidence': float(confidence),
                    'description': disease_data.get('description', 'No description available'),
                    'precautions': disease_data.get('precautions', []),
                    'severity_score': disease_data.get('severity_score', 0),
                    'critical_symptoms': list(critical_symptoms),
                    'matched_symptoms': valid_symptoms
                })

            # Calculate overall risk score (weighted average by symptom severity)
            severity_scores = [self.symptom_severity.get(s, 0) for s in valid_symptoms]
            risk_score = np.average(
                severity_scores,
                weights=[1 + s/10 for s in severity_scores]  # Give more weight to severe symptoms
            )

            return {
                'predictions': results,
                'emergency': emergency_flag,
                'overall_risk_score': float(risk_score),
                'severe_symptoms_present': severe_symptoms,
                'matched_symptoms': valid_symptoms
            }

        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    print("Initializing medical diagnosis system...")
    system = MedicalDiagnosisSystem()
    
    if system.load_data():
        print("Data loaded successfully")
        
        if system.train_models():
            print("Models trained successfully")
            
            if system.save_models():
                print("Models saved successfully")
                
                # Sample prediction
                sample_symptoms = ['high fever', 'chest pain', 'breathlessness']
                print("\nSample Prediction:")
                result = system.predict_disease(sample_symptoms)
                pp = pprint.PrettyPrinter(indent=2)
                pp.pprint(result)
            else:
                print("Failed to save models")
        else:
            print("Failed to train models")
    else:
        print("Failed to load data")