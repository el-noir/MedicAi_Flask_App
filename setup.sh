#!/bin/bash

# Medical Diagnosis API Setup Script

echo "🏥 Setting up Medical Diagnosis API..."

# Create project directory
mkdir -p medical-diagnosis-api
cd medical-diagnosis-api

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# For Windows users, use: venv\Scripts\activate

echo "✅ Virtual environment created and activated"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p app/{api,models,services,utils}
mkdir -p data/{raw,models}
mkdir -p tests
mkdir -p scripts
mkdir -p logs

# Create __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py
touch tests/__init__.py

echo "✅ Directory structure created"

# Install dependencies
echo "📚 Installing dependencies..."
pip install Flask==3.0.3
pip install Flask-CORS==4.0.1
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install scikit-learn==1.5.1
pip install joblib==1.4.2
pip install python-dotenv==1.0.1
pip install gunicorn==22.0.0

echo "✅ Dependencies installed"

# Create sample data files (you'll need to replace these with your actual data)
echo "📊 Creating sample data structure..."
echo "Disease,Symptom_1,Symptom_2,Symptom_3" > data/raw/dataset.csv
echo "flu,fever,headache,cough" >> data/raw/dataset.csv

echo "Disease,Description" > data/raw/symptom_Description.csv
echo "flu,Common viral infection" >> data/raw/symptom_Description.csv

echo "Disease,Precaution_1,Precaution_2,Precaution_3,Precaution_4" > data/raw/symptom_precaution.csv
echo "flu,rest,drink fluids,take medicine,see doctor" >> data/raw/symptom_precaution.csv

echo "Symptom,weight" > data/raw/Symptom-severity.csv
echo "fever,6" >> data/raw/Symptom-severity.csv
echo "headache,3" >> data/raw/Symptom-severity.csv
echo "cough,4" >> data/raw/Symptom-severity.csv

echo "✅ Sample data files created"

echo "🎉 Setup complete! Next steps:"
echo "1. Copy your actual CSV data files to data/raw/"
echo "2. Copy the Python files from the project structure"
echo "3. Create .env file from .env.example"
echo "4. Run the application with: python run.py"
