# Medical Diagnosis API

A professional Flask-based REST API for medical diagnosis using machine learning models. This system predicts diseases based on symptoms using Random Forest and Gradient Boosting algorithms, with emergency condition detection and comprehensive health monitoring.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0.3-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

- **Disease Prediction**: Predict diseases based on input symptoms using multiple ML models
- **Multiple ML Models**: Random Forest and Gradient Boosting classifiers for improved accuracy
- **Emergency Detection**: Automatic detection of critical symptoms requiring immediate attention
- **Risk Assessment**: Overall risk scoring based on symptom severity
- **Professional Architecture**: Clean, scalable Flask application structure
- **Docker Support**: Containerized deployment ready
- **Comprehensive Testing**: Full test suite with pytest
- **Health Monitoring**: Built-in health check and system status endpoints
- **CORS Support**: Ready for frontend integration

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [Data Requirements](#-data-requirements)
- [Development](#-development)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Option 1: Quick Test (Recommended for first try)

Perfect for immediate testing without full setup:

\`\`\`bash
# 1. Download the quick start file
wget https://raw.githubusercontent.com/your-repo/medical-diagnosis-api/main/quick_start.py

# 2. Install basic dependencies
pip install flask flask-cors pandas numpy scikit-learn

# 3. Run the simplified API
python quick_start.py
\`\`\`

The API will start on `http://localhost:5000` with sample data.

**Test the API:**
\`\`\`bash
# Health check
curl http://localhost:5000/api/health

# Get available symptoms
curl http://localhost:5000/api/symptoms

# Make a prediction
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"symptoms": ["fever", "headache", "cough"]}'
\`\`\`

### Option 2: Docker (Easiest for full setup)

\`\`\`bash
# Clone the repository
git clone https://github.com/your-username/medical-diagnosis-api.git
cd medical-diagnosis-api

# Run with Docker Compose
docker-compose up --build
\`\`\`

### Option 3: Full Development Setup

\`\`\`bash
# Clone and setup
git clone https://github.com/your-username/medical-diagnosis-api.git
cd medical-diagnosis-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Add your data files to data/raw/
# Run the application
python run.py
\`\`\`

## 📁 Project Structure

\`\`\`
medical-diagnosis-api/
├── app/                           # Main application package
│   ├── __init__.py               # Flask app factory
│   ├── config.py                 # Configuration management
│   ├── extensions.py             # Flask extensions
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   ├── routes.py             # API endpoints
│   │   └── errors.py             # Error handlers
│   ├── models/                   # Data models and ML
│   │   ├── __init__.py
│   │   └── diagnosis.py          # ML models and data processing
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── diagnosis_service.py  # Diagnosis business logic
│   │   └── model_service.py      # Model management
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── validators.py         # Input validation
│       └── helpers.py            # Helper functions
├── data/                         # Data storage
│   ├── raw/                      # Raw CSV datasets
│   │   ├── dataset.csv
│   │   ├── symptom_Description.csv
│   │   ├── symptom_precaution.csv
│   │   └── Symptom-severity.csv
│   └── models/                   # Trained model files
│       ├── rf_disease_predictor.pkl
│       ├── gb_disease_predictor.pkl
│       ├── label_encoder.pkl
│       ├── disease_info.pkl
│       ├── all_symptoms.pkl
│       └── symptom_severity.pkl
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Test configuration
│   ├── test_api.py              # API tests
│   ├── test_services.py         # Service tests
│   └── test_models.py           # Model tests
├── scripts/                      # Utility scripts
│   ├── train_models.py          # Model training
│   └── setup_data.py            # Data setup
├── logs/                         # Application logs
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose setup
├── wsgi.py                      # WSGI entry point
├── run.py                       # Development server
└── README.md                    # This file
\`\`\`

## 🔧 Installation

### Prerequisites

- Python 3.9+ (3.12 recommended)
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Step-by-Step Installation

1. **Clone the Repository**
   \`\`\`bash
   git clone https://github.com/your-username/medical-diagnosis-api.git
   cd medical-diagnosis-api
   \`\`\`

2. **Create Virtual Environment**
   \`\`\`bash
   python -m venv venv
   
   # Activate virtual environment
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   \`\`\`

3. **Install Dependencies**
   \`\`\`bash
   # Production dependencies
   pip install -r requirements.txt
   
   # For development (includes testing tools)
   pip install -r requirements-dev.txt
   \`\`\`

4. **Environment Configuration**
   \`\`\`bash
   cp .env.example .env
   \`\`\`
   
   Edit `.env` file with your configuration:
   \`\`\`env
   FLASK_CONFIG=development
   FLASK_DEBUG=true
   SECRET_KEY=your-secret-key-here
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   MODEL_PATH=data/models
   DATA_PATH=data/raw
   CRITICAL_THRESHOLD=6
   MIN_CONFIDENCE=0.1
   LOG_LEVEL=INFO
   LOG_FILE=logs/app.log
   \`\`\`

5. **Prepare Data Files**
   
   Place your CSV files in the `data/raw/` directory:
   
   - **`dataset.csv`**: Main dataset with diseases and symptoms
     \`\`\`csv
     Disease,Symptom_1,Symptom_2,Symptom_3,...
     flu,fever,headache,cough,...
     cold,runny nose,sneezing,cough,...
     \`\`\`
   
   - **`symptom_Description.csv`**: Disease descriptions
     \`\`\`csv
     Disease,Description
     flu,Common viral infection affecting respiratory system
     cold,Mild viral infection of nose and throat
     \`\`\`
   
   - **`symptom_precaution.csv`**: Precautions for each disease
     \`\`\`csv
     Disease,Precaution_1,Precaution_2,Precaution_3,Precaution_4
     flu,rest,drink fluids,take medicine,see doctor if severe
     cold,rest,warm liquids,avoid cold air,use tissues
     \`\`\`
   
   - **`Symptom-severity.csv`**: Symptom severity weights
     \`\`\`csv
     Symptom,weight
     fever,6
     chest pain,8
     headache,3
     cough,4
     \`\`\`

6. **Initialize the Application**
   \`\`\`bash
   # Train models (optional - will happen automatically on first run)
   python scripts/train_models.py
   
   # Run the application
   python run.py
   \`\`\`

The API will be available at `http://localhost:5000`

## 📚 API Documentation

### Base URL
\`\`\`
http://localhost:5000/api
\`\`\`

### Endpoints

#### 1. Health Check
**GET** `/health`

Check API health and system status.

**Response:**
\`\`\`json
{
  "status": "healthy",
  "models_loaded": true,
  "symptoms_count": 132,
  "diseases_count": 41
}
\`\`\`

#### 2. Get All Symptoms
**GET** `/symptoms`

Retrieve all recognized symptoms.

**Response:**
\`\`\`json
[
  "fever",
  "headache",
  "cough",
  "fatigue",
  "nausea"
]
\`\`\`

#### 3. Get All Diseases
**GET** `/diseases`

Retrieve all known diseases.

**Response:**
\`\`\`json
[
  "flu",
  "cold",
  "migraine",
  "pneumonia"
]
\`\`\`

#### 4. Predict Disease
**POST** `/predict`

Predict disease based on symptoms.

**Request Body:**
\`\`\`json
{
  "symptoms": ["fever", "headache", "cough", "fatigue"]
}
\`\`\`

**Response:**
\`\`\`json
{
  "predictions": [
    {
      "disease": "flu",
      "confidence": 0.85,
      "description": "Common viral infection affecting respiratory system",
      "precautions": [
        "rest",
        "drink fluids",
        "take medicine",
        "see doctor if severe"
      ],
      "severity_score": 5.2,
      "critical_symptoms": ["fever"],
      "matched_symptoms": ["fever", "headache", "cough", "fatigue"]
    }
  ],
  "emergency": false,
  "overall_risk_score": 4.8,
  "severe_symptoms_present": ["fever"],
  "matched_symptoms": ["fever", "headache", "cough", "fatigue"]
}
\`\`\`

### Error Responses

**400 Bad Request:**
\`\`\`json
{
  "error": "No valid symptoms provided",
  "valid_symptoms": ["fever", "headache", "cough", ...]
}
\`\`\`

**500 Internal Server Error:**
\`\`\`json
{
  "error": "Internal server error"
}
\`\`\`

## 📊 Data Requirements

### Required CSV Files

Your data files should follow these formats:

1. **dataset.csv** - Main training data
   - First column: Disease name
   - Remaining columns: Symptoms (can have up to 17 symptom columns)
   - Missing symptoms should be empty or NaN

2. **symptom_Description.csv** - Disease information
   - Disease: Disease name (must match dataset.csv)
   - Description: Detailed description of the disease

3. **symptom_precaution.csv** - Prevention measures
   - Disease: Disease name
   - Precaution_1 to Precaution_4: Prevention/treatment measures

4. **Symptom-severity.csv** - Symptom weights
   - Symptom: Symptom name (must match symptoms in dataset.csv)
   - weight: Severity score (1-10, where 10 is most severe)

### Sample Data Generation

If you don't have data files, you can generate sample data:

\`\`\`bash
python scripts/generate_sample_data.py
\`\`\`

## 🛠 Development

### Running in Development Mode

\`\`\`bash
# Activate virtual environment
source venv/bin/activate

# Set development environment
export FLASK_CONFIG=development
export FLASK_DEBUG=true

# Run development server
python run.py
\`\`\`

### Code Quality Tools

\`\`\`bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type checking (if using mypy)
mypy app/
\`\`\`

### Pre-commit Hooks

\`\`\`bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
\`\`\`

## 🧪 Testing

### Running Tests

\`\`\`bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
\`\`\`

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test API endpoints and service interactions
- **Model Tests**: Test ML model functionality

### Manual API Testing

\`\`\`bash
# Test with curl
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"symptoms": ["fever", "headache", "cough"]}'

# Test with Python script
python test_api.py
\`\`\`

## 🚀 Deployment

### Production with Gunicorn

\`\`\`bash
# Install gunicorn
pip install gunicorn

# Run production server
gunicorn --bind 0.0.0.0:5000 --workers 4 wsgi:app

# With configuration file
gunicorn --config gunicorn.conf.py wsgi:app
\`\`\`

### Docker Deployment

\`\`\`bash
# Build image
docker build -t medical-diagnosis-api .

# Run container
docker run -p 5000:5000 medical-diagnosis-api

# Using Docker Compose
docker-compose up --build
\`\`\`

### Environment Variables for Production

\`\`\`env
FLASK_CONFIG=production
FLASK_DEBUG=false
SECRET_KEY=your-production-secret-key
CORS_ORIGINS=https://yourdomain.com
LOG_LEVEL=WARNING
\`\`\`

### Cloud Deployment

#### Heroku
\`\`\`bash
# Install Heroku CLI and login
heroku create your-app-name
git push heroku main
\`\`\`

#### AWS/GCP/Azure
- Use the provided Dockerfile
- Set environment variables in your cloud platform
- Ensure data files are accessible (use cloud storage)

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLASK_CONFIG` | Configuration mode | `development` | No |
| `FLASK_DEBUG` | Debug mode | `false` | No |
| `SECRET_KEY` | Flask secret key | Generated | Yes |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5173` | No |
| `MODEL_PATH` | Path to model files | `data/models` | No |
| `DATA_PATH` | Path to data files | `data/raw` | No |
| `CRITICAL_THRESHOLD` | Emergency threshold | `6` | No |
| `MIN_CONFIDENCE` | Minimum prediction confidence | `0.1` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `LOG_FILE` | Log file path | `logs/app.log` | No |

### Configuration Classes

- **DevelopmentConfig**: For local development
- **ProductionConfig**: For production deployment
- **TestingConfig**: For running tests

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
\`\`\`bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change port in run.py
app.run(port=5001)
\`\`\`

#### 2. Missing Data Files
\`\`\`
Error: FileNotFoundError: data/raw/dataset.csv not found
\`\`\`
**Solution**: Ensure all required CSV files are in `data/raw/` directory.

#### 3. Import Errors
\`\`\`
ModuleNotFoundError: No module named 'app'
\`\`\`
**Solution**: Activate virtual environment and ensure PYTHONPATH is set:
\`\`\`bash
source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
\`\`\`

#### 4. Model Training Fails
\`\`\`
Error: Failed to train models
\`\`\`
**Solution**: Check data format and ensure sufficient data samples.

#### 5. CORS Errors
\`\`\`
Access to fetch blocked by CORS policy
\`\`\`
**Solution**: Update `CORS_ORIGINS` in `.env` file with your frontend URL.

### Debug Mode

Enable debug mode for detailed error messages:
\`\`\`bash
export FLASK_DEBUG=true
python run.py
\`\`\`

### Logging

Check logs for detailed error information:
\`\`\`bash
tail -f logs/app.log
\`\`\`

### Health Check

Monitor system health:
\`\`\`bash
curl http://localhost:5000/api/health
\`\`\`

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   \`\`\`bash
   git checkout -b feature/amazing-feature
   \`\`\`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests**
   \`\`\`bash
   pytest
   \`\`\`
6. **Format code**
   \`\`\`bash
   black .
   isort .
   \`\`\`
7. **Commit changes**
   \`\`\`bash
   git commit -m "Add amazing feature"
   \`\`\`
8. **Push to branch**
   \`\`\`bash
   git push origin feature/amazing-feature
   \`\`\`
9. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation for new features
- Use meaningful commit messages
- Keep functions small and focused

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** for machine learning algorithms
- **Flask** for the web framework
- **pandas** for data manipulation
- **Contributors** who helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/el-noir/medical-diagnosis-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/el-noir/medical-diagnosis-api/discussions)
- **Email**: shmusy434@gmail.com

## 🔮 Roadmap

- [ ] Add more ML algorithms (SVM, Neural Networks)
- [ ] Implement user authentication
- [ ] Add rate limiting
- [ ] Create web dashboard
- [ ] Add model versioning
- [ ] Implement A/B testing for models
- [ ] Add real-time monitoring
- [ ] Create mobile app integration

---

**⚠️ Disclaimer**: This API is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

---

Made by [Mudasir Shah](https://github.com/el-noir)
