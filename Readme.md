# Medical Diagnosis API

A Flask-based REST API for medical diagnosis using machine learning models.

## Features

- Disease prediction based on symptoms
- Multiple ML models (Random Forest, Gradient Boosting)
- Emergency condition detection
- Comprehensive health monitoring
- Professional project structure
- Docker support
- Comprehensive testing

## Project Structure

\`\`\`
medical-diagnosis-api/
├── app/                    # Application package
│   ├── api/               # API routes and error handlers
│   ├── models/            # ML models and data structures
│   ├── services/          # Business logic layer
│   └── utils/             # Utility functions
├── data/                  # Data files
│   ├── raw/              # Raw CSV datasets
│   └── models/           # Trained model files
├── tests/                 # Test suite
├── scripts/              # Utility scripts
└── logs/                 # Application logs
\`\`\`

## Setup

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd medical-diagnosis-api
   \`\`\`

2. **Create virtual environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Set up environment variables**
   \`\`\`bash
   cp .env.example .env
   # Edit .env with your configuration
   \`\`\`

5. **Prepare data**
   - Place your CSV files in `data/raw/`:
     - `dataset.csv`
     - `symptom_Description.csv`
     - `symptom_precaution.csv`
     - `Symptom-severity.csv`

6. **Train models** (optional - models will be trained automatically on first run)
   \`\`\`bash
   python scripts/train_models.py
   \`\`\`

7. **Run the application**
   \`\`\`bash
   flask run
   \`\`\`

## API Endpoints

### Health Check
\`\`\`
GET /api/health
\`\`\`

### Get All Symptoms
\`\`\`
GET /api/symptoms
\`\`\`

### Get All Diseases
\`\`\`
GET /api/diseases
\`\`\`

### Predict Disease
\`\`\`
POST /api/predict
Content-Type: application/json

{
  "symptoms": ["fever", "headache", "cough"]
}
\`\`\`

## Development

### Running Tests
\`\`\`bash
pip install -r requirements-dev.txt
pytest
\`\`\`

### Code Formatting
\`\`\`bash
black .
isort .
flake8 .
\`\`\`

### Docker Development
\`\`\`bash
docker-compose up --build
\`\`\`

## Production Deployment

### Using Docker
\`\`\`bash
docker build -t medical-diagnosis-api .
docker run -p 5000:5000 medical-diagnosis-api
\`\`\`

### Using Gunicorn
\`\`\`bash
gunicorn --bind 0.0.0.0:5000 --workers 4 wsgi:app
\`\`\`

## Configuration

The application uses environment variables for configuration. See `.env.example` for available options.

## License

[Your License Here]
