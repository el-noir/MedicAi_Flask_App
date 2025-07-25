"""Development server runner."""
import os
from app import create_app
from app.config import config
from app.utils.helpers import setup_logging, ensure_directories

def main():
    """Run the development server."""
    # Ensure required directories exist
    ensure_directories('data/raw', 'data/models', 'logs')
    
    # Get configuration
    config_name = os.environ.get('FLASK_CONFIG', 'development')
    app = create_app(config[config_name])
    
    # Setup logging
    setup_logging(
        log_level=app.config.get('LOG_LEVEL', 'INFO'),
        log_file=app.config.get('LOG_FILE')
    )
    
    print("🏥 Medical Diagnosis API Starting...")
    print(f"📊 Data path: {app.config.get('DATA_PATH')}")
    print(f"🤖 Model path: {app.config.get('MODEL_PATH')}")
    print(f"🌐 CORS origins: {app.config.get('CORS_ORIGINS')}")
    print("🚀 Server starting on http://localhost:5000")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config.get('DEBUG', False)
    )

if __name__ == '__main__':
    main()
