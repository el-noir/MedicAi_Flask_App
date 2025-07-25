"""WSGI entry point for production deployment."""
import os
from app import create_app
from app.config import config
from app.utils.helpers import setup_logging

# Get configuration from environment
config_name = os.environ.get('FLASK_CONFIG', 'production')
app = create_app(config[config_name])

# Setup logging
setup_logging(
    log_level=app.config.get('LOG_LEVEL', 'INFO'),
    log_file=app.config.get('LOG_FILE')
)

if __name__ == "__main__":
    app.run()
