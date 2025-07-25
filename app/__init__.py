"""Flask application factory."""
import os
from flask import Flask
from app.extensions import cors
from app.config import Config


def create_app(config_class=Config):
    """Create Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    cors.init_app(app)
    
    # Register blueprints
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    from app.api.errors import register_error_handlers
    register_error_handlers(app)
    
    return app
