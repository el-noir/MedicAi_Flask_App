"""Test configuration."""
import pytest
import tempfile
import os
from app import create_app
from app.config import TestingConfig


@pytest.fixture
def app():
    """Create application for testing."""
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    
    class TestConfig(TestingConfig):
        MODEL_PATH = os.path.join(temp_dir, 'models')
        DATA_PATH = 'data/raw'  # Use real data path for tests
        LOG_FILE = os.path.join(temp_dir, 'test.log')
    
    app = create_app(TestConfig)
    
    with app.app_context():
        yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner."""
    return app.test_cli_runner()
