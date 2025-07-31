#!/usr/bin/env python3
"""
Flask App Runner for SPY Options  Dashboard
"""

import os
import sys
from app import app

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static', 'static/css', 'static/js']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def setup_environment():
    """Set up the Flask environment"""
    # Set Flask environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Create directories
    create_directories()
    
    print("Environment setup complete!")

if __name__ == '__main__':
    print("SPY 0DTE Options Analysis Dashboard - Flask Version")
    print("=" * 55)
    
    try:
        setup_environment()
        
        print("\nStarting Flask application...")
        print("Dashboard will be available at: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 55)
        
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\nShutting down Flask application...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError starting Flask application: {e}")
        sys.exit(1)
