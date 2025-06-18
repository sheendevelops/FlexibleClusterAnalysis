#!/usr/bin/env python3
"""
Create a downloadable package of the AISapien project
This script generates a zip file containing all necessary source code and documentation
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_package():
    """Create a downloadable package with all AISapien files"""
    
    # Define package name
    package_name = "aisapien_complete_package.zip"
    
    # Files and directories to include
    files_to_include = [
        "app.py",
        "setup_database.py", 
        "README.md",
        "INSTALL.md",
        "replit.md",
        ".streamlit/config.toml",
        "models/master_model.py",
        "models/conscience_model.py",
        "models/logic_model.py", 
        "models/personality_model.py",
        "utils/database_helper.py",
        "utils/openai_helper.py",
        "utils/content_filter.py",
        "utils/emotion_detector.py",
        "utils/web_scraper.py"
    ]
    
    # Create requirements.txt content
    requirements_content = """streamlit>=1.45.1
openai>=1.0.0
psycopg2-binary>=2.9.10
sqlalchemy>=2.0.41
requests>=2.32.4
trafilatura>=2.0.0
greenlet>=3.2.3"""
    
    # Create .env template
    env_template = """# AISapien Environment Configuration
# Copy this file to .env and fill in your actual values

# OpenAI API Key (required)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database URL (required)
# Format: postgresql://username:password@host:port/database
DATABASE_URL=postgresql://username:password@localhost:5432/aisapien

# Alternative: Individual PostgreSQL settings
# PGHOST=localhost
# PGPORT=5432
# PGUSER=postgres
# PGPASSWORD=your_password
# PGDATABASE=aisapien"""

    try:
        with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add main files
            for file_path in files_to_include:
                if os.path.exists(file_path):
                    zipf.write(file_path, file_path)
                    print(f"Added: {file_path}")
                else:
                    print(f"Warning: {file_path} not found")
            
            # Add requirements.txt
            zipf.writestr("requirements.txt", requirements_content)
            print("Added: requirements.txt")
            
            # Add .env template
            zipf.writestr(".env.template", env_template)
            print("Added: .env.template")
            
            # Create empty directories if needed
            zipf.writestr("data/.gitkeep", "# This directory stores legacy JSON files\n")
            print("Added: data/ directory")
        
        print(f"\n✅ Package created successfully: {package_name}")
        print(f"📦 Package size: {os.path.getsize(package_name) / 1024:.1f} KB")
        
        return package_name
        
    except Exception as e:
        print(f"❌ Error creating package: {e}")
        return None

if __name__ == "__main__":
    package_file = create_package()
    if package_file:
        print(f"\n🎉 Your AISapien package is ready!")
        print(f"📁 File: {package_file}")
        print(f"📋 See INSTALL.md for setup instructions")