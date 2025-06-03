#!/usr/bin/env python3
"""
Installation Verification Script for Flickd Fashion Analysis Pipeline
Run this script to verify all dependencies are correctly installed.
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def check_spacy_model():
    """Check if spaCy English model is installed"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model: Available")
        return True
    except (ImportError, OSError):
        print("⚠️  spaCy English model: Not installed (optional)")
        print("   Install with: python -m spacy download en_core_web_sm")
        return False

def check_data_structure():
    """Check if required directories and files exist"""
    required_dirs = ['data', 'data/videos', 'src']
    required_files = ['main.py', 'requirements.txt', 'data/images.csv', 'data/vibes_list.json']
    
    print("\n📁 Checking project structure...")
    
    all_good = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Main verification function"""
    print("🔍 Flickd Fashion Analysis Pipeline - Installation Verification")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check required packages
    print("\n📦 Checking required packages...")
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('transformers', 'transformers'),
        ('ultralytics', 'ultralytics'),
        ('opencv-python', 'cv2'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('faiss-cpu', 'faiss'),
        ('Pillow', 'PIL'),
        ('requests', 'requests'),
        ('tqdm', 'tqdm'),
        ('pyyaml', 'yaml'),
        ('spacy', 'spacy'),
    ]
    
    packages_ok = all(check_package(pkg, imp) for pkg, imp in packages)
    
    # Check spaCy model
    print("\n🔤 Checking NLP models...")
    spacy_model_ok = check_spacy_model()
    
    # Check project structure
    structure_ok = check_data_structure()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    if python_ok and packages_ok and structure_ok:
        print("🎉 ALL CHECKS PASSED! Your installation is ready.")
        print("\n🚀 You can now run: python main.py")
    else:
        print("⚠️  Some issues found. Please fix the above errors.")
        
        if not python_ok:
            print("   - Install Python 3.8+ from https://python.org")
        if not packages_ok:
            print("   - Run: pip install -r requirements.txt")
        if not spacy_model_ok:
            print("   - Run: python -m spacy download en_core_web_sm")
        if not structure_ok:
            print("   - Ensure all required files and directories are present")
    
    print("\n📖 For detailed instructions, see README.md")

if __name__ == "__main__":
    main()
