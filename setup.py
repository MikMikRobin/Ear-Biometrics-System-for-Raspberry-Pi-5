#!/usr/bin/env python3
"""
Setup script for Ear Biometrics System
Installs required dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("💻 Running on CPU (no GPU detected)")
    except ImportError:
        print("⚠️ PyTorch not installed yet")

def main():
    """Main setup function"""
    print("🔧 Setting up Ear Biometrics System...")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path(__file__).parent
    print(f"📁 Package directory: {current_dir}")
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed!")
        return False
    
    # Check GPU availability
    check_gpu()
    
    # Verify model files
    model_files = [
        "best.pt",
        "excellent_ear_model_best.pth", 
        "efficientnet_b4_ultimate_best.pth"
    ]
    
    print("\n📋 Checking model files...")
    for model_file in model_files:
        if (current_dir / model_file).exists():
            size = (current_dir / model_file).stat().st_size / (1024*1024)  # MB
            print(f"✅ {model_file} ({size:.1f} MB)")
        else:
            print(f"⚠️ {model_file} not found")
    
    # Check database
    db_file = current_dir / "ear_biometrics_v3.db"
    if db_file.exists():
        print(f"✅ Database: ear_biometrics_v3.db")
    else:
        print("ℹ️ Database will be created on first run")
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\n📖 To run the application:")
    print("   python run_ear_biometrics.py")
    print("\n💡 For help:")
    print("   See USAGE_GUIDE.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
