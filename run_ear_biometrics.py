#!/usr/bin/env python3
"""
Ear Biometrics System Launcher
Main entry point for the Ear Biometrics System
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Import and run the main GUI
    from ear_biometrics_gui import main
    
    print("🚀 Starting Ear Biometrics System...")
    print("📁 Working directory:", current_dir)
    
    # Change to package directory to ensure relative paths work
    os.chdir(current_dir)
    
    # Run the application
    main()
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📦 Make sure all required dependencies are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
