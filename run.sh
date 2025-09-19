#!/bin/bash
echo "🚀 Starting Ear Biometrics System..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python is not installed or not in PATH"
        echo "Please install Python 3.8+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Using Python $PYTHON_VERSION"

# Run the application
$PYTHON_CMD run_ear_biometrics.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application encountered an error"
    read -p "Press Enter to continue..."
fi
