#!/bin/bash
set -e

# specific check for python3-venv
if ! python3 -c "import venv" &> /dev/null; then
    echo "Error: Python venv module not found."
    echo "Please run: sudo apt update && sudo apt install -y python3-full"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv --system-site-packages
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. To activate run: source venv/bin/activate"
