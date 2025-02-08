#!/bin/bash

echo "Checking if python3-venv is installed..."
if ! dpkg -s python3-venv &> /dev/null; then
    echo "python3-venv is not installed. Installing..."
    sudo apt update && sudo apt install -y python3-venv
else
    echo "python3-venv is already installed."
fi

echo "Creating virtual environment..."
python3 -m venv IBEAvenv

echo "Activating virtual environment..."
source IBEAvenv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# Check if pyqt5 is installed, it is not necessary for IBEA and breaks the venv installer when attempting to install it.
echo "Checking if pyqt5 is installed..."
python -c "import PyQt5" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "pyqt5 is already installed. Skipping installation."
else
    # echo "pyqt5 is not installed. Attempting installation..."
    # pip install pyqt5 || { echo "Failed to install pyqt5. Skipping."; }
    echo "Skipping pyqt5, it is required to see the spike monitoring or other GUI, disabled in this IBEA onboard version of Bioemus because breaks the code (no pre-built wheel for this package with this python version)"
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Virtual environment setup complete! Run 'source IBEAvenv/bin/activate' to activate."
