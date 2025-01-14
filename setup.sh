#!/bin/bash

set -e
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-full python3-venv python3-pip ffmpeg

echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python packages..."
pip install openai-whisper torch requests python-dotenv pytz

echo "Setup complete!"
echo "Activate the virtual environment with: source venv/bin/activate"
