#!/bin/bash
# Script to run GUI applications in WSL2 with virtual display

# Start virtual X server if not already running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting virtual X server..."
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 2
fi

# Set display environment and Qt platform
export DISPLAY=:99
export QT_QPA_PLATFORM=offscreen

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rtv

echo "Environment setup complete. Display: $DISPLAY"
echo "Running rtl_demo.py..."

# Run the application
python rtl_demo.py
