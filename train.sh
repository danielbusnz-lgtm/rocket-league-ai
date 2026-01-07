#!/bin/bash

# Training wrapper script for Rocket League AI
# This ensures proper terminal handling

cd /home/dan/Projects/Personal/rocket-league-ai
source venv/bin/activate

echo "Starting Rocket League AI Training..."
echo ""
echo "TensorBoard is available at: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop training and save checkpoint"
echo ""

# Run with proper PTY allocation
python -u src/training/train.py
