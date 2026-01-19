#!/bin/bash

# TCR-seq Workflow Test Script
# This script tests the workflow with example data

echo "Starting TCR-seq workflow test..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

# Check if required directories exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

if [ ! -d "results" ]; then
    echo "Creating results directory..."
    mkdir -p results
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Run the workflow with example data
echo "Running TCR-seq analysis workflow..."
python3 main.py \
    --data_dir data/ \
    --config config/config.yaml \
    --output_dir results/ \
    --mode full \
    --verbose

if [ $? -eq 0 ]; then
    echo "✅ Workflow completed successfully!"
    echo "Results are available in the 'results/' directory"
    echo ""
    echo "Generated files:"
    ls -la results/
    echo ""
    echo "Plots generated:"
    ls -la results/plots/ 2>/dev/null || echo "No plots directory found"
else
    echo "❌ Workflow failed. Check the logs for details."
    exit 1
fi

echo ""
echo "Test completed!"