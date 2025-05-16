#!/bin/bash

# "splitter_type": ["recursive_character", "token"],
# Create tests directory if it doesn't exist
mkdir -p tests

# Create output directory
mkdir -p output/tests

# Ensure test config exists
if [ ! -f tests/configs/single_config.json ]; then
  echo "Error: tests/configs/config.json not found"
  exit 1
fi

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run tests with the sample configuration
python src/rag_refactored.py test \
  --config tests/configs/single_config.json \
  --output output/tests/test_results_${TIMESTAMP}.csv \
  --dossier_dir data/space_data/dossiers \
  --html_dir data/html \
  --max_workers 4