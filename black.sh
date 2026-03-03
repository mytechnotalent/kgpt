#!/bin/bash

# Format Python files with black
echo "Running black on Python files..."
black --quiet *.py
# Format Jupyter notebooks with black (requires black[jupyter])
echo "Running black on Jupyter notebooks..."
black --quiet *.ipynb
echo "Done! All Python files and notebooks formatted."
