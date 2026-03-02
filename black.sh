#!/bin/bash

# Format all Python files and Jupyter notebooks with black
echo "Running black on Python files and Jupyter notebooks..."
black --quiet *.py *.ipynb
echo "Done! All Python files and notebooks formatted."
