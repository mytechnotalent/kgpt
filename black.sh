#!/bin/bash

# Format Python files with black
echo "Running black on Python files..."
black --quiet *.py
echo "Done! All Python files and notebooks formatted."
