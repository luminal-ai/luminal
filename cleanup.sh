#!/bin/bash

echo "Starting cleanup..."

# Function to safely remove files
safe_remove() {
    find "$1" -type f -name "$2" -print -delete 2>/dev/null || true
}

# Function to safely remove directories
safe_remove_dir() {
    find "$1" -type d -name "$2" -print -exec rm -rf {} + 2>/dev/null || true
}

echo "Cleaning build artifacts..."
# Clean each Cargo project
find . -name "Cargo.toml" -execdir cargo clean \;

echo "Removing temporary files..."
# Remove temporary files
safe_remove . ".DS_Store"
safe_remove . "*.pyc"
safe_remove_dir . "__pycache__"
safe_remove . "*.swp"
safe_remove . "*.swo"
safe_remove . "*~"
safe_remove . "*.bak"
safe_remove . "*.orig"

echo "Cleanup complete!"
