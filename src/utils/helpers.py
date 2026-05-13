"""
Helper utility functions.
"""

import os
import json

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
