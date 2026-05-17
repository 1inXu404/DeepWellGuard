#!/usr/bin/env python3
"""Cleanup script to keep only the latest models in results/models.

It scans results/models/*, identifies the model type in each timestamp folder,
keeps only the newest folder for each model type, and safely deletes the rest.
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    models_dir = Path("results/models")
    if not models_dir.exists():
        print(f"Directory {models_dir} does not exist. Nothing to clean.")
        return

    # Dictionary to keep track of the latest directory for each model file type
    # e.g., 'lstmmodel.pt' -> Path('results/models/20260517_153236')
    latest_models = {}
    
    # List all subdirectories (which are timestamp folders)
    # Sort them alphabetically (which means chronological order for YYYYMMDD_HHMMSS)
    subdirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(subdirs)} timestamp directories in {models_dir}.")

    # Identify the latest directory for each model file
    for d in subdirs:
        # Check what .pt files are inside this directory
        pt_files = list(d.glob("*.pt"))
        if pt_files:
            # Assuming one model file per timestamp directory
            model_name = pt_files[0].name
            latest_models[model_name] = d

    # Print out what we are keeping
    directories_to_keep = set(latest_models.values())
    print("\n--- Keeping the latest models ---")
    for model_name, d in latest_models.items():
        print(f"[{model_name}]: {d.name}")

    if not directories_to_keep:
        print("\nNo model files found. Exiting.")
        return

    # Delete all other directories
    directories_to_delete = [d for d in subdirs if d not in directories_to_keep]
    
    print(f"\n--- Deleting {len(directories_to_delete)} old directories ---")
    for d in directories_to_delete:
        print(f"Deleting: {d}")
        shutil.rmtree(d)
        
    print("\nCleanup complete.")

if __name__ == "__main__":
    main()
