import os
import random
import pandas as pd

# The retained classes we want to show
# 0: Normal
# 1, 3, 4, 5, 6, 9: Abnormalities
SEQUENCE = [0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 9, 0]

# Number of rows to sample per segment (e.g., 1200 rows = 20 mins if 1Hz)
# Downsample rate is 2, so 1200 raw rows = 600 model rows = 5 windows.
ROWS_PER_SEGMENT = 1200

def get_random_file(class_label):
    folder = f"3w_dataset_2.0.0/{class_label}"
    if not os.path.exists(folder):
        return None
    files = [f for f in os.listdir(folder) if f.endswith('.parquet')]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

def create_demo():
    print("Building demo test file...")
    chunks = []
    
    for cls in SEQUENCE:
        file_path = get_random_file(cls)
        if not file_path:
            print(f"Warning: No files found for class {cls}. Skipping.")
            continue
            
        print(f"Loading {file_path} for Class {cls}...")
        df = pd.read_parquet(file_path)
        
        # Reset index to integer so slicing works correctly
        df = df.reset_index(drop=True)
        
        # We want to make sure we grab a segment that actually contains the target class
        # if it's an abnormal file (class > 0). Some files have mixed 0 and X.
        if cls > 0:
            abnormal_indices = df[df['class'] == cls].index
            if len(abnormal_indices) > 0:
                start_idx = abnormal_indices[0]
                # Try to get a chunk around the first abnormal index
                start = max(0, start_idx - 100)
                end = min(len(df), start + ROWS_PER_SEGMENT)
                chunk = df.iloc[start:end].copy()
            else:
                # Fallback to random if exact class label not found locally
                start = random.randint(0, max(0, len(df) - ROWS_PER_SEGMENT))
                chunk = df.iloc[start:start+ROWS_PER_SEGMENT].copy()
        else:
            # For normal (0), just pick a random chunk
            start = random.randint(0, max(0, len(df) - ROWS_PER_SEGMENT))
            chunk = df.iloc[start:start+ROWS_PER_SEGMENT].copy()
            
        # Ensure the chunk has the right length by taking exactly ROWS_PER_SEGMENT if possible
        if len(chunk) > ROWS_PER_SEGMENT:
            chunk = chunk.head(ROWS_PER_SEGMENT)
            
        chunks.append(chunk)

    if not chunks:
        print("Error: No data chunks created.")
        return

    # Concatenate all chunks vertically
    final_df = pd.concat(chunks, ignore_index=True)
    
    # Save the output
    out_dir = "uploads"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "demo_test_sequence.parquet")
    
    # Use PyArrow to write parquet
    final_df.to_parquet(out_file, engine="pyarrow")
    print(f"\nSuccess! Created demo file: {out_file}")
    print(f"Total rows: {len(final_df)}")
    print(f"This file simulates the sequence: Normal -> 1 -> Normal -> 3 -> Normal -> 4 -> Normal -> 5 -> Normal -> 6 -> Normal -> 9 -> Normal")
    print("You can upload this file directly to the Web UI for testing.")

if __name__ == "__main__":
    create_demo()
