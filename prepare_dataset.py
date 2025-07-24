# prepare_dataset.py

import os
import shutil
import glob
import random

# --- IMPORTANT: UPDATE THESE PATHS ---
# Path to the folder where you unzipped the CASIA 2.0 dataset
# SOURCE_DATA_DIR = "C:\Users\Uddhav\Documents\document-authenticator\CASIA" 
SOURCE_DATA_DIR = r"C:\Users\Uddhav\Documents\document-authenticator\CASIA"
# ------------------------------------

# Destination paths (should be correct if you run this from your project root)
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
SPLIT_RATIO = 0.9 # 90% for training, 10% for validation

def split_and_copy_files(source_folder_name, class_name):
    """
    Finds all images in the source folder, splits them into train/val sets,
    and copies them to the destination folders.
    """
    print(f"Processing class: {class_name}")

    # Create destination directories if they don't exist
    train_dest = os.path.join(TRAIN_DIR, class_name)
    val_dest = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)

    # Get a list of all image files
    source_path = os.path.join(SOURCE_DATA_DIR, source_folder_name)
    
    # CASIA dataset has multiple file extensions, so we search for common ones
    image_files = glob.glob(os.path.join(source_path, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(source_path, "*.png")))
    image_files.extend(glob.glob(os.path.join(source_path, "*.bmp")))
    image_files.extend(glob.glob(os.path.join(source_path, "*.tif")))

    if not image_files:
        print(f"Error: No image files found in {source_path}. Check your SOURCE_DATA_DIR path.")
        return

    # Shuffle the files randomly
    random.shuffle(image_files)
    
    # Calculate the split point
    split_point = int(len(image_files) * SPLIT_RATIO)
    
    # Split the files
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]

    # Copy files to the training directory
    print(f"Copying {len(train_files)} files to {train_dest}...")
    for file_path in train_files:
        shutil.copy(file_path, train_dest)

    # Copy files to the validation directory
    print(f"Copying {len(val_files)} files to {val_dest}...")
    for file_path in val_files:
        shutil.copy(file_path, val_dest)
        
    print(f"Finished processing {class_name}.\n")


if __name__ == "__main__":
    # Process the authentic images
    split_and_copy_files(source_folder_name="Au", class_name="authentic")
    
    # Process the tampered images
    split_and_copy_files(source_folder_name="Tp", class_name="tampered")

    print("Dataset preparation is complete!")