import os
import uuid
from PIL import Image

# 1. Define Paths
ROOT_DIR = r"C:\Users\acer\Desktop\BCA PROJECT\plant_disease_recognition\data\raw"
SOURCE_DIRS = [
    os.path.join(ROOT_DIR, r"Dataset\COFEE LEAF"),
    os.path.join(ROOT_DIR, r"Dataset\PLANT VILLAGE"),
    os.path.join(ROOT_DIR, r"Dataset\PLANTDOC"),
    os.path.join(ROOT_DIR, r"Dataset\RICE LEAF")
]
DEST_DIR = os.path.join(ROOT_DIR, "unified_dataset")

# 2. Safely Resize Function
def resize_and_save(src_path, dest_path, target_size=(256, 256)):
    """Resizes the image to exact CNN dimensions to prevent training memory crashes."""
    try:
        with Image.open(src_path) as img:
            img = img.convert('RGB')
            resized_image = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_image.save(dest_path, format="JPEG", quality=90)
            return True
    except Exception as e:
        # Silently skip corrupted images
        return False

def merge_all_datasets():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    total_copied = 0
    supported_formats = ('.jpg', '.jpeg', '.png')

    print("--- Starting Deep-Search Dataset Merge ---")
    
    for dataset_dir in SOURCE_DIRS:
        if not os.path.exists(dataset_dir):
            print(f"Warning: Directory not found -> {dataset_dir}")
            continue
            
        print(f"\nScanning: {os.path.basename(dataset_dir)}")
        
        # os.walk recursively digs into every nested folder automatically
        for root, dirs, files in os.walk(dataset_dir):
            for filename in files:
                if filename.lower().endswith(supported_formats):
                    src_file = os.path.join(root, filename)
                    
                    # The class name is exactly the folder directly containing the image
                    raw_class_name = os.path.basename(root)
                    
                    # Standardize spaces to underscores (e.g., 'Apple leaf' -> 'Apple_leaf')
                    clean_class_name = raw_class_name.strip().replace(" ", "_")
                    dest_class_path = os.path.join(DEST_DIR, clean_class_name)
                    
                    if not os.path.exists(dest_class_path):
                        os.makedirs(dest_class_path)
                        print(f"  [+] Created new class: {clean_class_name}")
                        
                    # Generate a unique UUID filename to prevent naming collisions
                    unique_filename = f"{uuid.uuid4().hex}.jpg"
                    dest_file = os.path.join(dest_class_path, unique_filename)
                    
                    if resize_and_save(src_file, dest_file):
                        total_copied += 1
                        
                        if total_copied % 2500 == 0:
                            print(f"  ... Processed {total_copied} images so far ...")

    print(f"\nSUCCESS! Total images merged and resized: {total_copied}")
    print(f"Your final dataset is ready at: {DEST_DIR}")

if __name__ == "__main__":
    merge_all_datasets()