"""
Dataset Structure Diagnostic Tool
Checks and fixes common dataset structure issues
"""

import os
from pathlib import Path

def check_dataset_structure(data_dir):
    """
    Check the dataset structure and provide detailed diagnostics
    """
    print("=" * 70)
    print("DATASET STRUCTURE DIAGNOSTIC")
    print("=" * 70)
    print(f"\nChecking directory: {data_dir}\n")
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Directory '{data_dir}' does not exist!")
        print("\nTo fix:")
        print(f"  1. Create the directory: mkdir -p {data_dir}")
        print(f"  2. Download dataset from Kaggle or create sample data")
        return False
    
    print(f"✓ Directory exists: {data_dir}\n")
    
    # Check what's in the root directory
    root_contents = os.listdir(data_dir)
    print("Root directory contents:")
    for item in root_contents:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            file_count = len([f for f in os.listdir(item_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  📁 {item}/ ({file_count} images)")
        else:
            print(f"  📄 {item}")
    
    print("\n" + "-" * 70)
    
    # Check expected structures
    structures = {
        "Original structure": {
            "paths": ["normal", "potholes"],
            "alt_paths": ["normal", "pothole"]
        },
        "Split structure": {
            "paths": ["train/normal", "train/pothole", "val/normal", "val/pothole", 
                     "test/normal", "test/pothole"]
        }
    }
    
    found_structure = None
    
    for struct_name, struct_info in structures.items():
        print(f"\nChecking for {struct_name}:")
        
        for paths_key in ['paths', 'alt_paths'] if 'alt_paths' in struct_info else ['paths']:
            paths = struct_info.get(paths_key, struct_info['paths'])
            all_exist = True
            total_images = 0
            
            for path in paths:
                full_path = os.path.join(data_dir, path)
                exists = os.path.exists(full_path)
                
                if exists:
                    image_files = [f for f in os.listdir(full_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    count = len(image_files)
                    total_images += count
                    print(f"  ✓ {path}/ - {count} images")
                else:
                    all_exist = False
                    print(f"  ❌ {path}/ - NOT FOUND")
            
            if all_exist and total_images > 0:
                found_structure = struct_name
                print(f"\n✓ Found valid {struct_name}! Total images: {total_images}")
                break
        
        if found_structure:
            break
    
    print("\n" + "=" * 70)
    
    if not found_structure:
        print("\n❌ NO VALID DATASET STRUCTURE FOUND!\n")
        print("Expected one of these structures:")
        print("\nOption 1 - Original structure:")
        print("  data/pothole-detection-dataset/")
        print("    ├── normal/")
        print("    │   ├── image1.jpg")
        print("    │   ├── image2.jpg")
        print("    │   └── ...")
        print("    └── potholes/  (or 'pothole/')")
        print("        ├── image1.jpg")
        print("        ├── image2.jpg")
        print("        └── ...")
        print("\nOption 2 - Split structure:")
        print("  data/pothole-detection-dataset/")
        print("    ├── train/")
        print("    │   ├── normal/")
        print("    │   └── pothole/")
        print("    ├── val/")
        print("    │   ├── normal/")
        print("    │   └── pothole/")
        print("    └── test/")
        print("        ├── normal/")
        print("        └── pothole/")
        print("\n" + "=" * 70)
        print("\nSOLUTIONS:\n")
        print("1. If you don't have data yet:")
        print("   python create_sample_data.py --output_dir data/pothole-detection-dataset")
        print("\n2. If you have data elsewhere, specify the correct path:")
        print("   python main.py --mode train --data_dir /path/to/your/data")
        print("\n3. Download the Kaggle dataset:")
        print("   https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset")
        
        return False
    else:
        print(f"\n✅ SUCCESS: Valid dataset structure found!")
        print(f"\nYou can proceed with training:")
        print(f"  python main.py --mode train --data_dir {data_dir} --epochs 20")
        return True


def quick_fix_structure(data_dir):
    """
    Attempt to automatically fix common structure issues
    """
    print("\n" + "=" * 70)
    print("ATTEMPTING AUTO-FIX...")
    print("=" * 70)
    
    # Check if split directories exist but are empty
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        train_files = os.listdir(train_dir)
        if len(train_files) == 0 or all(os.path.isdir(os.path.join(train_dir, f)) and 
                                        len(os.listdir(os.path.join(train_dir, f))) == 0 
                                        for f in train_files):
            print("\n⚠️  Found empty split directories. Removing them...")
            import shutil
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(data_dir, split)
                if os.path.exists(split_path):
                    shutil.rmtree(split_path)
                    print(f"  Removed: {split}/")
            print("\n✓ Empty directories removed. The training script will recreate them.")
            print("  Run the diagnostic again to verify.")
            return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check dataset structure")
    parser.add_argument('--data_dir', type=str, 
                       default='data/pothole-detection-dataset',
                       help='Path to dataset directory')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to auto-fix common issues')
    
    args = parser.parse_args()
    
    is_valid = check_dataset_structure(args.data_dir)
    
    if not is_valid and args.fix:
        if quick_fix_structure(args.data_dir):
            print("\nRe-checking after fix...")
            check_dataset_structure(args.data_dir)