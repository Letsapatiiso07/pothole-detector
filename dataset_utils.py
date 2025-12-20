"""
Dataset Integration Utilities
Tools for converting and merging multiple pothole datasets
"""

import os
import shutil
import json
import cv2
import numpy as np
from glob import glob
from pathlib import Path

def convert_yolo_to_binary(yolo_dataset_path, output_path, pothole_class_id=0):
    """
    Convert YOLO format dataset to binary classification (normal vs pothole).
    
    Args:
        yolo_dataset_path: Path to YOLO dataset with images/ and labels/ folders
        output_path: Output directory for binary classification
        pothole_class_id: Class ID for pothole in YOLO labels (default: 0)
    """
    print(f"Converting YOLO dataset from {yolo_dataset_path} to {output_path}")
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'pothole'), exist_ok=True)
    
    # Process train, val, test splits
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(yolo_dataset_path, 'images', split)
        labels_dir = os.path.join(yolo_dataset_path, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"Skipping {split} (not found)")
            continue
        
        print(f"Processing {split} split...")
        
        for img_file in glob(os.path.join(images_dir, '*.jpg')) + \
                       glob(os.path.join(images_dir, '*.png')):
            
            img_name = Path(img_file).stem
            label_file = os.path.join(labels_dir, f"{img_name}.txt")
            
            # Check if image has pothole
            has_pothole = False
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if class_id == pothole_class_id:
                            has_pothole = True
                            break
            
            # Copy to appropriate folder
            if has_pothole:
                dst = os.path.join(output_path, 'pothole', Path(img_file).name)
            else:
                dst = os.path.join(output_path, 'normal', Path(img_file).name)
            
            shutil.copy2(img_file, dst)
    
    print(f"Conversion complete!")
    print(f"Normal images: {len(os.listdir(os.path.join(output_path, 'normal')))}")
    print(f"Pothole images: {len(os.listdir(os.path.join(output_path, 'pothole')))}")


def convert_coco_to_binary(coco_json_path, images_path, output_path, pothole_category_name='pothole'):
    """
    Convert COCO format dataset to binary classification.
    
    Args:
        coco_json_path: Path to COCO annotations JSON file
        images_path: Path to images directory
        output_path: Output directory for binary classification
        pothole_category_name: Category name for pothole in COCO annotations
    """
    print(f"Converting COCO dataset from {coco_json_path} to {output_path}")
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'pothole'), exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get pothole category ID
    pothole_cat_id = None
    for cat in coco_data['categories']:
        if cat['name'].lower() == pothole_category_name.lower():
            pothole_cat_id = cat['id']
            break
    
    if pothole_cat_id is None:
        raise ValueError(f"Category '{pothole_category_name}' not found in COCO annotations")
    
    # Build image to annotations mapping
    image_has_pothole = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if ann['category_id'] == pothole_cat_id:
            image_has_pothole[img_id] = True
    
    # Copy images to appropriate folders
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        src_path = os.path.join(images_path, img_filename)
        
        if not os.path.exists(src_path):
            continue
        
        if img_id in image_has_pothole:
            dst_path = os.path.join(output_path, 'pothole', img_filename)
        else:
            dst_path = os.path.join(output_path, 'normal', img_filename)
        
        shutil.copy2(src_path, dst_path)
    
    print(f"Conversion complete!")
    print(f"Normal images: {len(os.listdir(os.path.join(output_path, 'normal')))}")
    print(f"Pothole images: {len(os.listdir(os.path.join(output_path, 'pothole')))}")


def merge_datasets(dataset_paths, output_path):
    """
    Merge multiple binary classification datasets into one.
    
    Args:
        dataset_paths: List of paths to datasets with 'normal' and 'pothole' folders
        output_path: Output directory for merged dataset
    """
    print(f"Merging {len(dataset_paths)} datasets into {output_path}")
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'pothole'), exist_ok=True)
    
    total_normal = 0
    total_pothole = 0
    
    for i, dataset_path in enumerate(dataset_paths):
        print(f"Processing dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
        
        for class_name in ['normal', 'pothole', 'potholes']:  # Handle both naming conventions
            src_dir = os.path.join(dataset_path, class_name)
            if not os.path.exists(src_dir):
                continue
            
            # Normalize class name
            dst_class = 'pothole' if class_name == 'potholes' else class_name
            dst_dir = os.path.join(output_path, dst_class)
            
            # Copy images with dataset prefix to avoid conflicts
            for img_file in glob(os.path.join(src_dir, '*')):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                filename = Path(img_file).name
                new_filename = f"ds{i}_{filename}"
                dst_path = os.path.join(dst_dir, new_filename)
                
                shutil.copy2(img_file, dst_path)
                
                if dst_class == 'normal':
                    total_normal += 1
                else:
                    total_pothole += 1
    
    print(f"\nMerge complete!")
    print(f"Total normal images: {total_normal}")
    print(f"Total pothole images: {total_pothole}")
    print(f"Output directory: {output_path}")


def balance_dataset(dataset_path, method='oversample'):
    """
    Balance dataset by oversampling minority class or undersampling majority class.
    
    Args:
        dataset_path: Path to dataset with 'normal' and 'pothole' folders
        method: 'oversample' or 'undersample'
    """
    normal_dir = os.path.join(dataset_path, 'normal')
    pothole_dir = os.path.join(dataset_path, 'pothole')
    
    normal_images = glob(os.path.join(normal_dir, '*'))
    pothole_images = glob(os.path.join(pothole_dir, '*'))
    
    print(f"Before balancing:")
    print(f"  Normal: {len(normal_images)}")
    print(f"  Pothole: {len(pothole_images)}")
    
    if method == 'oversample':
        # Oversample minority class
        if len(normal_images) < len(pothole_images):
            minority_images = normal_images
            minority_dir = normal_dir
            target_count = len(pothole_images)
        else:
            minority_images = pothole_images
            minority_dir = pothole_dir
            target_count = len(normal_images)
        
        # Duplicate minority images
        while len(glob(os.path.join(minority_dir, '*'))) < target_count:
            for img_path in minority_images:
                if len(glob(os.path.join(minority_dir, '*'))) >= target_count:
                    break
                
                filename = Path(img_path).stem
                ext = Path(img_path).suffix
                new_filename = f"{filename}_dup{np.random.randint(1000, 9999)}{ext}"
                dst_path = os.path.join(minority_dir, new_filename)
                
                # Copy with slight augmentation
                img = cv2.imread(img_path)
                if img is not None:
                    # Random horizontal flip
                    if np.random.rand() > 0.5:
                        img = cv2.flip(img, 1)
                    cv2.imwrite(dst_path, img)
    
    elif method == 'undersample':
        # Undersample majority class
        if len(normal_images) > len(pothole_images):
            majority_images = normal_images
            target_count = len(pothole_images)
        else:
            majority_images = pothole_images
            target_count = len(normal_images)
        
        # Randomly remove excess images
        np.random.shuffle(majority_images)
        for img_path in majority_images[target_count:]:
            os.remove(img_path)
    
    # Recount
    normal_images = glob(os.path.join(normal_dir, '*'))
    pothole_images = glob(os.path.join(pothole_dir, '*'))
    
    print(f"\nAfter balancing ({method}):")
    print(f"  Normal: {len(normal_images)}")
    print(f"  Pothole: {len(pothole_images)}")


def download_huggingface_dataset(dataset_name, output_path):
    """
    Download dataset from Hugging Face and convert to binary classification.
    
    Args:
        dataset_name: Name of Hugging Face dataset (e.g., 'michelpf/dataset-pothole')
        output_path: Output directory for binary classification
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
    
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    os.makedirs(os.path.join(output_path, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'pothole'), exist_ok=True)
    
    # Process all splits
    for split_name, split_data in dataset.items():
        print(f"Processing split: {split_name}")
        
        for i, example in enumerate(split_data):
            # Assuming dataset has 'image' and 'label' fields
            img = np.array(example['image'])
            label = example['label']
            
            # Determine class
            if label == 1 or label == 'pothole':
                class_name = 'pothole'
            else:
                class_name = 'normal'
            
            # Save image
            filename = f"{split_name}_{i:05d}.jpg"
            filepath = os.path.join(output_path, class_name, filename)
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Download and conversion complete!")
    print(f"Normal images: {len(os.listdir(os.path.join(output_path, 'normal')))}")
    print(f"Pothole images: {len(os.listdir(os.path.join(output_path, 'pothole')))}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Integration Tools")
    parser.add_argument('--action', type=str, required=True,
                       choices=['convert_yolo', 'convert_coco', 'merge', 'balance', 'download_hf'],
                       help="Action to perform")
    parser.add_argument('--input', type=str, help="Input path(s)")
    parser.add_argument('--output', type=str, help="Output path")
    parser.add_argument('--inputs', type=str, nargs='+', help="Multiple input paths for merge")
    parser.add_argument('--method', type=str, default='oversample',
                       choices=['oversample', 'undersample'],
                       help="Balancing method")
    parser.add_argument('--dataset_name', type=str, help="Hugging Face dataset name")
    
    args = parser.parse_args()
    
    if args.action == 'convert_yolo':
        convert_yolo_to_binary(args.input, args.output)
    
    elif args.action == 'convert_coco':
        # Assuming input is "coco.json,images_dir"
        coco_json, images_dir = args.input.split(',')
        convert_coco_to_binary(coco_json, images_dir, args.output)
    
    elif args.action == 'merge':
        merge_datasets(args.inputs, args.output)
    
    elif args.action == 'balance':
        balance_dataset(args.input, args.method)
    
    elif args.action == 'download_hf':
        download_huggingface_dataset(args.dataset_name, args.output)