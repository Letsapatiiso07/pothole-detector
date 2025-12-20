"""
Data Loading and Augmentation Module
Enhanced with proper train/val/test split and African road condition simulation
FIXED: Handles empty split directories and multiple source folders
"""

import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_dataset_structure(data_dir):
    """
    Prepare dataset structure and split into train/val/test.
    Creates proper directory structure if not exists.
    FIXED: Checks if split directories are actually populated.
    """
    # Check if split already done (and directories are NOT empty)
    train_dir = os.path.join(data_dir, 'train')
    train_normal = os.path.join(train_dir, 'normal')
    train_pothole = os.path.join(train_dir, 'pothole')
    
    split_exists_and_populated = (
        os.path.exists(train_normal) and 
        os.path.exists(train_pothole) and
        len([f for f in os.listdir(train_normal) if f.endswith(('.jpg', '.jpeg', '.png'))]) > 0 and
        len([f for f in os.listdir(train_pothole) if f.endswith(('.jpg', '.jpeg', '.png'))]) > 0
    )
    
    if split_exists_and_populated:
        print("✓ Dataset already split and populated. Using existing structure.")
        return
    
    # Remove empty split directories if they exist
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            # Check if empty
            try:
                contents = os.listdir(split_path)
                is_empty = len(contents) == 0 or all(
                    os.path.isdir(os.path.join(split_path, d)) and 
                    len(os.listdir(os.path.join(split_path, d))) == 0 
                    for d in contents
                )
                
                if is_empty:
                    print(f"Removing empty directory: {split}/")
                    shutil.rmtree(split_path)
            except:
                pass
    
    # Create fresh split directories
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'pothole']:
            os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)
    
    # Find source directories
    print("\nSearching for source images...")
    normal_images = []
    pothole_images = []
    
    # Collect all normal images
    normal_dir = os.path.join(data_dir, 'normal')
    if os.path.exists(normal_dir):
        normal_images = [(normal_dir, f) for f in os.listdir(normal_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Found {len(normal_images)} normal images in normal/")
    
    # Collect all pothole images (check both 'pothole' and 'potholes')
    for pothole_dirname in ['pothole', 'potholes']:
        pothole_dir = os.path.join(data_dir, pothole_dirname)
        if os.path.exists(pothole_dir):
            imgs = [(pothole_dir, f) for f in os.listdir(pothole_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
            pothole_images.extend(imgs)
            print(f"  Found {len(imgs)} pothole images in {pothole_dirname}/")
    
    if len(normal_images) == 0 or len(pothole_images) == 0:
        raise ValueError(
            f"Insufficient images found!\n"
            f"  Normal: {len(normal_images)}\n"
            f"  Pothole: {len(pothole_images)}\n"
            f"Please ensure images exist in normal/ and pothole/or potholes/ folders."
        )
    
    print(f"\nTotal images to split:")
    print(f"  Normal:  {len(normal_images)}")
    print(f"  Pothole: {len(pothole_images)}")
    print(f"\nSplitting dataset (70% train, 15% val, 15% test)...")
    
    # Split each class
    for class_name, images_list in [('normal', normal_images), ('pothole', pothole_images)]:
        # 70% train, 15% val, 15% test
        train_imgs, temp_imgs = train_test_split(images_list, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Copy files to respective directories
        for source_dir, img_file in train_imgs:
            src = os.path.join(source_dir, img_file)
            dst = os.path.join(data_dir, 'train', class_name, img_file)
            shutil.copy2(src, dst)
        
        for source_dir, img_file in val_imgs:
            src = os.path.join(source_dir, img_file)
            dst = os.path.join(data_dir, 'val', class_name, img_file)
            shutil.copy2(src, dst)
        
        for source_dir, img_file in test_imgs:
            src = os.path.join(source_dir, img_file)
            dst = os.path.join(data_dir, 'test', class_name, img_file)
            shutil.copy2(src, dst)
        
        print(f"  {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    print("\n✓ Dataset split complete!")


def get_data_generators(data_dir, batch_size, img_size):
    """
    Prepare data generators for training, validation, and testing.
    Enhanced with African road condition augmentation (dust, rain, lighting).
    """
    try:
        # Prepare dataset structure
        prepare_dataset_structure(data_dir)
        
        # Enhanced data augmentation for African road conditions
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,  # Roads at various angles
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.4, 1.6],  # Harsh sun to deep shadows
            channel_shift_range=30,  # Color variations (dust, wet roads)
            fill_mode='nearest'
        )
        
        # Validation: only rescaling
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Test: only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Use split directories
        train_path = os.path.join(data_dir, 'train')
        val_path = os.path.join(data_dir, 'val')
        test_path = os.path.join(data_dir, 'test')
        
        train_gen = train_datagen.flow_from_directory(
            train_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            classes=['normal', 'pothole']
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            classes=['normal', 'pothole']
        )
        
        test_gen = test_datagen.flow_from_directory(
            test_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            classes=['normal', 'pothole']
        )
        
        print(f"\n✓ Data generators ready:")
        print(f"  Train samples:      {train_gen.samples}")
        print(f"  Validation samples: {val_gen.samples}")
        print(f"  Test samples:       {test_gen.samples}")
        print(f"  Batch size:         {batch_size}")
        print(f"  Image size:         {img_size}x{img_size}")
        
        if train_gen.samples == 0:
            raise ValueError("No training samples found! Check dataset structure.")
        
        return train_gen, val_gen, test_gen
        
    except Exception as e:
        print(f"\n❌ Error in data loading: {e}")
        raise