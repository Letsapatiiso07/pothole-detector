"""
Create a small sample dataset for testing
This creates synthetic road images to test the pipeline
"""

import cv2
import numpy as np
import os

def create_sample_road_image(has_pothole=False, size=(640, 480)):
    """Create a synthetic road image"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Create road texture (gray)
    road_color = np.random.randint(80, 120, 3)
    img[:] = road_color
    
    # Add noise for texture
    noise = np.random.randint(-20, 20, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add road markings
    cv2.line(img, (size[0]//3, 0), (size[0]//3, size[1]), (200, 200, 200), 5)
    cv2.line(img, (2*size[0]//3, 0), (2*size[0]//3, size[1]), (200, 200, 200), 5)
    
    if has_pothole:
        # Add pothole (dark irregular shape)
        center_x = np.random.randint(size[0]//4, 3*size[0]//4)
        center_y = np.random.randint(size[1]//4, 3*size[1]//4)
        
        # Create irregular pothole shape
        for _ in range(3):
            radius = np.random.randint(30, 80)
            offset_x = np.random.randint(-20, 20)
            offset_y = np.random.randint(-20, 20)
            
            cv2.ellipse(img, 
                       (center_x + offset_x, center_y + offset_y),
                       (radius, radius//2),
                       np.random.randint(0, 180),
                       0, 360,
                       (30, 30, 30),
                       -1)
        
        # Add cracks around pothole
        for _ in range(5):
            x1 = center_x + np.random.randint(-100, 100)
            y1 = center_y + np.random.randint(-100, 100)
            x2 = x1 + np.random.randint(-50, 50)
            y2 = y1 + np.random.randint(-50, 50)
            cv2.line(img, (x1, y1), (x2, y2), (50, 50, 50), 2)
    
    return img

def setup_sample_dataset(output_dir='data/pothole-detection-dataset', num_samples=50):
    """
    Create a sample dataset for testing
    
    Args:
        output_dir: Output directory
        num_samples: Number of images per class
    """
    print(f"Creating sample dataset with {num_samples} images per class...")
    
    # Create directories
    normal_dir = os.path.join(output_dir, 'normal')
    pothole_dir = os.path.join(output_dir, 'potholes')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pothole_dir, exist_ok=True)
    
    # Generate normal road images
    print("Generating normal road images...")
    for i in range(num_samples):
        img = create_sample_road_image(has_pothole=False)
        filepath = os.path.join(normal_dir, f'normal_{i:04d}.jpg')
        cv2.imwrite(filepath, img)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} normal images")
    
    # Generate pothole images
    print("Generating pothole images...")
    for i in range(num_samples):
        img = create_sample_road_image(has_pothole=True)
        filepath = os.path.join(pothole_dir, f'pothole_{i:04d}.jpg')
        cv2.imwrite(filepath, img)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} pothole images")
    
    print(f"\nSample dataset created at: {output_dir}")
    print(f"  Normal images: {len(os.listdir(normal_dir))}")
    print(f"  Pothole images: {len(os.listdir(pothole_dir))}")
    print("\nYou can now run:")
    print(f"  python main.py --mode generate_synthetic --data_dir {output_dir}")
    print(f"  python main.py --mode train --data_dir {output_dir} --epochs 10")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample dataset for testing")
    parser.add_argument('--output_dir', type=str, 
                       default='data/pothole-detection-dataset',
                       help='Output directory for sample dataset')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of images per class')
    
    args = parser.parse_args()
    
    setup_sample_dataset(args.output_dir, args.num_samples)