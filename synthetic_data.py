"""
Synthetic Data Generation for African Road Conditions
Simulates: dust, rain, mud, harsh sunlight, shadows, worn surfaces
"""

import cv2
import numpy as np
import os
from glob import glob
import random

def add_dust_effect(img):
    """Add dust/haze effect common on African unpaved roads"""
    dust_color = np.array([200, 180, 150], dtype=np.float32)  # Brownish dust
    intensity = random.uniform(0.1, 0.4)
    
    # Create dust overlay
    dust_overlay = np.ones_like(img, dtype=np.float32) * dust_color
    result = cv2.addWeighted(img.astype(np.float32), 1 - intensity, 
                            dust_overlay, intensity, 0)
    
    # Add dust particles (noise)
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    result = np.clip(result + noise * intensity, 0, 255)
    
    return result.astype(np.uint8)


def add_rain_effect(img):
    """Simulate rain with wet surface and rain streaks"""
    h, w = img.shape[:2]
    
    # Wet surface (darker, more saturated)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(1.1, 1.3)  # Increase saturation
    hsv[:, :, 2] *= random.uniform(0.7, 0.85)  # Decrease brightness
    hsv = np.clip(hsv, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Add rain streaks
    rain_overlay = result.copy()
    num_streaks = random.randint(50, 150)
    
    for _ in range(num_streaks):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 30)
        length = random.randint(10, 30)
        thickness = random.randint(1, 2)
        cv2.line(rain_overlay, (x, y), (x + random.randint(-2, 2), y + length),
                (200, 200, 200), thickness)
    
    result = cv2.addWeighted(result, 0.7, rain_overlay, 0.3, 0)
    return result


def add_mud_splatter(img):
    """Add mud splatter effect"""
    h, w = img.shape[:2]
    overlay = img.copy()
    
    num_splatters = random.randint(3, 8)
    for _ in range(num_splatters):
        x = random.randint(0, w)
        y = random.randint(0, h)
        radius = random.randint(10, 40)
        
        # Brown mud color with variation
        mud_color = (random.randint(40, 80), random.randint(60, 100), 
                    random.randint(80, 130))
        
        # Random splatter shape
        thickness = random.randint(-1, 5)
        cv2.circle(overlay, (x, y), radius, mud_color, thickness)
        
        # Add smaller droplets around main splatter
        for _ in range(random.randint(2, 5)):
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            small_radius = random.randint(3, 10)
            cv2.circle(overlay, (x + dx, y + dy), small_radius, mud_color, -1)
    
    alpha = random.uniform(0.3, 0.6)
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return result


def add_harsh_sunlight(img):
    """Simulate harsh African sunlight with overexposure and strong shadows"""
    # Increase brightness and contrast
    alpha = random.uniform(1.2, 1.5)  # Contrast
    beta = random.randint(20, 50)     # Brightness
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Add lens flare effect
    h, w = img.shape[:2]
    center_x = random.randint(w // 4, 3 * w // 4)
    center_y = random.randint(0, h // 3)
    
    overlay = result.copy()
    for radius in range(30, 150, 20):
        intensity = int(255 * (1 - radius / 150))
        cv2.circle(overlay, (center_x, center_y), radius, 
                  (255, 250, 200), -1)
    
    result = cv2.addWeighted(result, 0.85, overlay, 0.15, 0)
    return result


def add_shadows(img):
    """Add irregular shadow patterns from trees, buildings"""
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Create irregular shadow shapes
    num_shadows = random.randint(1, 3)
    for _ in range(num_shadows):
        points = []
        num_points = random.randint(4, 8)
        for _ in range(num_points):
            points.append([random.randint(0, w), random.randint(0, h)])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 0)
    
    # Apply shadow (darken)
    shadow_intensity = random.uniform(0.3, 0.6)
    result = img.copy().astype(np.float32)
    mask_normalized = mask.astype(np.float32) / 255.0
    
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (mask_normalized + (1 - mask_normalized) * shadow_intensity)
    
    return result.astype(np.uint8)


def add_worn_surface(img):
    """Simulate worn, cracked road surface texture"""
    # Add texture noise
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    result = np.clip(img.astype(np.float32) + noise, 0, 255)
    
    # Reduce saturation for aged look
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.6, 0.8)
    hsv = np.clip(hsv, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result


def apply_random_effects(img):
    """Apply random combination of effects"""
    effects = [
        add_dust_effect,
        add_rain_effect,
        add_mud_splatter,
        add_harsh_sunlight,
        add_shadows,
        add_worn_surface
    ]
    
    # Apply 1-3 random effects
    num_effects = random.randint(1, 3)
    selected_effects = random.sample(effects, num_effects)
    
    result = img.copy()
    for effect in selected_effects:
        result = effect(result)
    
    return result


def generate_synthetic_data(source_dir, output_dir, num_images_per_class=500):
    """
    Generate synthetic data with African road conditions
    
    Args:
        source_dir: Directory with original images (normal/ and pothole/)
        output_dir: Directory to save synthetic images
        num_images_per_class: Number of synthetic images per class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in ['normal', 'pothole']:
        # Check both possible directory names
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path) and class_name == 'pothole':
            class_path = os.path.join(source_dir, 'potholes')
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found, skipping...")
            continue
        
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get all images
        image_files = glob(os.path.join(class_path, '*.jpg')) + \
                     glob(os.path.join(class_path, '*.jpeg')) + \
                     glob(os.path.join(class_path, '*.png'))
        
        if len(image_files) == 0:
            print(f"No images found in {class_path}")
            continue
        
        print(f"Generating {num_images_per_class} synthetic images for {class_name}...")
        
        for i in range(num_images_per_class):
            # Randomly select source image
            source_img_path = random.choice(image_files)
            img = cv2.imread(source_img_path)
            
            if img is None:
                continue
            
            # Apply random effects
            synthetic_img = apply_random_effects(img)
            
            # Save synthetic image
            output_path = os.path.join(output_class_dir, f'synthetic_{class_name}_{i:04d}.jpg')
            cv2.imwrite(output_path, synthetic_img)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_images_per_class}")
        
        print(f"Completed {class_name}: {num_images_per_class} synthetic images")
    
    print(f"\nSynthetic dataset created at: {output_dir}")
    print("You can now merge this with your original dataset for training.")