"""
Pothole & Road Anomaly Detector - Main Entry Point
Enhanced version with African road condition simulation and multi-dataset support
"""

import argparse
import os
import sys
from data import get_data_generators
from model import build_model
from train import train_and_evaluate
from infer import run_inference
from synthetic_data import generate_synthetic_data

def main():
    parser = argparse.ArgumentParser(description="Pothole & Road Anomaly Detector")
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'infer', 'generate_synthetic'], 
                       help="Mode: 'train', 'infer', or 'generate_synthetic'")
    parser.add_argument('--data_dir', type=str, default='data/pothole-detection-dataset', 
                       help="Path to dataset directory")
    parser.add_argument('--model_path', type=str, default='pothole_model.h5', 
                       help="Path to save/load model")
    parser.add_argument('--video_path', type=str, default='0', 
                       help="Path to video file or '0' for webcam")
    parser.add_argument('--batch_size', type=int, default=32, 
                       help="Batch size for training/inference")
    parser.add_argument('--epochs', type=int, default=20, 
                       help="Number of epochs for training")
    parser.add_argument('--img_size', type=int, default=224, 
                       help="Image size for model input")
    parser.add_argument('--synthetic_output', type=str, default='data/synthetic', 
                       help="Output directory for synthetic data")
    parser.add_argument('--num_synthetic', type=int, default=500, 
                       help="Number of synthetic images to generate per class")
    
    args = parser.parse_args()
    
    if args.mode == 'generate_synthetic':
        print("=== Generating Synthetic Data for African Road Conditions ===")
        if not os.path.exists(args.data_dir):
            print(f"Source dataset not found at {args.data_dir}")
            sys.exit(1)
        generate_synthetic_data(args.data_dir, args.synthetic_output, args.num_synthetic)
        print(f"Synthetic data generated at {args.synthetic_output}")
        return
    
    # Check dataset
    if not os.path.exists(args.data_dir):
        print(f"Dataset not found at {args.data_dir}")
        print("Please download from https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset")
        print(f"Extract to {args.data_dir}, ensuring subfolders 'normal' and 'potholes'.")
        sys.exit(1)
    
    if args.mode == 'train':
        print("=== Starting Training ===")
        train_gen, val_gen, test_gen = get_data_generators(args.data_dir, args.batch_size, args.img_size)
        model = build_model((args.img_size, args.img_size, 3))
        train_and_evaluate(model, train_gen, val_gen, test_gen, args.epochs, args.model_path)
    
    elif args.mode == 'infer':
        print("=== Starting Inference ===")
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Please train first.")
            sys.exit(1)
        run_inference(args.model_path, args.video_path, args.img_size)

if __name__ == "__main__":
    main()