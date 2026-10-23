"""
Pothole detector: entry point.

Modes: download the data, train, evaluate on the test set, predict on one frame.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Pothole classifier for dash-camera frames')
    parser.add_argument('--mode', required=True,
                        choices=['download', 'train', 'evaluate', 'predict'])
    parser.add_argument('--data_dir', default='data/stellenbosch',
                        help='Where the frames and manifest.csv live')
    parser.add_argument('--model_path', default='models/pothole_classifier.pt',
                        help='Where to save or load the model')
    parser.add_argument('--per_class', type=int, default=600,
                        help='Frames to download per class')
    parser.add_argument('--backbone', default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'resnet18'])
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--img_width', type=int, default=736,
                        help='Width the road band is resized to (height keeps the aspect)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--docs_dir', default='docs', help='Where evaluate writes figures')
    parser.add_argument('--image', help='Frame to score in predict mode')
    parser.add_argument('--out', help='Output picture in predict mode')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'download':
        from data import download_dataset
        download_dataset(args.data_dir, args.per_class, args.seed)

    elif args.mode == 'train':
        from data import ROAD_BAND, SAVE_SIZE, RoadDataset, get_splits
        from model import build_model
        from train import train_model
        # the road band is 920x200 in the stored frames; keep that aspect
        band_h = round((ROAD_BAND[1] - ROAD_BAND[0]) * SAVE_SIZE[1])
        img_size = (round(args.img_width * band_h / SAVE_SIZE[0]), args.img_width)
        print('Splitting by recording session:')
        splits = get_splits(args.data_dir, args.seed)
        loaders = {s: torch.utils.data.DataLoader(
            RoadDataset(rows, args.data_dir, img_size, train=(s == 'train')),
            batch_size=args.batch_size, shuffle=(s == 'train'), num_workers=0)
            for s, rows in splits.items() if s != 'test'}
        os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
        model = build_model(args.backbone)
        print(f'Training {args.backbone} on {img_size[1]}x{img_size[0]} road bands '
              f'for {args.epochs} epochs')
        train_model(model, loaders, args.epochs, args.lr, args.model_path,
                    backbone=args.backbone, img_size=img_size, seed=args.seed)
        print(f'Best model saved to {args.model_path}')

    elif args.mode == 'evaluate':
        from evaluate import evaluate
        evaluate(args.model_path, args.data_dir, args.docs_dir, args.threshold)

    elif args.mode == 'predict':
        if not args.image:
            sys.exit('predict mode needs --image')
        from infer import predict_image
        predict_image(args.model_path, args.image, args.out, args.threshold)


if __name__ == '__main__':
    main()
