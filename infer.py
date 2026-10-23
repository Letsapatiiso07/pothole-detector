"""
Inference module: score one dash-camera frame and save a picture of the
result with a Grad-CAM overlay on the road band.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data import crop_road, get_transforms, road_box
from model import last_conv_layer, load_model
from utils import grad_cam


def predict_image(model_path, image_path, out_path=None, threshold=0.5):
    """Return the pothole probability for one frame and save a picture."""
    model, info = load_model(model_path)
    img_size = tuple(info['img_size'])
    frame = Image.open(image_path).convert('RGB')
    strip = crop_road(frame)
    x = get_transforms(img_size, train=False)(strip)[None]

    with torch.no_grad():
        prob = torch.sigmoid(model(x)[0, 0]).item()
    label = 'pothole' if prob >= threshold else 'no pothole'
    print(f'{image_path}: {label} (pothole score {prob:.3f}, threshold {threshold})')

    if out_path is None:
        os.makedirs('predictions', exist_ok=True)
        out_path = os.path.join('predictions', os.path.splitext(os.path.basename(image_path))[0]
                                + '_pred.jpg')
    cam = np.asarray(Image.fromarray(np.uint8(255 * grad_cam(model, last_conv_layer(model), x)))
                     .resize(strip.size, Image.BILINEAR)) / 255.0
    left, top, right, bottom = road_box(*frame.size)
    fig, ax = plt.subplots(figsize=(8, 8 * frame.height / frame.width))
    ax.imshow(frame)
    ax.imshow(cam, cmap='jet', alpha=0.4, extent=(left, right, bottom, top))
    ax.set_xlim(0, frame.width)
    ax.set_ylim(frame.height, 0)
    ax.add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                               fill=False, ec='white', lw=1, ls='--'))
    ax.set_title(f'Predicted: {label} (pothole score {prob:.2f}). '
                 'Dashed box: road band the model sees. Heat map: Grad-CAM.',
                 fontsize=10, loc='left')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f'Saved {out_path}')
    return prob
