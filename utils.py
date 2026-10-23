"""
Utility module: metrics, Grad-CAM and plots.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from matplotlib.patches import Rectangle

BLUE = '#2a78d6'
GREY = '#8a8984'
INK = '#0b0b0b'
INK_2 = '#52514e'
BOX = '#1baf7a'


def binary_metrics(y_true, y_pred):
    """Accuracy, precision, recall and F1 for the positive class (1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'accuracy': (tp + tn) / len(y_true), 'precision': precision,
            'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def bootstrap_ci(y_true, y_pred, metric='f1', n=2000, seed=0):
    """95% bootstrap interval for one metric over test images."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(binary_metrics(y_true[idx], y_pred[idx])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def grad_cam(model, layer, x):
    """Grad-CAM heat map for the pothole logit.

    x: one input tensor (1, 3, H, W). Returns an (H, W) array in [0, 1]
    showing which parts of the input pushed the score up.
    """
    store = {}

    def keep(module, inputs, out):
        store['act'] = out
        out.register_hook(lambda g: store.update(grad=g))

    hook = layer.register_forward_hook(keep)
    try:
        model.zero_grad()
        model(x.requires_grad_(True))[0, 0].backward()
    finally:
        hook.remove()
    weights = store['grad'].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store['act']).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def plot_confusion_matrix(cm, classes, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=13,
                    color='white' if cm[i][j] > np.max(cm) / 2 else INK)
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes)
    ax.set_xlabel('Predicted', color=INK_2)
    ax.set_ylabel('True', color=INK_2)
    ax.set_title('Test set confusion matrix', color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curve(precision, recall, ap, base_rate, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(recall, precision, color=BLUE, lw=2, label=f'Model (AP {ap:.2f})')
    ax.axhline(base_rate, color=GREY, lw=1.5, ls='--',
               label=f'Always "pothole" ({base_rate:.2f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall', color=INK_2)
    ax.set_ylabel('Precision', color=INK_2)
    ax.set_title('Test set precision-recall curve', color=INK)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left', frameon=False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(examples, path):
    """examples: dicts with 'strip' (H, W, 3 uint8), 'cam' (H, W), 'boxes'
    (x, y, w, h in strip pixels) and 'title'. One row per example: the road
    band with the labelled pothole boxes, and the Grad-CAM overlay."""
    n = len(examples)
    h, w = examples[0]['strip'].shape[:2]
    height = n * (13 / 2 * h / w + 0.45) + 0.6
    fig, axes = plt.subplots(n, 2, figsize=(13, height))
    axes = np.atleast_2d(axes)
    for row, ex in zip(axes, examples):
        row[0].imshow(ex['strip'])
        for (x, y, bw, bh) in ex['boxes']:
            row[0].add_patch(Rectangle((x, y), bw, bh, fill=False, ec=BOX, lw=1.5))
        row[1].imshow(ex['strip'])
        row[1].imshow(ex['cam'], cmap='jet', alpha=0.4, vmin=0, vmax=1)
        row[0].set_title(ex['title'], loc='left', fontsize=10, color=INK)
        for ax in row:
            ax.axis('off')
    fig.suptitle('Left: road band, green boxes are potholes marked by the dataset authors.   '
                 'Right: Grad-CAM, red areas raised the pothole score most.',
                 x=0.01, ha='left', fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / height))
    fig.savefig(path, dpi=110, pil_kwargs={'quality': 85})
    plt.close(fig)
