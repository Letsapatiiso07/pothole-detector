"""
Training module: fine-tune the classifier and keep the epoch with the
lowest validation loss.
"""

import time

import numpy as np
import torch
from torch import nn

from model import save_model
from utils import binary_metrics


def predict_probs(model, loader):
    """Pothole probabilities and true labels for every image in a loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x)[:, 0]).numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_model(model, loaders, epochs, lr, model_path, **info):
    """Train, check the validation set each epoch and save the best model."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, history = float('inf'), []

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total, seen = 0.0, 0
        for x, y in loaders['train']:
            opt.zero_grad()
            loss = loss_fn(model(x)[:, 0], y.float())
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)
        sched.step()

        probs, labels = predict_probs(model, loaders['val'])
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
        val = binary_metrics(labels, (probs >= 0.5).astype(int))
        history.append({'epoch': epoch, 'train_loss': total / seen, 'val_loss': val_loss,
                        'val_f1': val['f1'], 'val_accuracy': val['accuracy']})
        saved = ''
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, model_path, epoch=epoch, history=history, **info)
            saved = '  (saved)'
        print(f'Epoch {epoch}/{epochs}  train loss {total / seen:.3f}  val loss {val_loss:.3f}  '
              f'val acc {val["accuracy"]:.3f}  val F1 {val["f1"]:.3f}  '
              f'{time.time() - start:.0f}s{saved}', flush=True)
    return history
