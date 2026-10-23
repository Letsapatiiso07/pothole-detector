"""
Evaluation module: test set metrics, trivial baselines, figures, a check of
where Grad-CAM points, and a stress test with the filters in synthetic_data.py.
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from data import CLASSES, ROAD_BAND, RoadDataset, crop_road, get_splits
from model import last_conv_layer, load_model
from synthetic_data import EFFECTS, pil_effect
from train import predict_probs
from utils import (binary_metrics, bootstrap_ci, grad_cam, plot_confusion_matrix,
                   plot_examples, plot_pr_curve)


def _loader(rows, data_dir, img_size, effect=None):
    ds = RoadDataset(rows, data_dir, img_size, effect=effect)
    return torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)


def _pick_examples(probs, labels, threshold, seed=0):
    """Two random picks from each outcome; for mistakes, the two worst."""
    pred = (probs >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    picks = []
    for name, y, p in [('correct pothole', 1, 1), ('correct normal', 0, 0),
                       ('missed pothole', 1, 0), ('false alarm', 0, 1)]:
        idx = np.where((labels == y) & (pred == p))[0]
        if name == 'missed pothole':
            idx = idx[np.argsort(probs[idx])]
        elif name == 'false alarm':
            idx = idx[np.argsort(-probs[idx])]
        else:
            idx = rng.permutation(idx)
        picks += [(name, i) for i in idx[:2]]
    return picks


def evaluate(model_path, data_dir, docs_dir, threshold=0.5):
    model, info = load_model(model_path)
    img_size = tuple(info['img_size'])
    splits = get_splits(data_dir, info['seed'])
    test = splits['test']

    probs, labels = predict_probs(model, _loader(test, data_dir, img_size))
    pred = (probs >= threshold).astype(int)
    m = binary_metrics(labels, pred)
    m['f1_95ci'] = bootstrap_ci(labels, pred, 'f1')
    m['average_precision'] = float(average_precision_score(labels, probs))

    train_pos = np.mean([r['y'] for r in splits['train']])
    baselines = {
        'always pothole': binary_metrics(labels, np.ones_like(labels)),
        'always normal': binary_metrics(labels, np.zeros_like(labels)),
    }
    print(f'\nTest set: {len(labels)} frames ({labels.sum()} pothole), '
          f'threshold {threshold}')
    print(f'{"":16s} accuracy precision recall    F1')
    for name, r in [('model', m)] + list(baselines.items()):
        print(f'{name:16s} {r["accuracy"]:8.3f} {r["precision"]:9.3f} '
              f'{r["recall"]:6.3f} {r["f1"]:5.3f}')
    print(f'Model F1 95% bootstrap interval: {m["f1_95ci"][0]:.3f} to {m["f1_95ci"][1]:.3f}')
    print(f'Model average precision: {m["average_precision"]:.3f}')
    print(f'Confusion matrix [[TN FP] [FN TP]]: [[{m["tn"]} {m["fp"]}] [{m["fn"]} {m["tp"]}]]')

    os.makedirs(docs_dir, exist_ok=True)
    cm = [[m['tn'], m['fp']], [m['fn'], m['tp']]]
    plot_confusion_matrix(cm, CLASSES, os.path.join(docs_dir, 'confusion_matrix.png'))
    prec, rec, _ = precision_recall_curve(labels, probs)
    plot_pr_curve(prec, rec, m['average_precision'], labels.mean(),
                  os.path.join(docs_dir, 'pr_curve.png'))

    ds = RoadDataset(test, data_dir, img_size)

    def cam_and_boxes(i):
        """Road band, Grad-CAM at band size, and marked boxes in band pixels."""
        frame = ds.load(i)
        strip = crop_road(frame)
        cam = grad_cam(model, last_conv_layer(model), ds[i][0][None])
        cam = np.asarray(torch.nn.functional.interpolate(
            torch.tensor(cam)[None, None], size=strip.size[::-1], mode='bilinear')[0, 0])
        top = round(ROAD_BAND[0] * frame.height)
        return strip, cam, [(x, y - top, w, h) for x, y, w, h in test[i]['boxes']]

    # Does the Grad-CAM peak sit on a marked pothole? Checked on pothole frames
    # the model flags; a hit is a peak within 40 px (about one Grad-CAM cell)
    # of a box. "chance" is the share of the band those zones cover.
    hits, chance = [], []
    for i in np.where((labels == 1) & (pred == 1))[0]:
        strip, cam, boxes = cam_and_boxes(i)
        zone = np.zeros(cam.shape, bool)
        for x, y, w, h in boxes:
            zone[max(0, y - 40):max(0, y + h + 40), max(0, x - 40):x + w + 40] = True
        if zone.any():
            hits.append(bool(zone[np.unravel_index(cam.argmax(), cam.shape)]))
            chance.append(float(zone.mean()))
    pointing = {'frames': len(hits), 'hits': int(sum(hits)),
                'hit_rate': float(np.mean(hits)), 'chance_rate': float(np.mean(chance))}
    print(f'Grad-CAM peak on a marked pothole: {pointing["hits"]} of {pointing["frames"]} '
          f'flagged pothole frames ({pointing["hit_rate"]:.2f}); a random point: '
          f'{pointing["chance_rate"]:.2f}')

    examples = []
    for name, i in _pick_examples(probs, labels, threshold):
        strip, cam, boxes = cam_and_boxes(i)
        examples.append({'strip': np.asarray(strip), 'cam': cam, 'boxes': boxes,
                         'title': f'{name}: {test[i]["file"]}, pothole score {probs[i]:.2f}'})
    plot_examples(examples, os.path.join(docs_dir, 'examples.jpg'))

    # stress test: same test frames with each filter applied
    stress = {}
    print('\nStress test (filters from synthetic_data.py on the test frames):')
    for name, effect in EFFECTS.items():
        p, y = predict_probs(model, _loader(test, data_dir, img_size, pil_effect(effect, 0)))
        r = binary_metrics(y, (p >= threshold).astype(int))
        stress[name] = r
        print(f'  {name:12s} accuracy {r["accuracy"]:.3f}  precision {r["precision"]:.3f}  '
              f'recall {r["recall"]:.3f}  F1 {r["f1"]:.3f}')

    results = {
        'model_path': model_path, 'backbone': info['backbone'], 'img_size': img_size,
        'best_epoch': info['epoch'], 'threshold': threshold,
        'splits': {s: {'frames': len(r), 'pothole': int(sum(x['y'] for x in r)),
                       'sessions': sorted({x['session'] for x in r})} for s, r in splits.items()},
        'train_pothole_share': float(train_pos),
        'test': m, 'baselines': baselines, 'gradcam_pointing': pointing,
        'stress_test': stress,
        'history': info['history'],
    }
    with open(os.path.join(docs_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved figures and results.json to {docs_dir}/')
    return results
