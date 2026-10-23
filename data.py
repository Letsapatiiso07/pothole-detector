"""
Data module: download the Stellenbosch pothole images, split them by
recording session and load them for training.

Source: S. Nienaber, M.J. Booysen, R.S. Kroon, Stellenbosch University (2015).
https://drive.google.com/drive/folders/1vUmCvdW3-2lMrhsMbXdMWeLcEz__Ocuy
"""

import csv
import io
import os
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from torchvision import transforms as T

# "Dataset 1 (Simplex)/Train data" in the authors' Google Drive folder
DRIVE_FOLDERS = {
    'pothole': '1457EtpVQf-m0wFRacSjPCQrASXJerHql',  # Positive data
    'normal': '1OaPwRpyI2TnHYa92Y3rCkOUuUrYRIZB5',   # Negative data
}
# simpleTrainFullPhotosSortedFullAnnotations.txt (pothole boxes, full size)
ANNOTATION_FILE_ID = '17Z2_jhaHSUvqmclVwpYeqcP0-2OpnhYQ'

CLASSES = ['normal', 'pothole']
FULL_WIDTH = 3680                  # original GoPro frames are 3680x2760
SAVE_SIZE = (920, 690)             # frames are stored at a quarter of that
SCALE = SAVE_SIZE[0] / FULL_WIDTH
ROAD_BAND = (0.42, 0.71)           # rows of the frame that show the road ahead


def _fetch(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def list_drive_folder(folder_id):
    """Return {file name: file id} for a public Google Drive folder."""
    html = _fetch(f'https://drive.google.com/embeddedfolderview?id={folder_id}').decode('utf-8')
    pairs = re.findall(r'href="https://drive.google.com/file/d/([^/"]+)/view[^"]*".*?'
                       r'flip-entry-title">([^<]*)<', html, re.S)
    return {name: file_id for file_id, name in pairs}


def parse_annotations(text):
    """Parse the authors' box file: '<path> <n> x y w h ...' per line.

    Returns {frame name: [(x, y, w, h), ...]} in full-size pixels.
    The file repeats some lines; repeats are dropped.
    """
    boxes = {}
    for line in text.splitlines():
        parts = line.split()
        # the path has spaces in it: "Train data\Positive data\G0010033.bmp"
        idx = next((i for i, p in enumerate(parts) if p.lower().endswith('.bmp')), None)
        if idx is None:
            continue
        name = re.split(r'[\\/]', parts[idx])[-1][:-4] + '.JPG'
        n = int(parts[idx + 1])
        vals = list(map(int, parts[idx + 2: idx + 2 + 4 * n]))
        boxes[name] = [tuple(vals[i:i + 4]) for i in range(0, 4 * n, 4)]
    return boxes


def session_of(name):
    """GoPro names frames G<session:3><frame:4>, e.g. G0010033 -> '001'."""
    return name[1:4]


def select_sample(pothole_names, normal_names, per_class, seed=42):
    """Pick frames so each recording session gives as many pothole frames
    as normal frames. This stops the model from telling the classes apart
    by session (light, time of day) instead of by potholes.

    Names found in both folders are the same image with both labels, so
    they are dropped. Returns ([(name, label), ...], dropped names).
    """
    clash = set(pothole_names) & set(normal_names)
    pos, neg = defaultdict(list), defaultdict(list)
    for n in sorted(set(pothole_names) - clash):
        pos[session_of(n)].append(n)
    for n in sorted(set(normal_names) - clash):
        neg[session_of(n)].append(n)

    pairs = {s: min(len(pos[s]), len(neg[s])) for s in pos if s in neg}
    frac = min(1.0, per_class / max(1, sum(pairs.values())))
    rng = random.Random(seed)
    chosen = []
    for s in sorted(pairs):
        k = round(pairs[s] * frac)
        chosen += [(n, 'pothole') for n in rng.sample(pos[s], k)]
        chosen += [(n, 'normal') for n in rng.sample(neg[s], k)]
    return chosen, clash


def group_split(groups, test_frac=0.2, val_frac=0.15, seed=42):
    """Assign whole groups to test, val or train.

    Groups are shuffled, then each one goes to the first split (test, then
    val) it fits in without passing 120% of that split's target size;
    the rest go to train. No group ends up in two splits.
    """
    sizes = Counter(groups)
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    target = {'test': test_frac * len(groups), 'val': val_frac * len(groups)}
    filled = {'test': 0, 'val': 0}
    assign = {}
    for g in order:
        assign[g] = 'train'
        for split in ('test', 'val'):
            if filled[split] + sizes[g] <= 1.2 * target[split]:
                assign[g] = split
                filled[split] += sizes[g]
                break
    return [assign[g] for g in groups]


def _download_frame(file_id, out_path):
    # Drive serves a resized copy through its thumbnail link. If that fails,
    # fetch the full 3680x2760 file and shrink it here.
    try:
        raw = _fetch(f'https://drive.google.com/thumbnail?id={file_id}&sz=w{SAVE_SIZE[0]}')
        img = Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception:
        raw = _fetch(f'https://drive.usercontent.google.com/download?id={file_id}&export=download')
        img = Image.open(io.BytesIO(raw)).convert('RGB')
    if img.size != SAVE_SIZE:
        img = img.resize(SAVE_SIZE, Image.BILINEAR)
    img.save(out_path, quality=95)


def download_dataset(out_dir, per_class=600, seed=42, workers=8):
    """Download a session-balanced sample of frames and write manifest.csv."""
    print('Listing the Google Drive folders...')
    listing = {label: list_drive_folder(fid) for label, fid in DRIVE_FOLDERS.items()}
    for label, files in listing.items():
        print(f'  {label}: {len(files)} frames')
    boxes = parse_annotations(_fetch(
        f'https://drive.usercontent.google.com/download?id={ANNOTATION_FILE_ID}&export=download'
    ).decode('latin-1'))

    chosen, clash = select_sample(listing['pothole'], listing['normal'], per_class, seed)
    print(f'Dropped {len(clash)} frames found in both folders (same image, both labels).')
    print(f'Selected {len(chosen)} frames ({len(chosen) // 2} per class).')

    jobs = []
    for name, label in chosen:
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)
        path = os.path.join(out_dir, label, name.replace('.JPG', '.jpg'))
        if not os.path.exists(path):
            jobs.append((listing[label][name], path))
    print(f'Downloading {len(jobs)} frames ({len(chosen) - len(jobs)} already present)...')
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(lambda job: _download_frame(*job), jobs), 1):
            if i % 100 == 0:
                print(f'  {i}/{len(jobs)}')

    with open(os.path.join(out_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label', 'session', 'boxes'])
        for name, label in chosen:
            scaled = [' '.join(str(round(v * SCALE)) for v in b) for b in boxes.get(name, [])]
            w.writerow([f'{label}/{name.replace(".JPG", ".jpg")}', label, session_of(name),
                        ';'.join(scaled) if label == 'pothole' else ''])
    print(f'Wrote {os.path.join(out_dir, "manifest.csv")}')


def load_manifest(data_dir, seed=42):
    """Read manifest.csv and add a 'split' column (split by session)."""
    path = os.path.join(data_dir, 'manifest.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run: python main.py --mode download')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for row, split in zip(rows, group_split([r['session'] for r in rows], seed=seed)):
        row['split'] = split
        row['y'] = CLASSES.index(row['label'])
        row['boxes'] = [tuple(map(int, b.split())) for b in row['boxes'].split(';') if b]
    return rows


def road_box(width, height):
    """(left, top, right, bottom) of the road band in a frame."""
    return 0, round(ROAD_BAND[0] * height), width, round(ROAD_BAND[1] * height)


def crop_road(img):
    """Cut the road band out of a PIL image. The band suits the dash-mounted
    GoPro used for this dataset; other cameras may need other values."""
    return img.crop(road_box(*img.size))


def get_transforms(img_size, train):
    steps = [T.Resize(img_size)]
    if train:
        # no random crops: a crop can cut the only pothole out of the frame
        steps += [T.RandomHorizontalFlip(),
                  T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)]
    steps += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(steps)


class RoadDataset(torch.utils.data.Dataset):
    """Returns (road band tensor, label) for each manifest row."""

    def __init__(self, rows, data_dir, img_size, train=False, effect=None):
        self.rows = rows
        self.data_dir = data_dir
        self.transform = get_transforms(img_size, train)
        self.effect = effect  # optional function applied to the full frame (stress test)

    def __len__(self):
        return len(self.rows)

    def load(self, i):
        img = Image.open(os.path.join(self.data_dir, self.rows[i]['file'])).convert('RGB')
        return self.effect(img) if self.effect else img

    def __getitem__(self, i):
        return self.transform(crop_road(self.load(i))), self.rows[i]['y']


def get_splits(data_dir, seed=42, verbose=True):
    """Return {'train': rows, 'val': rows, 'test': rows}."""
    rows = load_manifest(data_dir, seed)
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    if verbose:
        for s, part in splits.items():
            n_pos = sum(r['y'] for r in part)
            sessions = ', '.join(sorted({r['session'] for r in part}))
            print(f'  {s:5s}: {len(part):4d} frames ({n_pos} pothole, '
                  f'{len(part) - n_pos} normal), sessions {sessions}')
    return splits
