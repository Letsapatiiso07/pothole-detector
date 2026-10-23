"""
Simple image filters that imitate dust, rain, mud, harsh sun, shadows and
worn paint. They are used to stress-test a trained model on the test set
(see evaluate.py). They are rough imitations made with OpenCV drawing
calls, not measurements of real weather or road conditions.
"""

import random

import cv2
import numpy as np
from PIL import Image


def add_dust_effect(img):
    """Brown haze with grain"""
    dust_color = np.array([150, 180, 200], dtype=np.float32)  # brown in BGR
    intensity = random.uniform(0.1, 0.4)

    dust_overlay = np.ones_like(img, dtype=np.float32) * dust_color
    result = cv2.addWeighted(img.astype(np.float32), 1 - intensity,
                             dust_overlay, intensity, 0)

    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    result = np.clip(result + noise * intensity, 0, 255)
    return result.astype(np.uint8)


def add_rain_effect(img):
    """Darker, more saturated surface with rain streaks"""
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(1.1, 1.3)
    hsv[:, :, 2] *= random.uniform(0.7, 0.85)
    hsv = np.clip(hsv, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    rain_overlay = result.copy()
    for _ in range(random.randint(50, 150)):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 30)
        length = random.randint(10, 30)
        cv2.line(rain_overlay, (x, y), (x + random.randint(-2, 2), y + length),
                 (200, 200, 200), random.randint(1, 2))

    return cv2.addWeighted(result, 0.7, rain_overlay, 0.3, 0)


def add_mud_splatter(img):
    """Brown spots, as if mud hit the lens"""
    h, w = img.shape[:2]
    overlay = img.copy()

    for _ in range(random.randint(3, 8)):
        x = random.randint(0, w)
        y = random.randint(0, h)
        radius = random.randint(10, 40)
        mud_color = (random.randint(40, 80), random.randint(60, 100),
                     random.randint(80, 130))
        cv2.circle(overlay, (x, y), radius, mud_color, random.choice([-1, 2, 3, 4, 5]))

        for _ in range(random.randint(2, 5)):
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            cv2.circle(overlay, (x + dx, y + dy), random.randint(3, 10), mud_color, -1)

    alpha = random.uniform(0.3, 0.6)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def add_harsh_sunlight(img):
    """More contrast and brightness, plus a flare in the top third"""
    result = cv2.convertScaleAbs(img, alpha=random.uniform(1.2, 1.5),
                                 beta=random.randint(20, 50))

    h, w = img.shape[:2]
    center = (random.randint(w // 4, 3 * w // 4), random.randint(0, h // 3))
    overlay = result.copy()
    for radius in range(30, 150, 20):
        cv2.circle(overlay, center, radius, (200, 250, 255), -1)

    return cv2.addWeighted(result, 0.85, overlay, 0.15, 0)


def add_shadows(img):
    """Dark polygons, like shadows from trees or buildings"""
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    for _ in range(random.randint(1, 3)):
        points = np.array([[random.randint(0, w), random.randint(0, h)]
                           for _ in range(random.randint(4, 8))], dtype=np.int32)
        cv2.fillPoly(mask, [points], 0)

    shadow_intensity = random.uniform(0.3, 0.6)
    mask = mask.astype(np.float32)[:, :, None] / 255.0
    result = img.astype(np.float32) * (mask + (1 - mask) * shadow_intensity)
    return result.astype(np.uint8)


def add_worn_surface(img):
    """Grain and faded colour"""
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    result = np.clip(img.astype(np.float32) + noise, 0, 255)

    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.6, 0.8)
    hsv = np.clip(hsv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


EFFECTS = {
    'dust': add_dust_effect,
    'rain': add_rain_effect,
    'mud': add_mud_splatter,
    'harsh sun': add_harsh_sunlight,
    'shadows': add_shadows,
    'worn surface': add_worn_surface,
}


def pil_effect(effect, seed):
    """Wrap a BGR effect so it takes and returns a PIL image. The seed makes
    each image get the same random filter every run."""
    def apply(img):
        key = seed + sum(img.tobytes()[::4099])
        random.seed(key)
        np.random.seed(key % 2**32)
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return Image.fromarray(cv2.cvtColor(effect(bgr), cv2.COLOR_BGR2RGB))
    return apply
