"""Dataset loading and image helpers."""
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 64


def load_dataset(image_dir="images", size=IMAGE_SIZE, cache_path=None):
    """Load every image of `image_dir` that is at least `size`x`size`, resized to `size`x`size`.

    Smaller images are skipped so that the generator doesn't learn to produce upscaled ones.
    Returns a uint8 array (N, size, size, 3) in RGB order. If `cache_path` is given the array
    is saved there on the first run and reloaded on the following ones.
    """
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)

    paths = sorted(Path(image_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in '{image_dir}' (see download_dataset.py)")

    images = []
    for path in tqdm(paths, desc="Loading images"):
        with Image.open(path) as img:
            if min(img.size) < size:
                continue
            images.append(np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC)))
    dataset = np.stack(images)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, dataset)
    return dataset


def to_model_range(images):
    """uint8 [0, 255] -> float32 [-1, 1], the range of the generator's tanh output."""
    return images.astype(np.float32) / 127.5 - 1.0


def to_uint8(images):
    """float [-1, 1] -> uint8 [0, 255]."""
    return np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)


def image_grid(images, rows, cols):
    """Tile `rows * cols` images of shape (H, W, 3) into a single image."""
    h, w, c = images.shape[1:]
    return images[:rows * cols].reshape(rows, cols, h, w, c).swapaxes(1, 2).reshape(rows * h, cols * w, c)
