"""Compare generators with FID, KID, precision and recall computed on InceptionV3 features.

The generated images are compared with the training set. All models are evaluated on the same
latent vectors, so the numbers of different models (and checkpoints) are directly comparable.
The Inception weights are Keras' ImageNet ones, so the absolute values are not comparable with
the ones in the literature (which use a different port of the network and 299x299 inputs).
"""
import argparse
import hashlib
from pathlib import Path

import keras
import numpy as np
from keras import layers

from data import load_dataset, to_model_range, to_uint8
from models import LATENT_DIM, load_generator

KID_SUBSETS = 50
KID_SUBSET_SIZE = 1000


def build_feature_extractor():
    """64x64 images in [-1, 1] -> 2048 InceptionV3 features (the range InceptionV3 expects)."""
    inception = keras.applications.InceptionV3(include_top=False, pooling="avg", weights="imagenet")
    inputs = keras.Input((64, 64, 3))
    return keras.Model(inputs, inception(layers.Resizing(299, 299)(inputs)))


def extract_features(extractor, images_uint8, batch_size):
    features = extractor.predict(to_model_range(images_uint8), batch_size=batch_size, verbose=0)
    return features.astype(np.float64)


def generate_images(generator, latents, bgr, batch_size=500):
    images = []
    for start in range(0, len(latents), batch_size):
        batch = to_uint8(generator.predict(latents[start:start + batch_size], verbose=0))
        images.append(batch[..., ::-1] if bgr else batch)
    return np.concatenate(images)


def fid(real, fake):
    """Frechet distance between the Gaussians fitted on the two sets of features."""
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    # trace(sqrt(cov_r @ cov_f)) is the sum of the square roots of the eigenvalues of the product
    eigenvalues = np.linalg.eigvals(cov_r @ cov_f)
    trace_sqrt = np.sqrt(np.clip(eigenvalues.real, 0, None)).sum()
    return float(((mu_r - mu_f) ** 2).sum() + np.trace(cov_r) + np.trace(cov_f) - 2 * trace_sqrt)


def kid(real, fake, rng):
    """Unbiased MMD^2 with a cubic polynomial kernel, averaged over random subsets."""
    size = min(KID_SUBSET_SIZE, len(real), len(fake))
    d = real.shape[1]
    scores = []
    for _ in range(KID_SUBSETS):
        x = real[rng.choice(len(real), size, replace=False)]
        y = fake[rng.choice(len(fake), size, replace=False)]
        k_xx, k_yy, k_xy = ((a @ b.T / d + 1) ** 3 for a, b in ((x, x), (y, y), (x, y)))
        scores.append((k_xx.sum() - np.trace(k_xx)) / (size * (size - 1))
                      + (k_yy.sum() - np.trace(k_yy)) / (size * (size - 1))
                      - 2 * k_xy.mean())
    return float(np.mean(scores)), float(np.std(scores))


def squared_distances(a, b):
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0)


def knn_radii(points, ks, chunk=2000):
    """Squared distance of every point from its k-th nearest neighbour in the same set, for each k."""
    radii = np.empty((len(points), len(ks)))
    for start in range(0, len(points), chunk):
        distances = squared_distances(points[start:start + chunk], points)
        # column 0 of the sorted distances is the point itself
        radii[start:start + chunk] = np.partition(distances, ks, axis=1)[:, ks]
    return radii


def coverage(points, support, support_radii, chunk=2000):
    """For each k, fraction of `points` inside the hypersphere of at least one point of `support`."""
    inside = np.zeros(support_radii.shape[1])
    for start in range(0, len(points), chunk):
        distances = squared_distances(points[start:start + chunk], support)
        for i in range(len(inside)):
            inside[i] += (distances <= support_radii[None, :, i]).any(axis=1).sum()
    return inside / len(points)


def precision_recall(real, fake, ks):
    """Improved precision and recall (Kynkaanniemi et al. 2019), one value per k in `ks`.

    Precision: fraction of generated images inside the support of the real ones (quality).
    Recall: fraction of real images inside the support of the generated ones (coverage).
    The larger k is, the larger the hyperspheres that make up the support, and the higher both are.
    """
    real_radii, fake_radii = knn_radii(real, ks), knn_radii(fake, ks)
    return coverage(fake, real, real_radii), coverage(real, fake, fake_radii)


def cached(path, compute):
    if path.exists():
        return np.load(path)
    values = compute()
    np.save(path, values)
    return values


def real_images(args, rng):
    dataset = load_dataset(args.images, cache_path=Path(args.images) / "dataset_64.npy")
    return dataset[rng.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="generators to evaluate (.keras, or .h5 from the old scripts)")
    parser.add_argument("--images", default="images", help="folder with the training images")
    parser.add_argument("--num-samples", type=int, default=10000, help="real and generated images used")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10, 20],
                        help="neighbours used for precision and recall (one result for each)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size of the Inception network")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ks = sorted(set(args.k))
    extractor = build_feature_extractor()

    # Features are cached: those of the real images depend on the dataset, the number of samples and
    # the seed; those of a generator on the file, the number of samples and the seed
    cache_dir = Path(args.images)
    real = cached(cache_dir / f"inception_real_{args.num_samples}_{args.seed}.npy", lambda: extract_features(
        extractor, real_images(args, np.random.default_rng([args.seed, 0])), args.batch_size))

    latents = np.random.default_rng([args.seed, 1]).normal(size=(len(real), LATENT_DIM)).astype(np.float32)
    kid_rng = np.random.default_rng([args.seed, 2])
    header = "".join(f"  P/R@{k}".rjust(14) for k in ks)
    print(f"{'model':<40} {'FID':>8} {'KID x1e3':>16}{header}")
    for path in args.models:
        path = Path(path)
        key = hashlib.md5(f"{path.resolve()}{path.stat().st_mtime_ns}".encode()).hexdigest()[:10]
        # Generators saved by the old scripts, trained on OpenCV's BGR images, output BGR
        fake = cached(cache_dir / f"inception_fake_{key}_{args.num_samples}_{args.seed}.npy", lambda: extract_features(
            extractor, generate_images(load_generator(path), latents, bgr=path.suffix == ".h5"), args.batch_size))
        kid_mean, kid_std = kid(real, fake, kid_rng)
        precision, recall = precision_recall(real, fake, ks)
        pr = "".join(f"{p:7.3f}/{r:.3f}" for p, r in zip(precision, recall))
        print(f"{path.name:<40} {fid(real, fake):8.2f} {kid_mean * 1e3:9.2f} +-{kid_std * 1e3:5.2f}  {pr}")


if __name__ == "__main__":
    main()
