"""Train the GAN on the anime faces dataset.

The Keras backend is chosen with the KERAS_BACKEND environment variable (tensorflow or torch).
"""
import argparse
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from data import image_grid, load_dataset, to_uint8
from models import GAN, LATENT_DIM, build_discriminator, build_generator

NOISE_START = 0.7   # std of the noise added to the discriminator's input at the first epoch...
NOISE_DECAY = 0.05  # ...reduced by this amount every epoch...
NOISE_MIN = 1e-4    # ...down to this value
D_LEARNING_RATE = 2e-5  # Adam learning rate of the discriminator
G_LEARNING_RATE = 1e-5  # Adam learning rate of the generator


class NoiseSchedule(keras.callbacks.Callback):
    """Linearly decays the std of the discriminator's input noise."""

    def on_epoch_begin(self, epoch, logs=None):
        std = max(NOISE_START - epoch * NOISE_DECAY, NOISE_MIN)
        self.model.discriminator.get_layer("input_noise").set_stddev(std)
        print(f"Noise standard deviation: {std:.3f}")


class SaveSamples(keras.callbacks.Callback):
    """Saves a grid of generated faces after every epoch and a generator checkpoint every few."""

    def __init__(self, results_dir, checkpoints_dir, checkpoint_every, rows=5, cols=5):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoint_every = checkpoint_every
        self.rows, self.cols = rows, cols
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        generator = self.model.generator
        z = np.random.normal(size=(self.rows * self.cols, LATENT_DIM)).astype(np.float32)
        images = to_uint8(generator.predict(z, verbose=0))
        Image.fromarray(image_grid(images, self.rows, self.cols)).save(
            self.results_dir / f"plt_{epoch + 1:04d}.png")
        if (epoch + 1) % self.checkpoint_every == 0:
            generator.save(self.checkpoints_dir / f"anime_generator_epoch{epoch + 1}.keras")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="images", help="folder with the .jpg training images")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="trained_generator.keras", help="final generator file")
    parser.add_argument("--checkpoints", default="checkpoints", help="folder for the periodic generator checkpoints")
    parser.add_argument("--results", default="results", help="folder for the sample images of each epoch")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="epochs between checkpoints")
    args = parser.parse_args()

    dataset = load_dataset(args.images, cache_path=Path(args.images) / "dataset_64.npy")
    print(f"{len(dataset)} images, Keras {keras.__version__}, backend: {keras.backend.backend()}")

    gan = GAN(build_generator(), build_discriminator())
    gan.compile(d_optimizer=keras.optimizers.Adam(D_LEARNING_RATE),
                g_optimizer=keras.optimizers.Adam(G_LEARNING_RATE))
    gan.fit(dataset, epochs=args.epochs, batch_size=args.batch_size, callbacks=[
        NoiseSchedule(),
        SaveSamples(args.results, args.checkpoints, args.checkpoint_every),
    ])
    gan.generator.save(args.output)


if __name__ == "__main__":
    main()
