"""Generator, discriminator and combined GAN for 64x64 anime faces (Keras 3)."""
import keras
from keras import layers, ops

IMAGE_SHAPE = (64, 64, 3)
LATENT_DIM = 100
BN_MOMENTUM = 0.85
LEAKY_SLOPE = 0.1


@keras.saving.register_keras_serializable(package="ganime")
class AnnealedNoise(layers.Layer):
    """Gaussian noise (training only) whose std can be changed without rebuilding the model.

    Noise on the discriminator's input stabilises training. The std is decayed epoch after
    epoch, so it is stored in a variable instead of being a constant of the layer.
    """

    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.initial_stddev = float(stddev)
        self.seed_generator = keras.random.SeedGenerator()

    def build(self, input_shape):
        self.stddev = self.add_weight(
            name="stddev", shape=(), trainable=False,
            initializer=keras.initializers.Constant(self.initial_stddev))

    def set_stddev(self, value):
        self.stddev.assign(value)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        noise = keras.random.normal(ops.shape(inputs), stddev=1.0, seed=self.seed_generator)
        return inputs + noise * self.stddev

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "stddev": self.initial_stddev}


def _conv_block(x, filters, strides=1):
    x = layers.Conv2D(filters, 3, strides=strides, padding="same")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    return layers.LeakyReLU(negative_slope=LEAKY_SLOPE)(x)


def _deconv_block(x, filters):
    x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    return layers.LeakyReLU(negative_slope=LEAKY_SLOPE)(x)


def build_generator(latent_dim=LATENT_DIM):
    """Latent vector -> 64x64 RGB image in [-1, 1]."""
    z = keras.Input((latent_dim,))
    x = layers.Dense(4 * 4 * 512)(z)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_SLOPE)(x)
    x = layers.Reshape((4, 4, 512))(x)
    for filters in (512, 256, 128, 64):  # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        x = _deconv_block(x, filters)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_SLOPE)(x)
    img = layers.Conv2D(3, 7, padding="same", activation="tanh")(x)
    return keras.Model(z, img, name="generator")


def load_generator(path):
    """Load a generator saved as .keras, or one saved as .h5 by the old Keras 2 scripts."""
    path = str(path)
    if path.endswith(".h5"):
        # Keras 3 misreads the input shape stored in old .h5 files: rebuild the model, load the weights
        generator = build_generator()
        generator.load_weights(path)
        return generator
    return keras.saving.load_model(path, compile=False)


def build_discriminator(noise_std=0.6):
    """64x64 RGB image -> probability that the image is real."""
    img = keras.Input(IMAGE_SHAPE)
    x = AnnealedNoise(noise_std, name="input_noise")(img)
    x = _conv_block(x, 64)  # 64x64
    for filters in (64, 128, 256, 512):  # 32x32 -> 16x16 -> 8x8 -> 4x4
        x = _conv_block(x, filters, strides=2)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(192)(x)
    x = layers.LeakyReLU(negative_slope=LEAKY_SLOPE)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(img, out, name="discriminator")


def _loss_and_gradients(loss_fn, variables):
    """Backend-specific part of the training step: evaluate `loss_fn` and differentiate it."""
    backend = keras.backend.backend()
    if backend == "tensorflow":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = loss_fn()
        return loss, tape.gradient(loss, variables)
    if backend == "torch":
        import torch
        loss = loss_fn()
        return loss, torch.autograd.grad(loss, [v.value for v in variables])
    raise NotImplementedError(f"The '{backend}' backend is not supported, use tensorflow or torch")


class GAN(keras.Model):
    """Generator + discriminator trained together. `fit` takes uint8 images in [0, 255]."""

    def __init__(self, generator, discriminator, label_smoothing=0.05, generator_loss_limit=3.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = generator.input_shape[-1]
        self.label_smoothing = label_smoothing
        # When the generator's loss is above this limit it gets an extra update
        self.generator_loss_limit = generator_loss_limit
        self.seed_generator = keras.random.SeedGenerator()
        self.built = True  # the sub-models are already built, and the GAN itself is never called

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def _latent_points(self, n):
        return keras.random.normal((n, self.latent_dim), seed=self.seed_generator)

    def _smooth_labels(self, n, real):
        """One-sided noisy labels: real in [1 - s, 1], fake in [0, s]."""
        noise = keras.random.uniform((n, 1), maxval=self.label_smoothing, seed=self.seed_generator)
        return 1.0 - noise if real else noise

    def _discriminator_step(self, real_images):
        n = ops.shape(real_images)[0]
        fake_images = self.generator(self._latent_points(n), training=True)
        real_labels, fake_labels = self._smooth_labels(n, True), self._smooth_labels(n, False)

        def loss_on(images, labels):
            predictions = self.discriminator(images, training=True)
            return ops.mean(keras.losses.binary_crossentropy(labels, predictions))

        # The discriminator sees real and generated images in two separate batches
        losses = []
        for images, labels in ((real_images, real_labels), (fake_images, fake_labels)):
            variables = self.discriminator.trainable_variables
            loss, grads = _loss_and_gradients(lambda: loss_on(images, labels), variables)
            self.d_optimizer.apply(grads, variables)
            losses.append(loss)
        return losses

    def _generator_step(self, n):
        """Updates only the generator, so that the discriminator classifies its images as real."""
        def loss_fn():
            fake_images = self.generator(self._latent_points(n), training=True)
            predictions = self.discriminator(fake_images, training=True)
            return ops.mean(keras.losses.binary_crossentropy(ops.ones((n, 1)), predictions))

        variables = self.generator.trainable_variables
        loss, grads = _loss_and_gradients(loss_fn, variables)
        self.g_optimizer.apply(grads, variables)
        return loss

    def train_step(self, data):
        real_images = ops.cast(data, "float32") / 127.5 - 1.0
        d_loss_real, d_loss_fake = self._discriminator_step(real_images)
        n = ops.shape(real_images)[0]
        g_loss = self._generator_step(n)
        g_loss = ops.cond(g_loss > self.generator_loss_limit,
                          lambda: self._generator_step(n), lambda: g_loss)
        return {"d_real": d_loss_real, "d_fake": d_loss_fake, "g": g_loss}
