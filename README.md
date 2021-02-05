# GANime
CGAN to generate anime-like faces. The training set used can be found [here](https://www.kaggle.com/splcher/animefacedataset).
</br>
This is an example of a batch of 24 images generated by the convolutional network. Every image as a resolution of 64x64.
</br>
<img src="Examples.png"
     alt="Generated Anime Faces"
     style="float: left; margin-top: 50px; margin-bottom: 50px;" />
     

# Architecture
These are the arcitectures choosen for the generator and the discriminator:

</br>
<img src="Architecture.png"
     alt="Layer diagram"
     style="float: left; margin-top: 50px; margin-bottom: 50px;" />

# Strategies for a stable training
For a stable GAN training, I combined different strategies that are known to improve GAN training. I took inspiration from this extremely useful guide [https://github.com/soumith/ganhacks](https://github.com/soumith/ganhacks).
These are the implemeented startegies:
* Input images are normalized between -1 and 1
* The last layer of the generator uses Tanh as the activation function
* The loss function for the generator is `-log(D)` instead of `log(1-D)` (This is achieved by flipping labels when training generator: real = fake, fake = real)
* Input noise is Gaussian, not Uniform
* The discriminator is trained with a batch of only real images, then with a batch of only fake images
* Both generator and discriminator implement batch normalization layers
* Sparse Gradients are avoided:
     - Leaky ReLU is used instead of ReLU (with alpha=0.1 by default)
     - Strided convolution is used instead of MaxPooling
     - Transposed Convolution is used instead of Upsampling
* Label smoothing is supported (in the default implementation is On, but is very subtle. Fake images have a label between 0 and 0.05, real images have a label between 0.95 and 1)
* Adam optimizer with small learning rate
* <b>Added Gaussian Noise to the Discriminator's inputs:</b> the initial noise standard deviation is very high (0.75) and decreases linearly during training by 0.02 per epoch until reaching 0. It makes the training much more stable because it slows down the initial training phase, especially for the discriminator. When the noise is strong, the fine details are unrecognizable and both generator and discriminator must focus on the overall structure of the image. Additional resources on the topic can be found here [http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/](http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/) and here [https://openreview.net/forum?id=Hk4_qw5xe](https://openreview.net/forum?id=Hk4_qw5xe).
* Lower learning rate for the generator than the discriminator makes mode collapse improbable (the generator cannot "run away" from the discriminator jumping from one mode to another)
* Scheduling: if the generator has a loss that is too high it is trained with an additional batch (to avoid discriminator dominating)


# How to train the generator

# How to Test the generator

# Possible improvements
