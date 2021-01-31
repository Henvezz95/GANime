# example of training an unconditional gan on the fashion mnist dataset
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from glob import glob
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, UpSampling2D
from keras.layers import Reshape
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import Conv2D, GaussianNoise, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
import cv2, os
import numpy as np
import keras

#This function defines the discriminator model. The discriminator has a gaussian noise layer attached to the input
#It is used to help training stability. The variable std is the standard deviation of the network
def define_discriminator(in_shape=(64,64,3), std=0.6, opt=None):
    model = Sequential()
    model.add(GaussianNoise(stddev=std, input_shape=in_shape))
    # Resolution 64x64
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    # Resolution 32x32
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    # Resolution 16x16
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    # Resolution 8x8
    model.add(Conv2D(512, (3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    # Resolution 4x4
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(192))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    if opt == None:
        opt = Adam(lr=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#This function defines the generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 512 * 4 *4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Reshape((4, 4, 512)))
    #Resolution: 4x4x512
    model.add(Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    #Resolution: 8x8x512
    model.add(Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    #Resolution: 16x16x256
    model.add(Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    #Resolution: 32x32x128
    model.add(Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.85))
    model.add(LeakyReLU(alpha=0.1))
    #Resolution: 64x64x64
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))

    # generate
    model.add(Conv2D(3, (7,7), padding='same', activation='tanh'))
    return model

#Defines the combined generator and discriminator model, used for updating the generator
def define_gan(generator, discriminator, opt=None):
    #Weights in the discriminator are set has not-trainable
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # compile model
    if opt==None:
        opt = Adam(lr=0.000005)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#Generates a batch of real examples with labels. It supports uniformely distributed label noise
def generate_real_samples(dataset, n_samples, max_label_noise=0.05):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))-np.random.uniform(0, max_label_noise)
    return X, y

#Generates a batch of random gaussian noise vectors to use as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.normal(0,1,size=(latent_dim * n_samples))
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

#Generates n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, max_label_noise=0.05):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = zeros((n_samples, 1))+np.random.uniform(0, max_label_noise)
    return X, y

def show_plot(examples, m, n):
    for i in range(m * n):
        # define subplot
        plt.subplot(m, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        img = cv2.cvtColor((np.clip((examples[i,:,:,:]+1)*127.5,0,255)).astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.pause(0.01)

def train_epoch(g_model, d_model, gan_model, dataset, latent_dim, batch_size=128):
    bat_per_epo = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    # enumerate batches over the training set
    for j in range(bat_per_epo):
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss1, _ = d_model.train_on_batch(X_real, y_real)
        d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
        X_gan = generate_latent_points(latent_dim, batch_size)
        y_gan = ones((batch_size, 1))
        # update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)            
        #If the generator's loss is too high, the generator will be trained on one more batch
        if g_loss > 3.0:
            X_gan = generate_latent_points(latent_dim, batch_size)
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('double batch')
        print('%d/%d, d1=%.3f, d2=%.3f, g=%.3f' %
                (j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

    latent_points = generate_latent_points(100, 24)
    X = generator.predict(latent_points)
    show_plot(X, 6, 4)


#The main starts here
file_list = glob('images/*.jpg')

#Select only the train images that have a resolution greater or equal to 64x64
#To avoid the generator to produce lower resolution images
img_paths = []
for element in file_list:
    img = cv2.imread(element, cv2.IMREAD_GRAYSCALE)
    try:
        H, W = img.shape
    except:
        continue
    if H>=64 and W>=64:
        img_paths.append(element)

#The memory of the training set is allocated before loading the images
dataset = np.zeros((len(img_paths), 64, 64, 3), dtype=np.float32)
for i in range(len(img_paths)):
    dataset[i] = cv2.resize(cv2.imread(img_paths[i]), (64,64), cv2.INTER_CUBIC)/127.5 - 1.0

plt.ion()
latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

for i in range(300):
    sigma = max(0.75-(i+0)*0.02, 0.000001)
    print('Epoch: ',i+1, ', Noise standard deviation:  ', sigma)
    d_weights = discriminator.get_weights()
    dopt = discriminator.optimizer
    g_weights = generator.get_weights()
    gan_weights = gan_model.get_weights()
    ganopt = gan_model.optimizer
    
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del discriminator
    del generator
    del gan_model
    
    discriminator = define_discriminator(std=sigma, opt=dopt)
    discriminator.set_weights(d_weights)
    
    generator = define_generator(latent_dim)
    generator.set_weights(g_weights)
    
    gan_model = define_gan(generator, discriminator, opt=ganopt)
    gan_model.set_weights(gan_weights)
   
    train_epoch(generator, discriminator, gan_model, dataset, latent_dim, batch_size=128)
    if i%10 == 9:
        generator.save('checkpoint models/anime_generator_epoch',i+1,'.h5')
generator.save('trained_generator.h5')