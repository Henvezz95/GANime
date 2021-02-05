# example of training an unconditional gan on the fashion mnist dataset
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.models import load_model
import cv2, os
import numpy as np
import keras

#Generates a batch of random gaussian noise vectors to use as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.normal(0,1,size=(latent_dim * n_samples))
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def show_plot(examples, m, n):
    for i in range(m * n):
        # define subplot
        plt.subplot(m, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        img = cv2.cvtColor((np.clip((examples[i,:,:,:]+1)*127.5,0,255)).astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.pause(10)

m=6
n=4
latent_dim = 100

model = load_model('trained_generator.h5')
latent_points = generate_latent_points(latent_dim, m*n)
X = model.predict(latent_points)
show_plot(X, m, n)
plt.show()

latent_points = generate_latent_points(latent_dim, m*n)
X = model.predict(latent_points)
show_plot(X, m, n)
plt.show()