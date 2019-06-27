import timeit
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

from plots import plot_digit, plot_pca
from mlps import mlp1, mlp2, mlp3
from pca import apply_pca

start = timeit.default_timer()

print('\nreading MNIST dataset from tensorflow sample...')
mnist = input_data.read_data_sets("MNIST_data/")
# mnist = tf.keras.datasets.mnist(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
t_x_full, t_y = mnist.train.images, mnist.train.labels
x_full, y = mnist.test.images, mnist.test.labels

print('\nploting random MNIST sample to prove that dataset loaded properly and understand data...')
plot_digit(x_full, y, random.randint(1, 1001))

print('\napplying PCA 1 to dataset...')
x_pca1 = apply_pca(x_full, t_x_full)
plot_pca(x_pca1, y, "1")

print('\napplying PCA 2 to dataset...')
x_pca2 = apply_pca(x_pca1, t_x_full)
plot_pca(x_pca2, y, "2")

# apply MLP

print("\napply MLPs and display accuracy to each...")

print("\n------------------------------------------------")
print("\n---\n[RES] Full: \n")
mlp1.fit(x_full, y)
mlp2.fit(x_full, y)
mlp3.fit(x_full, y)
print('[RES] MLP 1: %.2f%%' % (100 * mlp1.score(x_full, y)))
print('[RES] MLP 2: %.2f%%' % (100 * mlp2.score(x_full, y)))
print('[RES] MLP 3: %.2f%%' % (100 * mlp3.score(x_full, y)))
print("\n---\n[RES] PCA 1: \n")
mlp1.fit(x_pca1, y)
mlp2.fit(x_pca1, y)
mlp3.fit(x_pca1, y)
print('[RES] MLP 1: %.2f%%' % (100 * mlp1.score(x_pca1, y)))
print('[RES] MLP 2: %.2f%%' % (100 * mlp2.score(x_pca1, y)))
print('[RES] MLP 3: %.2f%%' % (100 * mlp3.score(x_pca1, y)))
print("\n---\n[RES] PCA 2: \n")
mlp1.fit(x_pca2, y)
mlp2.fit(x_pca2, y)
mlp3.fit(x_pca2, y)
print('[RES] MLP 1: %.2f%%' % (100 * mlp1.score(x_pca2, y)))
print('[RES] MLP 2: %.2f%%' % (100 * mlp2.score(x_pca2, y)))
print('[RES] MLP 3: %.2f%%' % (100 * mlp3.score(x_pca2, y)))

print("\n------------------------------------------------")

stop = timeit.default_timer()
print('\nExecution time: ', stop - start)  
