import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from shared.variational_autoencoder import *
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave

import hyperchamber as hc

hc.set("model", "255bits/vae-mnist")

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

# only works for n_z=2
def visualize(config, vae):
    if(config['n_z'] != 2):
        print("Skipping visuals since n_z is not 2")
        return
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            x_mean = vae.generate(np.tile(z_mu, [config['batch_size'], 1]))
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()

    img = "samples/2d-visualization.png"
    plt.savefig(img)
    hc.io.sample(config, [{"label": "2d visualization", "image": img}])

def train(config, vae, learning_rate=0.001,
          training_epochs=10, display_step=5):

   batch_size=config['batch_size']
   # Training cycle
   for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
   return avg_cost

def sample(config, vae):
    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = vae.reconstruct(x_sample)

    plt.figure(figsize=(8, 12))
    for i in range(5):

        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    img = "samples/reconstruction.png"
    plt.savefig(img)
    hc.io.sample(config, [{"label": "Reconstruction", "image": img}])

hc.permute.set("learning_rate", list(np.linspace(0.0001, 0.003, num=10, dtype=np.float32)))
hc.permute.set("n_hidden_recog_1", list(np.linspace(100, 1000, num=10, dtype=np.int32)))
hc.permute.set("n_hidden_recog_2", list(np.linspace(100, 1000, num=10, dtype=np.int32)))
hc.permute.set("n_hidden_gener_1", list(np.linspace(100, 1000, num=10, dtype=np.int32)))
hc.permute.set("n_hidden_gener_2", list(np.linspace(100, 1000, num=10, dtype=np.int32)))

hc.set("n_input", 784) # MNIST data input (img shape: 28*28)
hc.permute.set("n_z", [1,2,4,8,16,32,64,128]) # dimensionality of latent space
hc.set('batch_size', 100)
hc.permute.set("transfer_fct", [tf.tanh, tf.nn.elu, tf.nn.relu, tf.nn.relu6, tf.nn.softplus, tf.nn.softsign]);

hc.set("epochs", 10)

print("hypersearch space", hc.count_configs())
for config in hc.configs(10000):
    print("Config", config)
    vae = VariationalAutoencoder(config, 
                                 learning_rate=config['learning_rate'],
                                 transfer_fct=config['transfer_fct'])
    cost = 1000
    for i in range(config['epochs']):
        cost = train(config, vae, training_epochs=1)
        sample(config, vae)
    hc.io.record(config, {"rank": cost})
    visualize(config, vae)
