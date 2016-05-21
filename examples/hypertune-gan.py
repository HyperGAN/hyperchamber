import hyperchamber as hc
from shared.mnist_data import *
from shared.ops import *

import os
import time
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

hc.set("g_learning_rate", .16)
hc.set("d_learning_rate", .08)

g_layers = [ [16, 32], [26,26], 
            [], [256,128,26],[64,32], [32,48,32] 
        ]
d_layers = [ [16, 32], [26,26], 
            [64,32,26], [128,64,26],[32,16], [32,32]
        ]


conv_g_layers = [None for d in d_layers]
conv_g_layers[0]=[10, 1]
conv_g_layers[1]=[16, 1]
conv_g_layers[2]=[10, 1]
conv_g_layers[3]=[8, 1]
conv_g_layers[4]=[4, 1]
conv_g_layers[5]=[6, 1]

conv_d_layers = [None for d in d_layers]
conv_d_layers[0]=[4, 8]
conv_d_layers[1]=[10, 18]
conv_d_layers[2]=[10]
conv_d_layers[3]=[16]
conv_d_layers[4]=[4]
conv_d_layers[5]=[6]

hc.set("conv_g_layers", conv_g_layers)
hc.set("conv_d_layers", conv_d_layers)

hc.set("g_layers", g_layers)
hc.set("d_layers", d_layers)
hc.set("z_dim", 64)

hc.set("batch_size", 256)

X_DIMS=[26,26]
Y_DIMS=10

def generator(config, y):
    output_shape = X_DIMS[0]*X_DIMS[1]
    z = tf.random_uniform([config["batch_size"], 64],0,1)
    result = tf.concat(1, [y, z])
    result = linear(result, 64, 'g_input_proj')

    if config['conv_g_layers']:
        result = tf.reshape(result, [config['batch_size'], 8,8,1])
        for i, layer in enumerate(config['conv_g_layers']):
            j=int(result.get_shape()[1]*2)
            k=int(result.get_shape()[2]*2)
            output = [config['batch_size'], j,k,layer]
            result = deconv2d(result, output, scope="g_conv_"+str(i))
            result = tf.nn.sigmoid(result)
        result = tf.reshape(result,[config['batch_size'], -1])
    else:
        for i, layer in enumerate(config['g_layers']):
            result = linear(result, layer, scope="g_linear_"+str(i))
            result = tf.nn.sigmoid(result)
    if(result.get_shape()[1] != output_shape):
        result = linear(result, output_shape, scope="g_proj")
    result = tf.reshape(result, [config["batch_size"], X_DIMS[0], X_DIMS[1]])
    return result

def discriminator(config, x, reuse=False):
    if(reuse):
      tf.get_variable_scope().reuse_variables()
    if config['conv_d_layers']:
        result = tf.reshape(x, [config["batch_size"], 26,26,1])
        for i, layer in enumerate(config['conv_d_layers']):
            result = conv2d(result, layer, scope='d_conv'+str(i))
            result = tf.nn.relu(result)
        result = tf.reshape(x, [config["batch_size"], -1])
    else:
        result = tf.reshape(x, [config["batch_size"], -1])
        for i, layer in enumerate(config['d_layers']):
            result = linear(result, layer, scope="d_linear_"+str(i))
            result = tf.nn.relu(result)

    #result = linear(x, 128, scope="d_proj0")
    #result = tf.nn.relu(result)
    result = linear(result, 11, scope="d_proj")
    return result
    
def create(config):
    batch_size = config["batch_size"]
    x = tf.placeholder(tf.float32, [batch_size, X_DIMS[0], X_DIMS[1], 1], name="x")
    y = tf.placeholder(tf.float32, [batch_size, Y_DIMS], name="y")

    g = generator(config, y)
    d_fake = discriminator(config,g)
    d_real = discriminator(config,x, reuse=True)
    
    fake_symbol = tf.tile(tf.constant([0,0,0,0,0,0,0,0,0,0,1], dtype=tf.float32), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],11])

    real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])


    d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, fake_symbol)
    d_real_loss = tf.nn.softmax_cross_entropy_with_logits(d_real, real_symbols)

    g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols))
    d_loss = tf.reduce_mean(0.5*d_fake_loss + 0.5*d_real_loss)

    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    g_optimizer = tf.train.GradientDescentOptimizer(config['g_learning_rate']).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.GradientDescentOptimizer(config['d_learning_rate']).minimize(d_loss, var_list=d_vars)

    set_tensor("x", x)
    set_tensor("y", y)
    set_tensor("g_loss", g_loss)
    set_tensor("d_loss", d_loss)
    set_tensor("d_fake_loss", tf.reduce_mean(tf.nn.softmax(d_fake) * (1-fake_symbol)))
    set_tensor("d_real_loss", tf.reduce_mean(tf.nn.softmax(d_real) * (real_symbols)))
    set_tensor("g_optimizer", g_optimizer)
    set_tensor("d_optimizer", d_optimizer)
    set_tensor("g", g)
    set_tensor("d_fake", tf.reduce_mean(d_fake))
    set_tensor("d_real", tf.reduce_mean(d_real))
    
def train(sess, config, x_input, y_input):
    x = get_tensor("x")
    y = get_tensor("y")
    g_loss = get_tensor("g_loss")
    d_loss = get_tensor("d_loss")
    d_real_loss = get_tensor("d_real_loss")
    d_fake_loss = get_tensor("d_fake_loss")
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")

    _, d_cost, d_real_cost, d_fake_cost = sess.run([d_optimizer, d_loss, d_real_loss, d_fake_loss], feed_dict={x:x_input, y:y_input})
    _, g_cost = sess.run([g_optimizer, g_loss], feed_dict={x:x_input, y:y_input})

    print("g cost %.2f d cost %.2f real %.2f fake %.2f" % (g_cost, d_cost, d_real_cost, d_fake_cost))

def test(sess, config, x_input, y_input):
    x = get_tensor("x")
    y = get_tensor("y")
    d_fake = get_tensor("d_fake")
    d_real = get_tensor("d_real")
    g_loss = get_tensor("g_loss")

    g_cost, d_fake_cost, d_real_cost = sess.run([g_loss, d_fake, d_real], feed_dict={x:x_input, y:y_input})


    #hc.event(costs, sample_image = sample[0])
    
    #print("test g_loss %.2f d_fake %.2f d_loss %.2f" % (g_cost, d_fake_cost, d_real_cost))
    return g_cost,d_fake_cost, d_real_cost


def sample(sess, config):
    generator = get_tensor("g")
    y = get_tensor("y")
    rand = np.random.randint(0,10, size=config['batch_size'])
    random_one_hot = np.eye(10)[rand]
    sample = sess.run(generator, feed_dict={y:random_one_hot})
    #sample =  np.concatenate(sample, axis=0)
    return np.reshape(sample[0:4], [26*4,26])

def plot_mnist_digit(config, image, file):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print(np.array(image).shape)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.suptitle(config)
    plt.savefig(file)

def epoch(sess, config, mnist):
    batch_size = config["batch_size"]
    n_samples = mnist.num_examples
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        x, y = mnist.next_batch(batch_size, with_label=True)
        train(sess, config, x, y)

def test_config(sess, config):
    batch_size = config["batch_size"]
    mnist = read_test_sets(one_hot=True)
    n_samples = mnist.num_examples
    total_batch = int(n_samples / batch_size)
    results = []
    for i in range(total_batch):
        x, y = mnist.next_batch(batch_size, with_label=True )
        results.append(test(sess, config, x, y))
    return results


mnist = read_data_sets(one_hot=True)
j=0
for config in hc.configs(100):
    print("Testing configuration", config)
    sess = tf.Session()
    graph = create(config)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(200):
        epoch(sess, config, mnist)
    results = test_config(sess, config)
    loss = np.array(results)
    #results = np.reshape(results, [results.shape[1], results.shape[0]])
    g_loss = [g for g,_,_ in loss]
    g_loss = np.mean(g_loss)
    d_fake = [d_ for _,d_,_ in loss]
    d_fake = np.mean(d_fake)
    d_real = [d for _,_,d in loss]
    d_real = np.mean(d_real)
    # calculate D.difficulty = reduce_mean(d_loss_fake) - reduce_mean(d_loss_real)
    difficulty = d_real * (1-d_fake)
    # calculate G.ranking = reduce_mean(g_loss) * D.difficulty
    ranking = g_loss * (1.0-difficulty)

    s = sample(sess, config)
    plot_mnist_digit(config, s, "samples/config-"+str(j)+".png")
    j+=1
    results =  {
        'difficulty':difficulty,
        'ranking':ranking,
        'g_loss':g_loss,
        'd_fake':d_fake,
        'd_real':d_real,
        }
    print("results: difficulty %.2f, ranking %.2f, g_loss %.2f, d_fake %.2f, d_real %.2f" % (difficulty, ranking, g_loss, d_fake, d_real))
    hc.record(config, results)
    ops.reset_default_graph()
    sess.close()


def by_ranking(x):
    config,result = x
    return result['ranking']

for config, result in hc.top(by_ranking):
    print("RESULTS")
    print(config, result)
    


    #print("Done testing.  Final cost was:", hc.cost())

print("Done")

#for gold, silver, bronze in hc.top_configs(3):

