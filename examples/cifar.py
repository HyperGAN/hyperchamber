import hyperchamber as hc
from shared.ops import *

import os
import sys
import time
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from tensorflow.models.image.cifar10 import cifar10_input
import shared.cifar_utils as cifar_utils

start=.0001
end=.01
num=20
hc.permute.set("g_learning_rate", list(np.linspace(start, end, num=num)))
hc.permute.set("d_learning_rate", list(np.linspace(start, end, num=num)))

conv_g_layers = [[32*4, 32*2, 1], [64*4,64*2,1], [16*4,16*2,1]]

conv_d_layers = [[32, 32*2, 32*4],[64, 64*2, 64*4],[64,64*2]]

hc.permute.set("conv_g_layers", conv_g_layers)
hc.permute.set("conv_d_layers", conv_d_layers)

hc.permute.set("z_dim", [32*8, 64*8, 16*8, 64])

hc.permute.set("regularize", [False, True])
hc.permute.set("regularize_lambda", list(np.linspace(0.0001, 1, num=30)))

hc.permute.set("g_batch_norm", [True])
hc.permute.set("d_batch_norm", [True])

hc.permute.set("d_activation", [tf.tanh, tf.nn.relu, tf.nn.relu6, tf.nn.softplus, tf.nn.softsign, tf.sigmoid]);
hc.permute.set("g_activation", [tf.tanh, tf.nn.relu, tf.nn.relu6, tf.nn.softplus, tf.nn.softsign, tf.sigmoid]);

hc.set("epochs", 100)

BATCH_SIZE=64
hc.set("batch_size", BATCH_SIZE)
hc.set("model", "255bits/cifar-gan")
hc.set("version", "0.0.1")


X_DIMS=[32,32]
Y_DIMS=10


def generator(config, y):
    output_shape = X_DIMS[0]*X_DIMS[1]*3
    z = tf.random_uniform([config["batch_size"], 10],0,1)
    result = tf.concat(1, [y, z])
    result = linear(result, config['z_dim'], 'g_input_proj')

    if config['conv_g_layers']:
        result = tf.reshape(result, [config['batch_size'], 4,4,config['z_dim']//16])
        #result = tf.nn.dropout(result, 0.7)
        if(config['g_batch_norm']):
            result = batch_norm(result, name='g_lin_bn')
        result = config['g_activation'](result)
        for i, layer in enumerate(config['conv_g_layers']):
            j=int(result.get_shape()[1]*2)
            k=int(result.get_shape()[2]*2)
            output = [config['batch_size'], j,k,layer]
            result = deconv2d(result, output, scope="g_conv_"+str(i))
            if(config['g_batch_norm']):
                result = batch_norm(result, name='g_conv_bn_'+str(i))
            result = config['g_activation'](result)
        result = tf.reshape(result,[config['batch_size'], -1])

    result = linear(result, output_shape, scope="g_proj")
    result = tf.reshape(result, [config["batch_size"], X_DIMS[0], X_DIMS[1], 3])
    if(config['g_batch_norm']):
        result = batch_norm(result, name='g_lin_bn_out')
    result = tf.nn.sigmoid(result)
    return result

def discriminator(config, x, reuse=False):
    if(reuse):
      tf.get_variable_scope().reuse_variables()
    if config['conv_d_layers']:
        result = tf.reshape(x, [config["batch_size"], X_DIMS[0],X_DIMS[1],3])
        for i, layer in enumerate(config['conv_d_layers']):
            result = conv2d(result, layer, scope='d_conv'+str(i))
            if(config['d_batch_norm']):
                if(i!=0):
                    result = batch_norm(result, name='d_conv_bn_'+str(i))
            result = tf.nn.relu(result)
        result = tf.reshape(x, [config["batch_size"], -1])

    #result = tf.nn.dropout(result, 0.7)
    last_layer = result
    result = linear(result, 11, scope="d_proj")

    return result, last_layer


def create(config, x,y):
    batch_size = config["batch_size"]
    y = tf.one_hot(tf.cast(y,tf.int64), Y_DIMS, 1.0, 0.0)
    print(y)

    d_real, d_last_layer = discriminator(config,x)
    g = generator(config, y)
    d_fake, _ = discriminator(config,g, reuse=True)

    fake_symbol = tf.tile(tf.constant([0,0,0,0,0,0,0,0,0,0,1], dtype=tf.float32), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],11])

    real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])


    d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, fake_symbol)
    d_real_loss = tf.nn.softmax_cross_entropy_with_logits(d_real, real_symbols)

    g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols))
    d_loss = tf.reduce_mean(0.5*d_fake_loss + 0.5*d_real_loss)

    if config['regularize']:
        ws = None
        with tf.variable_scope("g_input_proj"):
            tf.get_variable_scope().reuse_variables()
            ws = tf.get_variable('Matrix')
        lam = config['regularize_lambda']
        g_loss += lam*tf.nn.l2_loss(ws)


    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    print('g_learning_rate', config['g_learning_rate'])
    g_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate'])).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.AdamOptimizer(np.float32(config['d_learning_rate'])).minimize(d_loss, var_list=d_vars)

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

def train(sess, config):
    g_loss = get_tensor("g_loss")
    d_loss = get_tensor("d_loss")
    d_real_loss = get_tensor("d_real_loss")
    d_fake_loss = get_tensor("d_fake_loss")
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")

    _, d_cost, d_real_cost, d_fake_cost = sess.run([d_optimizer, d_loss, d_real_loss, d_fake_loss])
    _, g_cost = sess.run([g_optimizer, g_loss])

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

def sample_input(sess, config):
    x = get_tensor("x")
    sample = sess.run(x)
    return sample[0]


def split_sample(n, sample):
    return [np.reshape(sample[0+i:1+i], [X_DIMS[0],X_DIMS[1], 3]) for i in range(n)]
def samples(sess, config):
    generator = get_tensor("g")
    y = get_tensor("y")
    x = get_tensor("x")
    rand = np.random.randint(0,10, size=config['batch_size'])
    random_one_hot = np.eye(10)[rand]
    x_input = np.random.uniform(0, 1, [config['batch_size'], X_DIMS[0],X_DIMS[1],3])
    sample = sess.run(generator, feed_dict={x:x_input, y:random_one_hot})
    #sample =  np.concatenate(sample, axis=0)
    return split_sample(10, sample)

def plot_mnist_digit(config, image, file):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    #plt.suptitle(config)
    plt.savefig(file)

def epoch(sess, config):
    batch_size = config["batch_size"]
    n_samples =  cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        #x=np.reshape(x, [batch_size, X_DIMS[0], X_DIMS[1], 3])
        train(sess, config)

def test_config(sess, config):
    batch_size = config["batch_size"]
    n_samples =  cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    total_batch = int(n_samples / batch_size)
    results = []
    for i in range(total_batch):
        x=np.reshape(x, [batch_size, X_DIMS[0], X_DIMS[1], 3])
        results.append(test(sess, config))
    return results

def test_epoch(epoch, j, sess, config):
    #x = sample_input(sess, config)
    #sample_file = "samples/input-"+str(j)+".png"
    #cifar_utils.plot(config, x, sample_file)
    
    sample = samples(sess, config)
    sample_list = []
    for s in sample:
        sample_file = "samples/config-"+str(j)+".png"
        cifar_utils.plot(config, s, sample_file)
        sample_list.append(sample_file)
        j+=1
    hc.io.sample(config, sample_list)
    return j

    ranking = 10000
    results =  {
    #    'difficulty':float(difficulty),
        'ranking':float(ranking),
    #    'g_loss':float(g_loss),
    #    'd_fake':float(d_fake),
    #    'd_real':float(d_real),
        }


def record_run(config, results):
    #  results = test_config(sess, config)
    #  loss = np.array(results)
    #  #results = np.reshape(results, [results.shape[1], results.shape[0]])
    #  g_loss = [g for g,_,_ in loss]
    #  g_loss = np.mean(g_loss)
    #  d_fake = [d_ for _,d_,_ in loss]
    #  d_fake = np.mean(d_fake)
    #  d_real = [d for _,_,d in loss]
    #  d_real = np.mean(d_real)
    #  # calculate D.difficulty = reduce_mean(d_loss_fake) - reduce_mean(d_loss_real)
    #  difficulty = d_real * (1-d_fake)
    #  # calculate G.ranking = reduce_mean(g_loss) * D.difficulty
    #  ranking = g_loss * (1.0-difficulty)
    hc.io.record(config, results)



print("Generating configs with hyper search space of ", hc.count_configs())

j=0
k=0
cifar_utils.maybe_download_and_extract()
for config in hc.configs(100):
    print("Testing configuration", config)
    print("TODO: TEST BROKEN")
    sess = tf.Session()
    x,y = cifar_utils.inputs(eval_data=False,data_dir="/tmp/cifar/cifar-10-batches-bin",batch_size=BATCH_SIZE)
    graph = create(config, x,y)
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    for i in range(10):
        epoch(sess, config)
        j=test_epoch(i, j, sess, config)
    #print("results: difficulty %.2f, ranking %.2f, g_loss %.2f, d_fake %.2f, d_real %.2f" % (difficulty, ranking, g_loss, d_fake, d_real))

    record_run(config, results)
    #with g.as_default():
    tf.reset_default_graph()
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

