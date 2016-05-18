import hyperchamber as hc
from shared.mnist_data import *
from shared.ops import *

import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

hc.set("g_learning_rate", 0.7)
hc.set("d_learning_rate", 0.7)
hc.set("g_layers", [10])
hc.set("d_layers", [20])
hc.set("z_dim", 64)

hc.set("batch_size", 128)

X_DIMS=[26,26]

def generator(config):
    z = tf.random_uniform([config["batch_size"], 64],0,1)
    result = linear(z, X_DIMS[0]*X_DIMS[1], scope="g_proj0")
    result = tf.nn.tanh(result)

    result = linear(result, X_DIMS[0]*X_DIMS[1], scope="g_proj")
    result = tf.reshape(result, [config["batch_size"], X_DIMS[0], X_DIMS[1]])
    return result

def discriminator(config, x, reuse=False):
    if(reuse):
      tf.get_variable_scope().reuse_variables()
    x = tf.reshape(x, [config["batch_size"], -1])
    result = linear(x, 128, scope="d_proj0")
    result = tf.nn.tanh(result)
    result = linear(result, 1, scope="d_proj")
    return result
    
def create(config):
    batch_size = config["batch_size"]
    x = tf.placeholder(tf.float32, [batch_size, X_DIMS[0], X_DIMS[1], 1], name="x")

    g = generator(config)
    d_fake = discriminator(config,g)
    d_real = discriminator(config,x, reuse=True)
    
    d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake, tf.zeros_like(d_fake))
    d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real, tf.ones_like(d_real))

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake, tf.ones_like(d_fake)))
    d_loss = tf.reduce_mean(0.5*d_fake_loss + 0.5*d_real_loss)

    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    g_optimizer = tf.train.GradientDescentOptimizer(config['g_learning_rate']).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.GradientDescentOptimizer(config['d_learning_rate']).minimize(d_loss, var_list=d_vars)

    set_tensor("x", x)
    set_tensor("g_loss", g_loss)
    set_tensor("d_loss", d_loss)
    set_tensor("d_fake_loss", tf.reduce_mean(d_fake_loss))
    set_tensor("d_real_loss", tf.reduce_mean(d_real_loss))
    set_tensor("g_optimizer", g_optimizer)
    set_tensor("d_optimizer", d_optimizer)
    set_tensor("g", g)
    set_tensor("d_fake", tf.reduce_mean(tf.nn.sigmoid(d_fake)))
    set_tensor("d_real", tf.reduce_mean(tf.nn.sigmoid(d_real)))
    
def train(sess, config, x_input):
    x = get_tensor("x")
    g_loss = get_tensor("g_loss")
    d_loss = get_tensor("d_loss")
    d_real_loss = get_tensor("d_real_loss")
    d_fake_loss = get_tensor("d_fake_loss")
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")

    _, d_cost, d_real_cost, d_fake_cost = sess.run([d_optimizer, d_loss, d_real_loss, d_fake_loss], feed_dict={x:x_input})
    _, g_cost = sess.run([g_optimizer, g_loss], feed_dict={x:x_input})

    print("g cost %.2f d cost %.2f real %.2f fake %.2f" % (g_cost, d_cost, d_real_cost, d_fake_cost))

def test(sess, config, x_input, y_labels):
    x = get_tensor("x")
    d_fake = get_tensor("d_fake")
    d_real = get_tensor("d_real")

    d_fake_cost, d_real_cost = sess.run([d_fake, d_real], feed_dict={x:x_input})


    print("test g_loss %.2f d_loss %.2f" % (d_fake_cost, d_real_cost))
    return d_fake_cost, d_real_cost


def epoch(sess, config):
    batch_size = config["batch_size"]
    mnist = read_data_sets(one_hot=True)
    n_samples = mnist.num_examples
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        x, y = mnist.next_batch(batch_size, with_label=True)
        train(sess, config, x)

def test_config(sess, config):
    batch_size = config["batch_size"]
    mnist = read_test_sets(one_hot=True)
    n_samples = mnist.num_examples
    total_batch = int(n_samples / batch_size)
    accuracies = []
    costs = []
    for i in range(total_batch):
        x, y = mnist.next_batch(batch_size, with_label=True )
        accuracy, cost = test(sess, config, x, y)
        accuracies.append(accuracy)
        costs.append(cost)
    return accuracies, costs


for config in hc.configs(100):
    print("Testing configuration", config)
    sess = tf.Session()
    graph = create(config)
    init = tf.initialize_all_variables()
    sess.run(init)
    epoch(sess, config)
    accuracies, costs = test_config(sess, config)

    # calculate D.difficulty = reduce_mean(d_loss) - reduce_mean(d_loss_fake)
    # calculate G.ranking = reduce_mean(g_loss) * D.difficulty
    accuracy, cost = np.mean(accuracies), np.mean(costs)
    results =  {
        'accuracy':accuracy,
        'cost':cost
        }
    hc.record(config, results)
    ops.reset_default_graph()
    sess.close()


def by_accuracy(x):
    config,result = x
    return 1-result['accuracy']

for config, result in hc.top(by_accuracy):
    print("RESULTS")
    print(config, result)
    


    #print("Done testing.  Final cost was:", hc.cost())

print("Done")

#for gold, silver, bronze in hc.top_configs(3):

