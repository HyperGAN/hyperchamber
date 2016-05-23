import hyperchamber as hc
from shared.ops import *

import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rates = [1, 0.8, 0.75, 0.75, 0.5, 0.25, 0.125]
hc.set("learning_rate", learning_rates)
hidden_layers = [ [],[26*26, 26*26], [26*26],  [128], [64, 64], [16, 32], [26,26] ]
hc.set("hidden_layer", hidden_layers)

hc.set("batch_size", 128)

X_DIMS=[28,28]
Y_DIMS=10

#def validate(value):
#    return value != value #NaN

#hc.evolve.evolve("learn_rate", 0.2, validate)

def hidden_layers(config, x):
    output = tf.reshape(x, [config["batch_size"], X_DIMS[0]*X_DIMS[1]])
    for i, layer in enumerate(config['hidden_layer']):
        output = linear(output, layer, scope="l"+str(i))
        output = tf.nn.tanh(output)
    return output

def output_layer(config, x):
    return linear(x, Y_DIMS)

def create(config):
    batch_size = config["batch_size"]
    x = tf.placeholder(tf.float32, [batch_size, X_DIMS[0], X_DIMS[1], 1], name="x")
    y = tf.placeholder(tf.float32, [batch_size, Y_DIMS], name="y")

    hidden = hidden_layers(config, x)
    output = output_layer(config, hidden)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y), name="loss")

    output = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variables = tf.trainable_variables()

    optimizer = tf.train.GradientDescentOptimizer(config['learning_rate']).minimize(loss)


    set_tensor("x", x)
    set_tensor("y", y)
    set_tensor("loss", loss)
    set_tensor("optimizer", optimizer)
    set_tensor("accuracy", accuracy)
    
def train(sess, config, x_input, y_labels):
    x = get_tensor("x")
    y = get_tensor("y")
    cost = get_tensor("loss")
    optimizer = get_tensor("optimizer")
    accuracy = get_tensor("accuracy")

    _, accuracy, cost = sess.run([optimizer, accuracy, cost], feed_dict={x:x_input, y:y_labels})


    #hc.cost(config, cost)
    #print("Accuracy %.2f Cost %.2f" % (accuracy, cost))

def test(sess, config, x_input, y_labels):
    x = get_tensor("x")
    y = get_tensor("y")
    cost = get_tensor("loss")
    accuracy = get_tensor("accuracy")

    accuracy, cost = sess.run([accuracy, cost], feed_dict={x:x_input, y:y_labels})


    print("Accuracy %.2f Cost %.2f" % (accuracy, cost))
    return accuracy, cost


def epoch(sess, config):
    batch_size = config["batch_size"]
    n_samples = mnist.train.num_examples
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        x, y = mnist.train.next_batch(batch_size)
        x=np.reshape(x, [batch_size, X_DIMS[0], X_DIMS[1], 1])
        train(sess, config, x, y)

def test_config(sess, config):
    batch_size = config["batch_size"]
    n_samples = mnist.test.num_examples
    total_batch = int(n_samples / batch_size)
    accuracies = []
    costs = []
    for i in range(total_batch):
        x, y = mnist.test.next_batch(batch_size)
        x=np.reshape(x, [batch_size, X_DIMS[0], X_DIMS[1], 1])
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
    for i in range(10):
        epoch(sess, config)
    accuracies, costs = test_config(sess, config)
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

