import hyperchamber as hc
from shared.mnist_data import *
from shared.ops import *

import os
import time
import numpy as np
import tensorflow as tf

learning_rates = [ 2e-3 ]
hc.set("learning_rate", learning_rates)
hc.set("adam_beta1", [0.9])

hc.set("batch_size", [128])
hc.set("x_dims", [[26, 26]])
hc.set("y_dims", [10])

#def validate(value):
#    return value != value #NaN

#hc.evolve.evolve("learn_rate", 0.2, validate)

def hidden_layers(config, x):
    output = tf.reshape(x, [config["batch_size"], config["x_dims"][0]*config["x_dims"][1]])
    output = linear(output, 26*26, scope="l2")
    output = tf.nn.tanh(output)
    return output

def output_layer(config, x):
    return linear(x, config["y_dims"])

def create(config):
    batch_size = config["batch_size"]
    x = tf.placeholder(tf.float32, [batch_size, config["x_dims"][0], config["x_dims"][1], 1], name="x")
    y = tf.placeholder(tf.float32, [batch_size, config["y_dims"]], name="y")

    hidden = hidden_layers(config, x)
    output = output_layer(config, hidden)

    #output = tf.nn.softmax(output)
    #loss  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=[1]))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y), name="loss")

    output = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variables = tf.trainable_variables()

    #optimizer = tf.train.AdamOptimizer(loss, beta1=config["adam_beta1"], name="optimizer") \
    #                                  .minimize(loss, var_list=variables)
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


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
    print("Accuracy %.2f Cost %.2f" % (accuracy, cost))

def epoch(sess, config):
    batch_size = config["batch_size"]
    mnist = read_data_sets(one_hot=True)
    n_samples = mnist.num_examples
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        x, y = mnist.next_batch(batch_size, with_label=True)
        train(sess, config, x, y)

for config in hc.configs(1):
    print(config)
    print("Testing configuration", config)
    sess = tf.Session()
    graph = create(config)
    init = tf.initialize_all_variables()
    sess.run(init)
    epoch(sess, config)
    sess.close()
    #print("Done testing.  Final cost was:", hc.cost())

print("Done")

#for gold, silver, bronze in hc.top_configs(3):
#    print("Gold medal with: %.2f  " % gold.cost, gold.config)
#    print("Silver medal with: %.2f  " % silver.cost, silver.config)
#    print("Bronze medal with: %.2f  " % bronze.cost, bronze.config)
    

