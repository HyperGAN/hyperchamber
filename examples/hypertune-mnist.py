import hyperchamber as hc
import shared.mnist_data as mnist
from shared.ops import *

import os
import time
import numpy as np
import tensorflow as tf

learning_rates = [ 2e-3, 2e-4 ]
hc.set("learning_rate", learning_rates)
hc.set("adam_beta1", [0.9]*len(learning_rates))

hc.set("batch_size", [128] * len(learning_rates))
hc.set("x_dims", [[26, 26] * len(learning_rates)])
hc.set("y_dims", [[10] * len(learning_rates)])

#def validate(value):
#    return value != value #NaN

#hc.evolve.evolve("learn_rate", 0.2, validate)

def hidden_layers(config, x):
    return x

def output_layer(config, x):
    return linear(x, config["y_dims"])

def create(config):
    batch_size = config["batch_size"]
    x = tf.placeholder(tf.float32, [batch_size, *config["x_dims"]], name="x")
    y = tf.placeholder(tf.float32, [batch_size, config["y_dims"]], name="y")

    hidden = hidden_layers(config, x)
    output = output_layer(config, hidden)

    loss = tf.nn.softmax_cross_entropy_with_logits(output, y) 

    opt = tf.train.AdamOptimizer(loss, beta1=config["adam_beta1"], name="optimizer") \
                                      .minimize(self.d_loss, var_list=self.d_vars)

def train(sess, config, x_input, y_labels):
    x = hc.tf.getTensor("x")
    y = hc.tf.getTensor("y")
    cost = hc.tf.getTensor("cost")
    optimizer = hc.tf.getTensor("optimizer")

    cost = sess.run([cost], feed_dict={x:x_input, y:y_labels})

    #hc.cost(config, cost)
    print("Cost "+str(cost))

def epoch(sess):
    # TODO: load mnist and train on data
    print("TODO")

configs = [
    {
        "learning_rate": 2e-3,
        "adam_beta1": 0.9, 
        "batch_size": 128,
        "x_dims": [26, 26],
        "y_dims": 10
    }
]

for config in configs:
#for config in hc.configs(1):
#    config = config[0]
    print("Testing configuration", config)
    sess = tf.Session()
    graph = create(config)
    epoch(sess)
    sess.close()
    print("Done testing.  Final cost was:", hc.cost())


for gold, silver, bronze in hc.top_configs(3):
    print("Gold medal with: %.2f  " % gold.cost, gold.config)
    print("Silver medal with: %.2f  " % silver.cost, silver.config)
    print("Bronze medal with: %.2f  " % bronze.cost, bronze.config)
    

