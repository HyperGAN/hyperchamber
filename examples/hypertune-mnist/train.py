import ../../hyperchamber as hc

import os
import time
import numpy as np
import tensorflow as tf

hc.set("batch_size", 128)
hc.set("x_dims", [26, 26])
hc.set("y_dims", [19])

def validate(value):
    return value != value #NaN

hc.evolve.evolve("learn_rate", 0.2, validate)

def hidden_layers(x):
    return x*w + b

def create():
    batch_size = hc.get("batch_size")
    x = tf.placeholder(tf.float32, [batch_size, *hc.get("x_size")], name="x")
    y = tf.placeholder(tf.float32, [batch_size, *hc.get("y_size")], name="y")

        hidden = hidden_layers(x)
        output = output_layer(hidden)


        opt = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1, name="optimizer") \
                                      .minimize(self.d_loss, var_list=self.d_vars)
    # cost = Xent

def train(sess, x_input, y_labels):
    x = hc.tf.getTensor("x")
    y = hc.tf.getTensor("y")
    cost = hc.tf.getTensor("cost")
    optimizer = hc.tf.getTensor("optimizer")

    cost = sess.run([cost], feed_dict={x:x_input, y:y_labels})

    hc.cost(cost)
        print("Cost "+str(cost))

    def epoch(sess):
        # TODO: load mnist and train on data

for config in hc.evolve.configs(8):
    print("Testing configuration", config)
    sess = tf.Session()
    epoch(sess)
    sess.close()
    print("Done testing.  Final cost was:", hc.cost())


for gold, silver, bronze in hc.top_configs(3):
    print("Gold medal with: %.2f  " % gold.cost, gold.config)
    print("Silver medal with: %.2f  " % silver.cost, silver.config)
    print("Bronze medal with: %.2f  " % bronze.cost, bronze.config)
    

