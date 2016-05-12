import ../hyperchamber as hc

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

def create():
    batch_size = hc.get("batch_size")
    x = tf.placeholder(tf.float32, [batch_size, *hc.get("x_size")], name="x")
    y = tf.placeholder(tf.float32, [batch_size, *hc.get("y_size")], name="y")

    # hidden layers
    # output layer
    # adam optimizer
    # cost = Xent

def train(sess, x_input, y_labels):
    x = hc.tf.getTensor("x")
    y = hc.tf.getTensor("y")
    cost = hc.tf.getTensor("cost")

    result = sess.run([cost], feed_dict={x:x_input, y:y_labels})

    hc.append("cost", result)


    

