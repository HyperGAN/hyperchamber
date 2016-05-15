
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

tensors = {}
def get_tensor(name, graph=tf.get_default_graph(), isOperation=False):
  return tensors[name]# || graph.as_graph_element(name)

def set_tensor(name, tensor):
  tensors[name]=tensor
