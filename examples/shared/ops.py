
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

def deconv2d(input_, output_shape,
        k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, biasstart=0.0, padding='SAME',
        scope="deconv2d", with_w=False,reuse=False):
    with tf.variable_scope(scope):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, padding=padding,
                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(biasstart))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv



def conv2d(input_, output_dim, 
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME',
        scope="conv2d", reuse=False):
    with tf.variable_scope(scope):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


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

def get_tensors(name):
    return tensors_list[name]

tensors_list={}
def set_tensors(name, tensor):
    if(name in tensors_list):
        tensors_list[name].append(tensor)
    else:
        tensors_list[name]=[tensor]


def set_tensor(name, tensor):
  tensors[name]=tensor


def shared_placeholder(dtype, dims, name):
    if(name not in tensors):
        tensors[name] = tf.placeholder(dtype, dims, name=name)
    return tensors[name]

