
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

def deconv2d(input_, output_shape,
        k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, bias_start=0.0, padding='SAME',
        scope="deconv2d", with_w=False,reuse=False):
    with tf.variable_scope(scope):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.contrib.layers.xavier_initializer())

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, padding=padding,
                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(bias_start))
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
                initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def batch_norm(x, name):
    return (batch_norm_(name=name))(x)

class batch_norm_(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


tensors = {}
def get_tensor(name, graph=tf.get_default_graph(), isOperation=False):
  return tensors[name]# || graph.as_graph_element(name)

def set_tensor(name, tensor):
  tensors[name]=tensor
