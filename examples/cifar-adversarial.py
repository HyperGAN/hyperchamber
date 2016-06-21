import hyperchamber as hc
from shared.ops import *
import shared

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
import argparse

parser = argparse.ArgumentParser(description='Runs the GAN.')
parser.add_argument('--load_config', type=str)
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--d_batch_norm', type=bool)
parser.add_argument('--no_stop', type=bool)

args = parser.parse_args()
start=.0002
end=.01
num=20
hc.set("g_learning_rate", list(np.linspace(start, end, num=num)))
hc.set("d_learning_rate", list(np.linspace(start, end, num=num)))

hc.set("n_hidden_recog_1", list(np.linspace(100, 1000, num=100)))
hc.set("n_hidden_recog_2", list(np.linspace(100, 1000, num=100)))
hc.set("transfer_fct", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("d_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("g_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("last_layer", [lrelu_2]);

hc.set("n_input", 32*32*3)

conv_g_layers = [[i*8, i*4, 3] for i in [16,32]]
conv_g_layers = [[i*8, i*4, i*2, 3] for i in [16,32]]
conv_g_layers += [[i*8, i*4, i*2, i] for i in [16,32]]
conv_g_layers += [[i*16, i*8, i*4, i*2, 3] for i in [8, 16]]
conv_g_layers += [[i*16, i*8, i*4, i*2, i, 3] for i in [4, 6, 8]]

conv_g_layers+=[[i*16,i*8, i*4, 3] for i in list(np.arange(2, 16))]

conv_d_layers = [[i, i*2, i*4, i*8] for i in list(np.arange(32, 128))] 
conv_d_layers += [[i, i*2, i*4, i*8] for i in list(np.arange(16,32))] 
conv_d_layers += [[i, i*2, i*4, i*8, i*16] for i in [12, 16, 32, 64]] 
#conv_d_layers = [[32, 32*2, 32*4],[32, 64, 64*2],[64,64*2], [16,16*2, 16*4], [16,16*2]]

hc.set("conv_size", [3, 4, 5])
hc.set("d_conv_size", [3, 4, 5])
hc.set("conv_g_layers", conv_g_layers)
hc.set("conv_d_layers", conv_d_layers)

g_encoder_layers = conv_d_layers
hc.set("g_encode_layers", g_encoder_layers)

hc.set("z_dim", list(np.arange(32,128)))

hc.set("regularize", [True])
hc.set("regularize_lambda", list(np.linspace(0.0001, 1, num=30)))

hc.set("g_batch_norm", [True])
hc.set("d_batch_norm", [True])

hc.set("g_encoder", [True])

hc.set('d_linear_layer', [False, True])
hc.set('d_linear_layers', list(np.arange(50, 600)))

hc.set("g_target_prob", .75 /2.)
hc.set("d_label_smooth", 0.25)

hc.set("d_kernels", list(np.arange(25, 80)))
hc.set("d_kernel_dims", list(np.arange(200, 400)))

hc.set("loss", ['custom'])

hc.set("mse_loss", [False])
hc.set("mse_lambda",list(np.linspace(0.0001, 0.1, num=30)))

hc.set("latent_loss", [True])
hc.set("latent_lambda", list(np.linspace(0.01, .5, num=30)))
hc.set("g_dropout", list(np.linspace(0.6, 0.99, num=30)))

hc.set("g_project", ['zeros'])
hc.set("d_project", ['zeros'])
hc.set("e_project", ['zeros'])

BATCH_SIZE=64
hc.set("batch_size", BATCH_SIZE)
hc.set("model", "mikkel/cifar-adversarial:0.2")
hc.set("version", "0.0.1")
hc.set("machine", "mikkel")


X_DIMS=[32,32]
Y_DIMS=10


def generator(config, y,z, reuse=False):
    with(tf.variable_scope("generator", reuse=reuse)):
        output_shape = X_DIMS[0]*X_DIMS[1]*3
        z_proj_dims = int(config['conv_g_layers'][0])*2
        z_dims = int(z.get_shape()[1])
        print("z_proj_dims", z_proj_dims, z_dims, Y_DIMS)
        noise_dims = z_proj_dims-z_dims-Y_DIMS
        print(noise_dims)
        noise = tf.random_uniform([config['batch_size'], noise_dims],-1, 1)
        if(config['g_project'] == 'noise'):
            result = tf.concat(1, [y, z, noise])
        elif(config['g_project'] == 'zeros'):
            result = tf.concat(1, [y, z])
            #result = z
            result = tf.pad(result, [[0, 0],[noise_dims//2, noise_dims//2]])
        else:
            result = tf.concat(1, [y, z])
            #result = z
            result = linear(result, z_proj_dims, 'g_input_proj')

        def build_layers(result, z_proj_dims, offset):
            if config['conv_g_layers']:
                result = tf.reshape(result, [config['batch_size'], 4,4,z_proj_dims//16])
                #result = tf.nn.dropout(result, 0.7)
                for i, layer in enumerate(config['conv_g_layers']):
                    j=int(result.get_shape()[1]*2)
                    k=int(result.get_shape()[2]*2)
                    stride=2
                    if(j > X_DIMS[0]):
                        j = X_DIMS[0]
                        k = X_DIMS[1]
                        stride=1
                    output = [config['batch_size'], j,k,int(layer)]
                    result = deconv2d(result, output, scope="g_conv_"+str(i+offset), k_w=config['conv_size'], k_h=config['conv_size'], d_h=stride, d_w=stride)
                    if(config['g_batch_norm']):
                        result = batch_norm(result, name='g_conv_bn_'+str(i+offset))
                    if(len(config['conv_g_layers']) == i+1):
                        print("Skipping last layer")
                    else:
                        print("Adding nonlinear")
                        result = config['g_activation'](result)
                return result
        result = build_layers(result, z_proj_dims, 0)

        print("Output shape is ", result.get_shape(), output_shape)
        if(result.get_shape()[1]*result.get_shape()[2]*result.get_shape()[3] != output_shape):
            print("Adding linear layer", result.get_shape(), output_shape)
            result = tf.reshape(result,[config['batch_size'], -1])
            result = tf.nn.dropout(result, config['g_dropout'])
            result = config['g_activation'](result)
            result = linear(result, output_shape, scope="g_proj")
            result = tf.reshape(result, [config["batch_size"], X_DIMS[0], X_DIMS[1], 3])
            result = batch_norm(result, name='g_proj_bn')
            if(config['g_batch_norm']):
                result = batch_norm(result, name='g_proj_bn')
        print("Adding last layer", config['last_layer'])
        if(config['last_layer']):
            result = config['last_layer'](result)
        return result

def discriminator(config, x, y,z,g,gz, reuse=False):
    if(reuse):
        tf.get_variable_scope().reuse_variables()
    batch_size = config['batch_size']*2
    single_batch_size = config['batch_size']
    x = tf.concat(0, [x,g])
    z = tf.concat(0, [z,gz])
    x = tf.reshape(x, [batch_size, -1, 3])
    #x += tf.random_normal(x.get_shape(), mean=0, stddev=0.1)

    if(config['d_project'] == 'zeros'):
        noise_dims = int(x.get_shape()[1])-int(z.get_shape()[1])
        z = tf.pad(z, [[0, 0],[noise_dims//2, noise_dims//2]])
        z = tf.reshape(z, [batch_size, int(x.get_shape()[1]), 1])
        print("CONCAT", x.get_shape(), z.get_shape())
        result = tf.concat(2, [x,z])
    else:
        x = tf.reshape(x, [batch_size, -1])
        result = tf.concat(1, [z,x])
        result = linear(result, X_DIMS[0]*X_DIMS[1]*4, scope='d_z')
        result = config['d_activation'](result)

    if config['conv_d_layers']:
        result = tf.reshape(result, [batch_size, X_DIMS[0],X_DIMS[1],4])
        for i, layer in enumerate(config['conv_d_layers']):
            filter = config['d_conv_size']
            stride = 2
            if(filter > result.get_shape()[1]):
                filter = int(result.get_shape()[1])
                stride = 1
            result = conv2d(result, layer, scope='d_conv'+str(i), k_w=filter, k_h=filter, d_h=stride, d_w=stride)
            if(config['d_batch_norm']):
                result = batch_norm(result, name='d_conv_bn_'+str(i))
            result = config['d_activation'](result)
        result = tf.reshape(x, [batch_size, -1])

    def get_minibatch_features(h):
        n_kernels = int(config['d_kernels'])
        dim_per_kernel = int(config['d_kernel_dims'])
        x = linear(h, n_kernels * dim_per_kernel, scope="d_h")
        activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

        big = np.zeros((batch_size, batch_size), dtype='float32')
        big += np.eye(batch_size)
        big = tf.expand_dims(big, 1)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask
        def half(tens, second):
            m, n, _ = tens.get_shape()
            m = int(m)
            n = int(n)
            return tf.slice(tens, [0, 0, second * single_batch_size], [m, n, single_batch_size])
        # TODO: speedup by allocating the denominator directly instead of constructing it by sum
        #       (current version makes it easier to play with the mask and not need to rederive
        #        the denominator)
        f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
        f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

        return [f1, f2]
    minis = get_minibatch_features(result)
    g_proj = tf.concat(1, [result]+minis)

    #result = tf.nn.dropout(result, 0.7)
    if(config['d_linear_layer']):
        result = linear(result, config['d_linear_layers'], scope="d_linear_layer")
        result = tf.reshape(result, [batch_size, 1, 1, -1])
        if(config['d_batch_norm']):
            result = batch_norm(result, name='d_linear_layer_bn')
        result = tf.reshape(result, [batch_size, -1])
        result = config['d_activation'](result)

    last_layer = result
    result = linear(result, Y_DIMS+1, scope="d_proj")


    def build_logits(class_logits, num_classes):

        generated_class_logits = tf.squeeze(tf.slice(class_logits, [0, num_classes - 1], [batch_size, 1]))
        positive_class_logits = tf.slice(class_logits, [0, 0], [batch_size, num_classes - 1])

        """
        # make these a separate matmul with weights initialized to 0, attached only to generated_class_logits, or things explode
        generated_class_logits = tf.squeeze(generated_class_logits) + tf.squeeze(linear(diff_feat, 1, stddev=0., scope="d_indivi_logits_from_diff_feat"))
        assert len(generated_class_logits.get_shape()) == 1
        # re-assemble the logits after incrementing the generated class logits
        class_logits = tf.concat(1, [positive_class_logits, tf.expand_dims(generated_class_logits, 1)])
        """

        mx = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
        safe_pos_class_logits = positive_class_logits - mx

        gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
        assert len(gan_logits.get_shape()) == 1

        return class_logits, gan_logits
    num_classes = Y_DIMS +1
    class_logits, gan_logits = build_logits(result, num_classes)
    print("Class logits gan logits", class_logits, gan_logits)
    return [tf.slice(class_logits, [0, 0], [single_batch_size, num_classes]),
                tf.slice(gan_logits, [0], [single_batch_size]),
                tf.slice(class_logits, [single_batch_size, 0], [single_batch_size, num_classes]),
                tf.slice(gan_logits, [single_batch_size], [single_batch_size]), 
                last_layer]



def encoder(config, x,y):
    deconv_shape = None
    output_shape = config['z_dim']
    x = tf.reshape(x, [config["batch_size"], -1,3])
    if(config['e_project'] == 'zeros'):
        noise_dims = int(x.get_shape()[1])-int(y.get_shape()[1])
        y = tf.pad(y, [[0, 0],[noise_dims//2, noise_dims//2]])
    else:
        y = linear(y, int(x.get_shape()[1]), scope='g_y')
 
    y = tf.reshape(y, [config['batch_size'], int(x.get_shape()[1]), 1])
    result = tf.concat(2, [x,y])
    result = tf.reshape(result, [config["batch_size"], X_DIMS[0],X_DIMS[1],4])

    if config['g_encode_layers']:
        print('-!-', tf.reshape(result, [config['batch_size'], -1]))
        for i, layer in enumerate(config['g_encode_layers']):
            print(layer)
            filter = config['conv_size']
            stride = 2
            if filter > result.get_shape()[2]:
                filter = int(result.get_shape()[2])
                stride = 1
            result = conv2d(result, layer, scope='g_enc_conv'+str(i), k_w=filter, k_h=filter, d_h=stride, d_w=stride)
            if(config['d_batch_norm']):
                result = batch_norm(result, name='g_enc_conv_bn_'+str(i))
            if(len(config['g_encode_layers']) == i+1):
                print("Skipping last layer")
            else:
                print("Adding nonlinear")
                result = config['g_activation'](result)
            print(tf.reshape(result, [config['batch_size'], -1]))
        result = tf.reshape(result, [config["batch_size"], -1])

    if(result.get_shape()[1] != output_shape):
        print("(e)Adding linear layer", result.get_shape(), output_shape)
        result = config['g_activation'](result)
        result = linear(result, output_shape, scope="g_enc_proj")
        result = tf.reshape(result, [config['batch_size'], 1, 1, -1])
        if(config['g_batch_norm']):
            result = batch_norm(result, name='g_enc_proj_bn')
        result = tf.reshape(result, [config['batch_size'], -1])

    if(config['last_layer']):
        result = config['last_layer'](result)
    return result

def approximate_z(config, x, y):
    transfer_fct = config['transfer_fct']
    n_input = config['n_input']
    n_hidden_recog_1 = int(config['n_hidden_recog_1'])
    n_hidden_recog_2 = int(config['n_hidden_recog_2'])
    n_z = int(config['z_dim'])
    print('nz', n_z, type(n_z))
    weights = {
            'h1': tf.get_variable('g_h1', [n_input+Y_DIMS, n_hidden_recog_1], initializer=tf.contrib.layers.xavier_initializer()),
            'h2': tf.get_variable('g_h2', [n_hidden_recog_1, n_hidden_recog_2], initializer=tf.contrib.layers.xavier_initializer()),
            'out_mean': tf.get_variable('g_out_mean', [n_hidden_recog_2, n_z], initializer=tf.contrib.layers.xavier_initializer()),
            'out_log_sigma': tf.get_variable('g_out_log_sigma', [n_hidden_recog_2, n_z], initializer=tf.contrib.layers.xavier_initializer()),
            'b1': tf.get_variable('g_b1', initializer=tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.get_variable('g_b2', initializer=tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b_out_mean': tf.get_variable('g_b_out_mean', initializer=tf.zeros([n_z], dtype=tf.float32)),
            'b_out_log_sigma': tf.get_variable('g_b_out_log_sigma', initializer=tf.zeros([n_z], dtype=tf.float32))
            }
    x = tf.reshape(x, [config['batch_size'], n_input])
    y = tf.reshape(y, [config['batch_size'], Y_DIMS])
    x = tf.concat(1,[x,y])
    layer_1 = transfer_fct(tf.add(tf.matmul(x, weights['h1']), 
                                       weights['b1'])) 
    layer_2 = transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                       weights['b2'])) 
    mu = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        weights['b_out_mean'])
    sigma = \
        tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
               weights['b_out_log_sigma'])


    n_z = int(config["z_dim"])
    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=tf.float32)

    return tf.add(mu, tf.mul(sigma, eps)), mu, sigma


def sigmoid_kl_with_logits(logits, targets):
    print(targets)
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy


def create(config, x,y):
    batch_size = config["batch_size"]
    print(y)

    #x = x/tf.reduce_max(tf.abs(x), 0)
    encoded_z = encoder(config, x,y)
    z, z_mu, z_sigma = approximate_z(config, x, y)


    print("Build generator")
    g = generator(config, y, z)
    print("Build encoder")
    encoded = generator(config, y, encoded_z, reuse=True)
    print("shape of g,x", g.get_shape(), x.get_shape())
    print("shape of z,encoded_z", z.get_shape(), encoded_z.get_shape())
    d_real, d_real_sig, d_fake, d_fake_sig, d_last_layer = discriminator(config,x, y, encoded_z, g, z, reuse=False)

    latent_loss = -config['latent_lambda'] * tf.reduce_mean(1 + z_sigma
                                       - tf.square(z_mu)
                                       - tf.exp(z_sigma), 1)
    fake_symbol = tf.tile(tf.constant([0,0,0,0,0,0,0,0,0,0,1], dtype=tf.float32), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],11])

    real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])
    #real_symbols = y


    if(config['loss'] == 'softmax'):
        d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, fake_symbol)
        d_real_loss = tf.nn.softmax_cross_entropy_with_logits(d_real, real_symbols)

        g_loss= tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols)

    else:
        zeros = tf.zeros_like(d_fake_sig)
        ones = tf.zeros_like(d_real_sig)

        #d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake, zeros)
        #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real, ones)

        generator_target_prob = config['g_target_prob']
        d_label_smooth = config['d_label_smooth']
        d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake_sig, zeros)
        #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real_sig, ones)
        d_real_loss = sigmoid_kl_with_logits(d_real_sig, 1.-d_label_smooth)
        d_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_real,real_symbols)
        d_fake_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake,fake_symbol)

        g_loss= sigmoid_kl_with_logits(d_fake_sig, generator_target_prob)
        g_class_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols)

        #g_loss_encoder = tf.nn.sigmoid_cross_entropy_with_logits(d_real, zeros)
        #TINY = 1e-12
        #d_real = tf.nn.sigmoid(d_real)
        #d_fake = tf.nn.sigmoid(d_fake)
        #d_fake_loss = -tf.log(1-d_fake+TINY)
        #d_real_loss = -tf.log(d_real+TINY)
        #g_loss_softmax = -tf.log(1-d_real+TINY)
        #g_loss_encoder = -tf.log(d_fake+TINY)
    if(config['latent_loss']):
        g_loss = tf.reduce_mean(g_loss)+tf.reduce_mean(latent_loss)+tf.reduce_mean(g_class_loss)
    else:
        g_loss = tf.reduce_mean(g_loss)+tf.reduce_mean(g_class_loss)
    d_loss = tf.reduce_mean(d_fake_loss) + tf.reduce_mean(d_real_loss) + \
            tf.reduce_mean(d_class_loss)+tf.reduce_mean(d_fake_class_loss)
    print('d_loss', d_loss.get_shape())



    if config['regularize']:
        ws = None
        with tf.variable_scope("generator"):
            with tf.variable_scope("g_conv_0"):
                tf.get_variable_scope().reuse_variables()
                ws = tf.get_variable('w')
                tf.get_variable_scope().reuse_variables()
                b = tf.get_variable('biases')
            lam = config['regularize_lambda']
            g_loss += lam*tf.nn.l2_loss(ws)+lam*tf.nn.l2_loss(b)


    mse_loss = tf.reduce_max(tf.square(x-encoded))
    if config['mse_loss']:
        mse_lam = config['mse_lambda']
        g_loss += mse_lam * mse_loss

    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    print(config);
    print('vars', [v.name for v in tf.trainable_variables()])
    g_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate'])).minimize(g_loss, var_list=g_vars)
    d_optimizer = tf.train.AdamOptimizer(np.float32(config['d_learning_rate'])).minimize(d_loss, var_list=d_vars)

    mse_optimizer = tf.train.AdamOptimizer(np.float32(config['g_learning_rate'])).minimize(mse_loss, var_list=tf.trainable_variables())

    set_tensor("x", x)
    set_tensor("y", y)
    set_tensor("g_loss", g_loss)
    set_tensor("d_loss", d_loss)
    set_tensor("g_optimizer", g_optimizer)
    set_tensor("d_optimizer", d_optimizer)
    set_tensor("mse_optimizer", mse_optimizer)
    set_tensor("g", g)
    set_tensor("encoded", encoded)
    set_tensor("encoder_mse", mse_loss)
    set_tensor("d_fake", tf.reduce_mean(d_fake))
    set_tensor("d_real", tf.reduce_mean(d_real))
    set_tensor("d_fake_loss", tf.reduce_mean(d_fake_loss))
    set_tensor("d_real_loss", tf.reduce_mean(d_real_loss))
    set_tensor("d_class_loss", tf.reduce_mean(d_real_loss))
    set_tensor("g_class_loss", tf.reduce_mean(g_class_loss))
    set_tensor("d_loss", tf.reduce_mean(d_real_loss))

def train(sess, config):
    x = get_tensor('x')
    g = get_tensor('g')
    g_loss = get_tensor("g_loss")
    d_loss = get_tensor("d_loss")
    d_fake_loss = get_tensor('d_fake_loss')
    d_real_loss = get_tensor('d_real_loss')
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")
    d_class_loss = get_tensor("d_class_loss")
    g_class_loss = get_tensor("g_class_loss")
    mse_optimizer = get_tensor("mse_optimizer")
    encoder_mse = get_tensor("encoder_mse")
    _, d_cost = sess.run([d_optimizer, d_loss])
    _, g_cost, x, g,e_loss,d_fake,d_real, d_class, g_class = sess.run([g_optimizer, g_loss, x, g, encoder_mse,d_fake_loss, d_real_loss, d_class_loss, g_class_loss])
    #_ = sess.run([mse_optimizer])

    print("g cost %.2f d cost %.2f encoder %.2f d_fake %.6f d_real %.2f d_class %.2f g_class %.2f" % (g_cost, d_cost,e_loss, d_fake, d_real, d_class, g_class))
    print("X mean %.2f max %.2f min %.2f" % (np.mean(x), np.max(x), np.min(x)))
    print("G mean %.2f max %.2f min %.2f" % (np.mean(g), np.max(g), np.min(g)))

    return d_cost, g_cost

def test(sess, config):
    x = get_tensor("x")
    y = get_tensor("y")
    d_fake = get_tensor("d_fake")
    d_real = get_tensor("d_real")
    g_loss = get_tensor("g_loss")
    encoder_mse = get_tensor("encoder_mse")

    g_cost, d_fake_cost, d_real_cost, e_cost = sess.run([g_loss, d_fake, d_real, encoder_mse])


    #hc.event(costs, sample_image = sample[0])

    #print("test g_loss %.2f d_fake %.2f d_loss %.2f" % (g_cost, d_fake_cost, d_real_cost))
    return g_cost,d_fake_cost, d_real_cost, e_cost

def sample_input(sess, config):
    x = get_tensor("x")
    encoded = get_tensor('encoded')
    sample, encoded = sess.run([x, encoded])
    return sample[0], encoded[0]


def split_sample(n, sample):
    return [np.reshape(sample[0+i:1+i], [X_DIMS[0],X_DIMS[1], 3]) for i in range(n)]
def samples(sess, config):
    generator = get_tensor("g")
    y = get_tensor("y")
    x = get_tensor("x")
    rand = np.random.randint(0,10, size=config['batch_size'])
    random_one_hot = np.eye(10)[rand]
    #x_input = np.random.normal(0, 1, [config['batch_size'], X_DIMS[0],X_DIMS[1],3])
    sample = sess.run(generator, feed_dict={y:random_one_hot})
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
        d_loss, g_loss = train(sess, config)
        if(i > 30 and not args.no_stop):
        
            if(math.isnan(d_loss) or math.isnan(g_loss) or g_loss < -10 or g_loss > 1000 or d_loss > 1000):
                return False
        
            g = get_tensor('g')
            rX = sess.run([g])
            if(np.min(rX) < -1000 or np.max(rX) > 1000):
                return False


    return True

def test_config(sess, config):
    batch_size = config["batch_size"]
    n_samples =  cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    total_batch = int(n_samples / batch_size)
    results = []
    for i in range(total_batch):
        results.append(test(sess, config))
    return results

def test_epoch(epoch, j, sess, config):
    x, encoded = sample_input(sess, config)
    sample_file = "samples/input-"+str(j)+".png"
    cifar_utils.plot(config, x, sample_file)
    encoded_sample = "samples/encoded-"+str(j)+".png"
    cifar_utils.plot(config, encoded, encoded_sample)
    
    sample_file = {'image':sample_file, 'label':'input'}
    encoded_sample = {'image':encoded_sample, 'label':'reconstructed'}
    sample = samples(sess, config)
    sample_list = [sample_file, encoded_sample]
    for s in sample:
        sample_file = "samples/config-"+str(j)+".png"
        cifar_utils.plot(config, s, sample_file)
        sample_list.append({'image':sample_file,'label':'sample-'+str(j)})
        j+=1
    print("Creating sample")
    hc.io.sample(config, sample_list)
    return j

def record_run(config):
    results = test_config(sess, config)
    loss = np.array(results)
    #results = np.reshape(results, [results.shape[1], results.shape[0]])
    g_loss = [g for g,_,_,_ in loss]
    g_loss = np.mean(g_loss)
    d_fake = [d_ for _,d_,_,_ in loss]
    d_fake = np.mean(d_fake)
    d_real = [d for _,_,d,_ in loss]
    d_real = np.mean(d_real)
    e_loss = [e for _,_,_,e in loss]
    e_loss = np.mean(e_loss)

    # calculate D.difficulty = reduce_mean(d_loss_fake) - reduce_mean(d_loss_real)
    difficulty = d_real * (1-d_fake)
    # calculate G.ranking = reduce_mean(g_loss) * D.difficulty
    ranking = g_loss * (1.0-difficulty)

    ranking = e_loss
    results =  {
        'difficulty':float(difficulty),
        'ranking':float(ranking),
        'g_loss':float(g_loss),
        'd_fake':float(d_fake),
        'd_real':float(d_real),
        'e_loss':float(e_loss)
    }
    print("Recording ", results)
    hc.io.record(config, results)





print("Generating configs with hyper search space of ", hc.count_configs())

j=0
k=0
cifar_utils.maybe_download_and_extract()

def get_function(name):
    if not isinstance(name, str):
        return name
    print('name', name);
    if(name == "function:tensorflow.python.ops.gen_nn_ops.relu"):
        return tf.nn.relu
    if(name == "function:tensorflow.python.ops.nn_ops.relu"):
        return tf.nn.relu
    if(name == "function:tensorflow.python.ops.gen_nn_ops.relu6"):
        return tf.nn.relu6
    if(name == "function:tensorflow.python.ops.nn_ops.relu6"):
        return tf.nn.relu6
    if(name == "function:tensorflow.python.ops.gen_nn_ops.elu"):
        return tf.nn.elu
    if(name == "function:tensorflow.python.ops.nn_ops.elu"):
        return tf.nn.elu
    return eval(name.split(":")[1])
for config in hc.configs(1):
    if(args.load_config):
        print("Loading config", args.load_config)
        config.update(hc.io.load_config(args.load_config))
        if(not config):
            print("Could not find config", args.load_config)
            break
    if(args.d_batch_norm):
        config['d_batch_norm']=True
    config['g_activation']=get_function(config['g_activation'])
    config['d_activation']=get_function(config['d_activation'])
    config['transfer_fct']=get_function(config['transfer_fct'])
    config['last_layer']=get_function(config['last_layer'])
    print(config)
    print("Testing configuration", config)
    print("TODO: TEST BROKEN")
    sess = tf.Session()
    train_x,train_y = cifar_utils.inputs(eval_data=False,data_dir="/tmp/cifar/cifar-10-batches-bin",batch_size=BATCH_SIZE)
    test_x,test_y = cifar_utils.inputs(eval_data=True,data_dir="/tmp/cifar/cifar-10-batches-bin",batch_size=BATCH_SIZE)
    x = train_x
    y = train_y
    y=tf.one_hot(tf.cast(train_y,tf.int64), Y_DIMS, 1.0, 0.0)
    #x = tf.get_variable('x', [BATCH_SIZE, X_DIMS[0], X_DIMS[1], 3], tf.float32)
    #y = tf.get_variable('y', [BATCH_SIZE, Y_DIMS], tf.float32)
    graph = create(config,x,y)
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)



    #tf.assign(x,train_x)
    #tf.assign(y,tf.one_hot(tf.cast(train_y,tf.int64), Y_DIMS, 1.0, 0.0))
    sampled=False
    print("Running for ", args.epochs, " epochs")
    for i in range(args.epochs):
        if(not epoch(sess, config)):
            break
        j=test_epoch(i, j, sess, config)
        if(i == args.epochs-1):
            print("Recording run...")
            record_run(config)
    #x.assign(test_x)
    #y.assign(tf.one_hot(tf.cast(test_y,tf.int64), Y_DIMS, 1.0, 0.0))
    #print("results: difficulty %.2f, ranking %.2f, g_loss %.2f, d_fake %.2f, d_real %.2f" % (difficulty, ranking, g_loss, d_fake, d_real))

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

