import hyperchamber as hc
from shared.ops import *

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

start=.0001
end=.0005
num=20
hc.permute.set("g_learning_rate", list(np.linspace(start, end, num=num)))
hc.permute.set("d_learning_rate", list(np.linspace(start, end, num=num)))

hc.permute.set("n_hidden_recog_1", list(np.linspace(100, 1000, num=10)))
hc.permute.set("n_hidden_recog_2", list(np.linspace(100, 1000, num=10)))
hc.permute.set("transfer_fct", [tf.tanh, tf.nn.elu, tf.nn.relu, tf.nn.relu6, tf.nn.softplus, tf.nn.softsign]);

hc.set("n_input", 32*32*3)

conv_g_layers = [[(i+20)*4, (i+20)*2, 3] for i in range(60)]

conv_d_layers = [[(i+15), (i+15)*2, (i+15)*4] for i in range(30)]
#conv_d_layers = [[32, 32*2, 32*4],[32, 64, 64*2],[64,64*2], [16,16*2, 16*4], [16,16*2]]
g_encoder_layers = [[(i+15), (i+15)*2, (i+15)*4] for i in range(30)]

hc.permute.set("conv_g_layers", conv_g_layers)
hc.permute.set("conv_d_layers", conv_d_layers)
hc.permute.set("g_encode_layers", g_encoder_layers)

hc.permute.set("z_dim", [32*4, 64*4, 16*4, 64, 32])
hc.permute.set("z_lin_layer", [32*8, 64*8, 16*8, 64])

hc.permute.set("regularize", [False, True])
hc.permute.set("regularize_lambda", list(np.linspace(0.0001, 1, num=30)))

hc.permute.set("g_batch_norm", [True])
hc.permute.set("d_batch_norm", [True])

hc.permute.set("g_last_layer", [None])

hc.permute.set("g_encoder", [True])

hc.permute.set("loss", ['sigmoid', 'softmax'])

hc.permute.set("g_lrelu_leak", np.linspace(0.6,0.9, num=5))

hc.permute.set("mse_loss", [False])
hc.permute.set("mse_lambda",list(np.linspace(0.0001, 1, num=30)))

hc.permute.set("latent_lambda", list(np.linspace(0.0001, .2, num=30)))

BATCH_SIZE=64
hc.set("batch_size", BATCH_SIZE)
hc.set("model", "255bits/cifar-gan-nomse-epoch10")
hc.set("version", "0.0.1")


X_DIMS=[32,32]
Y_DIMS=10


def generator(config, y,z, reuse=False):
    if(reuse):
      tf.get_variable_scope().reuse_variables()
    output_shape = X_DIMS[0]*X_DIMS[1]*3
    result = tf.concat(1, [y, z])
    result = linear(result, config['z_lin_layer'], 'g_input_proj')

    if config['conv_g_layers']:
        result = tf.reshape(result, [config['batch_size'], 4,4,config['z_lin_layer']//16])
        #result = tf.nn.dropout(result, 0.7)
        for i, layer in enumerate(config['conv_g_layers']):
            j=int(result.get_shape()[1]*2)
            k=int(result.get_shape()[2]*2)
            output = [config['batch_size'], j,k,layer]
            result = deconv2d(result, output, scope="g_conv_"+str(i))
            if(config['g_batch_norm']):
                result = batch_norm(result, name='g_conv_bn_'+str(i))
            if(len(config['conv_g_layers']) == i+1):
                print("Skipping last layer")
            else:
                print("Adding nonlinear")
                result = lrelu(result, leak=config['g_lrelu_leak'])

    print("Output shape is ", result.get_shape(), output_shape)
    if(result.get_shape()[1]*result.get_shape()[2]*result.get_shape()[3] != output_shape):
        print("Adding linear layer")
        result = tf.reshape(result,[config['batch_size'], -1])
        result = linear(result, output_shape, scope="g_proj")
        result = tf.reshape(result, [config["batch_size"], X_DIMS[0], X_DIMS[1], 3])
        if(config['g_batch_norm']):
            result = batch_norm(result, name='g_proj_bn')
    print("Adding last layer", config['g_last_layer'])
    if(config['g_last_layer'] == None):
        pass
    elif(config['g_last_layer'] == "lrelu"):
        result = lrelu(result, config['g_lrelu_leak'])
    return result

def discriminator(config, x, z, reuse=False):
    if(reuse):
      tf.get_variable_scope().reuse_variables()
    x = tf.reshape(x, [config["batch_size"], 3, -1])
    z = linear(z, int(x.get_shape()[2]), scope='d_z')
    z = tf.reshape(z, [config['batch_size'], 1, int(x.get_shape()[2])])
    print("CONCAT", x.get_shape(), z.get_shape())
    result = tf.concat(1, [x,z])
    if config['conv_d_layers']:
        result = tf.reshape(result, [config["batch_size"], X_DIMS[0],X_DIMS[1],4])
        for i, layer in enumerate(config['conv_d_layers']):
            result = conv2d(result, layer, scope='d_conv'+str(i))
            if(config['d_batch_norm']):
                result = batch_norm(result, name='d_conv_bn_'+str(i))
            result = tf.nn.relu(result)
        result = tf.reshape(x, [config["batch_size"], -1])

    #result = tf.nn.dropout(result, 0.7)
    last_layer = result
    if(config['loss'] == 'softmax'):
        result = linear(result, 11, scope="d_proj")
    else:
        result = linear(result, 1, scope="d_proj")

    return result, last_layer


def encoder(config, x,y):
    deconv_shape = None
    output_shape = config['z_dim']
    x = tf.reshape(x, [config["batch_size"], 3, -1])
    y = linear(y, int(x.get_shape()[2]), scope='g_y')
    y = tf.reshape(y, [config['batch_size'], 1, int(x.get_shape()[2])])
    result = tf.concat(1, [x,y])
    result = tf.reshape(result, [config["batch_size"], X_DIMS[0],X_DIMS[1],4])

    if config['g_encode_layers']:
        for i, layer in enumerate(config['g_encode_layers']):
            result = conv2d(result, layer, scope='g_enc_conv'+str(i))
            if(config['d_batch_norm']):
                result = batch_norm(result, name='g_enc_conv_bn_'+str(i))
            if(len(config['g_encode_layers']) == i+1):
                print("Skipping last layer")
            else:
                print("Adding nonlinear")
                result = lrelu(result, leak=config['g_lrelu_leak'])
        result = tf.reshape(x, [config["batch_size"], -1])

    if(result.get_shape()[1] != output_shape):
        print("Adding linear layer")
        result = lrelu(result, leak=config['g_lrelu_leak'])
        result = linear(result, output_shape, scope="g_enc_proj")
        result = tf.reshape(result, [config['batch_size'], 1, 1, -1])
        if(config['g_batch_norm']):
            result = batch_norm(result, name='g_enc_proj_bn')
        result = tf.reshape(result, [config['batch_size'], -1])

    return result

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def approximate_z(config, x, y):
    transfer_fct = config['transfer_fct']
    n_input = config['n_input']
    n_hidden_recog_1 = int(config['n_hidden_recog_1'])
    n_hidden_recog_2 = int(config['n_hidden_recog_2'])
    n_z = config['z_dim']
    weights = {
            'h1': tf.get_variable('g_h1', initializer=xavier_init(n_input+Y_DIMS, n_hidden_recog_1)),
            'h2': tf.get_variable('g_h2', initializer=xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.get_variable('g_out_mean', initializer=xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.get_variable('g_out_log_sigma', initializer=xavier_init(n_hidden_recog_2, n_z)),
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


    n_z = config["z_dim"]
    eps = tf.random_normal((config['batch_size'], n_z), 0, 1, 
                           dtype=tf.float32)

    return tf.add(mu, tf.mul(sigma, eps)), mu, sigma

def create(config, x,y):
    batch_size = config["batch_size"]
    print(y)

    #x = x/tf.reduce_max(tf.abs(x), 0)
    encoded_z = encoder(config, x,y)
    d_real, d_last_layer = discriminator(config,x, encoded_z)
    z, z_mu, z_sigma = approximate_z(config, x, y)


    print("Build generator")
    g = generator(config, y, z)
    print("Build encoder")
    encoded = generator(config, y, encoded_z, reuse=True)
    print("shape of g,x", g.get_shape(), x.get_shape())
    print("shape of z,encoded_z", z.get_shape(), encoded_z.get_shape())
    d_fake, _ = discriminator(config,g, z, reuse=True)

    if(config['loss'] == 'softmax'):
        fake_symbol = tf.tile(tf.constant([0,0,0,0,0,0,0,0,0,0,1], dtype=tf.float32), [config['batch_size']])
        fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],11])

        real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])


        d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, fake_symbol)
        d_real_loss = tf.nn.softmax_cross_entropy_with_logits(d_real, real_symbols)

        g_loss_softmax = tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols)
        g_loss_encoder = tf.nn.softmax_cross_entropy_with_logits(d_real, fake_symbol)

        latent_loss = -config['latent_lambda'] * tf.reduce_mean(1 + z_sigma
                                           - z_mu
                                           - tf.exp(z_sigma), 1)

        g_loss = tf.reduce_mean(g_loss_softmax+g_loss_encoder+latent_loss)
        d_loss = tf.reduce_mean(d_fake_loss + d_real_loss)
    else:
        fake_symbol = 0

        #zeros = tf.zeros_like(d_fake)
        #ones = tf.zeros_like(d_real)

        #d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_fake, zeros)
        #d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_real, ones)

        #g_loss_softmax = tf.nn.sigmoid_cross_entropy_with_logits(d_fake, ones)
        #g_loss_encoder = tf.nn.sigmoid_cross_entropy_with_logits(d_real, zeros)
        d_real = tf.nn.sigmoid(d_real)
        d_fake = tf.nn.sigmoid(d_fake)
        d_fake_loss = -tf.log(d_real)
        d_real_loss =  -tf.log(1-d_fake)
        g_loss_softmax = -tf.log(1-d_real)
        g_loss_encoder = -tf.log(d_fake)
        g_loss = tf.reduce_mean(g_loss_softmax+g_loss_encoder)
        d_loss = tf.reduce_mean(d_fake_loss + d_real_loss)



    if config['regularize']:
        ws = None
        with tf.variable_scope("g_input_proj"):
            tf.get_variable_scope().reuse_variables()
            ws = tf.get_variable('Matrix')
            tf.get_variable_scope().reuse_variables()
            b = tf.get_variable('bias')
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

def train(sess, config):
    x = get_tensor('x')
    g = get_tensor('g')
    g_loss = get_tensor("g_loss")
    d_loss = get_tensor("d_loss")
    g_optimizer = get_tensor("g_optimizer")
    d_optimizer = get_tensor("d_optimizer")
    mse_optimizer = get_tensor("mse_optimizer")
    encoder_mse = get_tensor("encoder_mse")
    _, d_cost = sess.run([d_optimizer, d_loss])
    _, g_cost, x, g,e_loss = sess.run([g_optimizer, g_loss, x, g, encoder_mse])
    #_ = sess.run([mse_optimizer])

    print("g cost %.2f d cost %.2f encoder %.2f" % (g_cost, d_cost,e_loss))
    #print(" mean %.2f max %.2f min %.2f" % (np.mean(x), np.max(x), np.min(x)))
    #print(" mean %.2f max %.2f min %.2f" % (np.mean(g), np.max(g), np.min(g)))

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
    x_input = np.random.uniform(0, 1, [config['batch_size'], X_DIMS[0],X_DIMS[1],3])
    sample = sess.run(generator, feed_dict={x:x_input, y:random_one_hot})
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
        if(math.isnan(d_loss) or math.isnan(g_loss)):
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
for config in hc.configs(100):
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
    for i in range(1000):
        if(not epoch(sess, config)):
            break
        j=test_epoch(i, j, sess, config)
        sampled=True
    #x.assign(test_x)
    #y.assign(tf.one_hot(tf.cast(test_y,tf.int64), Y_DIMS, 1.0, 0.0))
    #print("results: difficulty %.2f, ranking %.2f, g_loss %.2f, d_fake %.2f, d_real %.2f" % (difficulty, ranking, g_loss, d_fake, d_real))

    print("Recording run...")
    if(sampled):
        record_run(config)
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

