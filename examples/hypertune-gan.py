import hyperchamber as hc
from shared.ops import *

import os
import time
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

GROUP=6
SINGLE_OPTIMIZER=True
g_layers = [ None for i in range(GROUP) ]
d_layers = [ None for i in range(GROUP) ]

start=.001
end=.005
hc.set("g_learning_rate", 3*start)
hc.set("d_learning_rate", start)#list(np.linspace(start, end, num=len(d_layers))))

conv_g_layers = list(np.linspace(1, len(d_layers), num=len(d_layers)))
conv_g_layers = [[28, 1] for x in conv_g_layers]

conv_d_layers = list(np.linspace(0, GROUP, num=GROUP))
conv_d_layers = [[28, 4] for x in conv_d_layers]

hc.set("conv_g_layers", conv_g_layers)
hc.set("conv_d_layers", conv_d_layers)

hc.set("g_layers", g_layers)
hc.set("d_layers", d_layers)
hc.set("dropout", list(np.linspace(0.5, 1.0,num=len(d_layers))))
hc.set("z_dim", 49)
hint_layers = np.linspace(1,100,num=len(conv_d_layers))
hc.set("hint_layers", [[28] for x in list(hint_layers)])

hc.set("batch_size", 64)

X_DIMS=[28,28]
Y_DIMS=10

EPOCHS=200

def generator(config, y, teach):
    output_shape = X_DIMS[0]*X_DIMS[1]
    z = tf.random_uniform([config["batch_size"], 10],0,1)
    result = tf.concat(1, [y, z, teach])
    result = linear(result, config['z_dim'], 'g_input_proj')
    result = tf.nn.dropout(result, config['dropout'])

    if config['conv_g_layers']:
        result = tf.reshape(result, [config['batch_size'], 7,7,1])
        for i, layer in enumerate(config['conv_g_layers']):
            if layer > 0:
                j=int(result.get_shape()[1]*2)
                k=int(result.get_shape()[2]*2)
                output = [config['batch_size'], j,k,int(layer)]
                result = deconv2d(result, output, scope="g_conv_"+str(i))
                result = tf.nn.sigmoid(result)
        result = tf.reshape(result,[config['batch_size'], -1])
    else:
        for i, layer in enumerate(config['g_layers']):
            result = linear(result, layer, scope="g_linear_"+str(i))
            result = tf.nn.sigmoid(result)
    if(result.get_shape()[1] != output_shape):
        result = linear(result, output_shape, scope="g_proj")
    result = tf.reshape(result, [config["batch_size"], X_DIMS[0], X_DIMS[1]])
    return result

def discriminator(config, x, reuse=False):
    if(reuse):
      tf.get_variable_scope().reuse_variables()
    result = tf.nn.dropout(x, config['dropout'])
    if config['conv_d_layers']:
        result = tf.reshape(result, [config["batch_size"], X_DIMS[0],X_DIMS[1],1])
        for i, layer in enumerate(config['conv_d_layers']):
            if(layer > 0):
                result = conv2d(result, int(layer), scope='d_conv'+str(i))
                result = tf.nn.relu(result)
        result = tf.reshape(x, [config["batch_size"], -1])
    else:
        result = tf.reshape(x, [config["batch_size"], -1])
        for i, layer in enumerate(config['d_layers']):
            result = linear(result, layer, scope="d_linear_"+str(i))
            result = tf.nn.relu(result)

    last_layer = result
    result = linear(result, 11, scope="d_proj")
    return result, last_layer

def teacher(config, result):
  for i, layer in enumerate(config['hint_layers']):
      result = linear(result, layer, scope="g_hint_linear_"+str(i))
      result = tf.nn.relu(result)

  return result


def create(config,x,y):
    batch_size = config["batch_size"]
    d_real, d_last_layer = discriminator(config,x)
    teach = teacher(config, d_last_layer)
    g = generator(config, y, teach)
    d_fake, _ = discriminator(config,g, reuse=True)

    fake_symbol = tf.tile(tf.constant([0,0,0,0,0,0,0,0,0,0,1], dtype=tf.float32), [config['batch_size']])
    fake_symbol = tf.reshape(fake_symbol, [config['batch_size'],11])

    real_symbols = tf.concat(1, [y, tf.zeros([config['batch_size'], 1])])


    d_fake_loss = tf.nn.softmax_cross_entropy_with_logits(d_fake, fake_symbol)
    d_real_loss = tf.nn.softmax_cross_entropy_with_logits(d_real, real_symbols)

    g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(d_fake, real_symbols))
    d_loss = tf.reduce_mean(0.5*d_fake_loss + 0.5*d_real_loss)

    set_tensors("g_loss", g_loss)
    set_tensors("d_loss", d_loss)
    set_tensors("d_fake_loss", tf.reduce_mean(tf.nn.softmax(d_fake) * (1-fake_symbol)))
    set_tensors("d_real_loss", tf.reduce_mean(tf.nn.softmax(d_real) * (real_symbols)))
    set_tensors("g", g)
    set_tensors("d_fake", tf.reduce_mean(d_fake))
    set_tensors("d_real", tf.reduce_mean(d_real))


def create_optimizers(configs):
    #TODO fix: multiple learning rates broken
    d_loss = get_tensors("d_loss")
    g_loss = get_tensors("g_loss")

    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

    if(SINGLE_OPTIMIZER):
        config = configs[0]
        d_loss = tf.add_n(d_loss)
        g_loss = tf.add_n(g_loss)
        print('g_learning_rate', config['g_learning_rate'])

        g_optimizer = [tf.train.AdamOptimizer(config['g_learning_rate']).minimize(g_loss, var_list=g_vars)]
        d_optimizer = [tf.train.AdamOptimizer(config['d_learning_rate']).minimize(d_loss, var_list=d_vars)]

        set_tensors("g_optimizer", g_optimizer)
        set_tensors("d_optimizer", d_optimizer)
    else:
        def build_optimizer(lr):
            return tf.train.AdamOptimizer(lr)

        set_tensors("g_optimizer", [build_optimizer(config['g_learning_rate']).minimize(g_l, var_list=g_vars) for config,g_l in zip(configs, g_loss)])
        set_tensors("d_optimizer", [build_optimizer(config['d_learning_rate']).minimize(d_l, var_list=d_vars) for config,d_l in zip(configs, d_loss)])

hack_count=0
def create_graph(config):
    global hack_count
    batch_size = config["batch_size"]
    x = shared_placeholder(tf.float32, [batch_size, X_DIMS[0], X_DIMS[1], 1], name="x")
    y = shared_placeholder(tf.float32, [batch_size, Y_DIMS], name="y")
    uuid = str(hack_count)
    hack_count+=1

    with tf.variable_scope(uuid):
        return create(config,x,y)

def parallel_run(sess, steps, feed_dict):
    length = len(steps)
    flat = np.hstack(steps)

    results = sess.run(list(flat), feed_dict=feed_dict)
    return np.hsplit(np.array(results), length)

trainCount=0
def train(sess, config, x_input, y_input):
    global trainCount
    x = get_tensor("x")
    y = get_tensor("y")
    g_loss = get_tensors("g_loss")
    d_loss = get_tensors("d_loss")
    d_real_loss = get_tensors("d_real_loss")
    d_fake_loss = get_tensors("d_fake_loss")
    g_optimizer = get_tensors("g_optimizer")
    d_optimizer = get_tensors("d_optimizer")


    _ = parallel_run(sess, d_optimizer, feed_dict={x:x_input, y:y_input})
    _ = parallel_run(sess, g_optimizer, feed_dict={x:x_input, y:y_input})

    print("Optimization step complete "+str(trainCount))
    trainCount+=1
    #for gc,dc,dr,df in zip(g_cost, d_cost, d_real_cost, d_fake_cost):
    #    print("g cost %.2f d cost %.2f real %.2f fake %.2f" % (gc, dc, dr, df))

def test(sess, configs, x_input, y_input):
    x = get_tensor("x")
    y = get_tensor("y")
    d_fake = get_tensors("d_fake")
    d_real = get_tensors("d_real")
    g_loss = get_tensors("g_loss")
    d_loss = get_tensors('d_loss')

    g_cost, d_cost, d_fake_cost, d_real_cost = parallel_run(sess, [g_loss, d_loss, d_fake, d_real], feed_dict={x:x_input, y:y_input})


    #hc.event(costs, sample_image = sample[0])

    #print("test g_loss %.2f d_fake %.2f d_loss %.2f" % (g_cost, d_fake_cost, d_real_cost))
    return g_cost,d_cost, d_fake_cost, d_real_cost


def sample(sess, configs):
    config = configs[0]
    generator = get_tensors("g")
    y = get_tensor("y")
    x = get_tensor("x")
    rand = np.random.randint(0,10, size=config['batch_size'])
    random_one_hot = np.eye(10)[rand]
    x_input = np.random.uniform(0, 1, [config['batch_size'], X_DIMS[0],X_DIMS[1],1])
    samples = parallel_run(sess, [generator], feed_dict={x:x_input, y:random_one_hot})
    #sample =  np.concatenate(sample, axis=0)
    print([np.shape(sample[0:4]) for sample in samples[0]], X_DIMS)
    return [np.reshape(sample[0:4], [X_DIMS[0]*4,X_DIMS[1]]) for sample in samples[0]]

def plot_mnist_digit(image, file):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    #plt.suptitle(config)
    plt.savefig(file)

def epoch(sess, configs, mnist):
    batch_size = configs[0]["batch_size"]
    n_samples = mnist.train.num_examples
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        x, y = mnist.train.next_batch(batch_size)
        x=np.reshape(x, [batch_size, X_DIMS[0], X_DIMS[1], 1])
        train(sess, configs, x, y)

def test_configs(sess, configs):
    batch_size = configs[0]["batch_size"]
    n_samples = mnist.test.num_examples
    total_batch = int(n_samples / batch_size)
    results = []
    for i in range(total_batch):
        x, y = mnist.test.next_batch(batch_size)
        x=np.reshape(x, [batch_size, X_DIMS[0], X_DIMS[1], 1])
        results.append(test(sess, configs, x, y))
    return results


def save_results(configs, results):
    j=0
    print(np.shape(results))
    results = np.sum(results, axis=0)
    results = np.reshape(results, [4,GROUP])
    results = np.swapaxes(results, 0,1)
    s = sample(sess, configs)
    for samp in s:
        plot_mnist_digit(samp, "samples/config-"+str(j)+".png")
        j+=1
    for config, result in zip(configs, results):
        loss = np.array(result)
        print("SHAPE", np.shape(loss))
        #results = np.reshape(results, [results.shape[1], results.shape[0]])
        #g_loss = [g for g,_,_,_ in loss]
        #g_loss = np.mean(g_loss)
        #d_loss = [d for _,d,_,_ in loss]
        #d_loss = np.mean(g_loss)
        #d_fake = [d_ for _,_,d_,_ in loss]
        #d_fake = np.mean(d_fake)
        #d_real = [d for _,_,_,d in loss]
        #d_real = np.mean(d_real)

        g_loss, d_loss, d_fake, d_real = loss
        # calculate D.difficulty = reduce_mean(d_loss_fake) - reduce_mean(d_loss_real)
        difficulty = d_real * (1-d_fake)
        # calculate G.ranking = reduce_mean(g_loss) * D.difficulty
        ranking = g_loss * (1.0-difficulty)

        results =  {
            'difficulty':difficulty,
            'ranking':ranking,
            'g_loss':g_loss,
            'd_loss':d_loss,
            'd_fake':d_fake,
            'd_real':d_real,
            }
        print("results: difficulty %.2f, ranking %.2f, g_loss %.2f, d_fake %.2f, d_real %.2f" % (difficulty, ranking, g_loss, d_fake, d_real))
        hc.record(config, results)

configs = hc.configs(GROUP)
sess = tf.Session()
print("Creating graphs")
graphs = [create_graph(config) for config in configs]
create_optimizers(configs)
print("Graphs created")
init = tf.initialize_all_variables()
sess.run(init)
for i in range(EPOCHS):
    epoch(sess, configs, mnist)
results = test_configs(sess, configs)
save_results(configs, results)
ops.reset_default_graph()

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

