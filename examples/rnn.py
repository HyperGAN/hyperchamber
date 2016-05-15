#inspired by http://people.idsia.ch/~juergen/icann2009bayer.pdf

import tensorflow as tf
import numpy as np
import hyperchamber as hc
from tensorflow.models.rnn import rnn_cell, rnn


hc.set("learning_rate",
        [0.01])
hc.set("batch_size",
        [128])
hc.set('rnn_size',
        [4])

TRAIN_STEPS=100
TESTS = TRAIN_STEPS/10

def create_rnn(config, x, scope='rnn'):
    with tf.variable_scope(scope):
        memory=config['rnn_size']
        cell = rnn_cell.BasicLSTMCell(memory)
        state = cell.zero_state(batch_size=config['batch_size'], dtype=tf.float32)
        x, state = rnn.rnn(cell, [tf.cast(x,tf.float32)], initial_state=state, dtype=tf.float32)
        x = x[-1]
        #w = tf.get_variable('w', [hc.get('rnn_size'),4])
        #b = tf.get_variable('b', [4])
        #x = tf.nn.xw_plus_b(x, w, b)
        x=tf.sign(x)
        return x, state

# Each step of the graph we have:
# x is [BATCH_SIZE, 4] where the data is an one hot binary vector of the form:
# [start_token end_token a b]
#
# y is [BATCH_SIZE, 4] is a binary vector of the chance each character is correct
#
def create_graph(config, x, y):
    output, state = create_rnn(config, x)
    y = tf.cast(y, tf.float32)
    return tf.reduce_sum(tf.reduce_mean(output*tf.log(y)))

def parallel_run(sess, train_step, costs):
    return [sess.run(train_step, cost) for cost in costs]

def create_optimizer(cost,learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grad_clip = 5.
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    train_step = optimizer.apply_gradients(zip(grads, tvars))
    return train_step


def run():
    #x_input = get_an_bn(1)
    #y_input = get_an_bn_grammar(1)

    print("Training RNNs")
    configs = hc.configs(1, offset=0)
    sess = tf.Session()
    # TODO: Create subgraphs on config
    graph_costs=[]
    optimizers=[]
    for config in configs:
        x = tf.placeholder("bool", [config['batch_size'], 4])
        y = tf.placeholder("bool", [config['batch_size'], 4])
        x_input = [[True,False,False,False],[False,True,False,False]]
        y_input = [[True,True,False,False],[True,False,True,False]]
        graph_cost = create_graph(config, x, y)
        optimizer = create_optimizer(graph_cost,config['learning_rate'])
        graph_costs.append(graph_cost)
        optimizers.append(optimizer)

    for i in range(TRAIN_STEPS):
        _, costs = parallel_run(sess, optimizers, graph_costs)
        hc.cost(configs, costs)

    for config in hc.top_configs(3):
        print(config)

def encode(chars):
    def vectorize(c):
        if(c=="S"):
            return [True,False,False,False]
        if(c=="T"):
            return [False,True,False,False]
        if(c=="a"):
            return [False,False,True,False]
        if(c=="b"):
            return [False,False,False,True]
    return [vectorize(c) for c in chars]


def get_an_bn(n):
    return encode("Sa"*n+"b"*n+"T")
def get_an_bn_an(n):
    return encode("Sa"*n+"b"*n+"a"*n+"T")

# returns a vector of a**nb**n grammars given a set of one-hot vectors x, denoting the next element in a generated sequence
# Each value is returned as an array of possible next inputs.
# for example:
# get_an_bn_grammar(ST) # T
# get_an_bn_grammar(SabT) # T/a-b-T
# get_an_bn_grammar(SaaabbbT) # T/a-a/b-a/b-b-b-b-T
#
# where S and T designate start and end symbols
def get_an_bn_grammar(x):
    return ""

# same as above but with extra 'a's
def get_an_bn_an_grammar(x):
    return ""

run()
