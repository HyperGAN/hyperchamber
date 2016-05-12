import tensorflow as tf
import hyperchamber as hc

hc.set("batch_size", 128)

def create_graph(x):

def train():

def run():
    rand = np.random_uniform()#TODO as int
    graph = create_graph(x)
    batch = get_an_bn_grammar(rand)
    _, cost = sess.run(train_step, graph['cost'])

# returns a vector of a**nb**n grammars given a set of integer values x, denoting a list of 'n's
# for example
# get_an_bn_grammar(0) # empty string
# get_an_bn_grammar(1) # an
# get_an_bn_grammar(3) # aaannn
def get_an_bn_grammar(x):
    return ""

# same as above but with extra 'a's
def get_an_bn_an_grammar(x):
    return ""

