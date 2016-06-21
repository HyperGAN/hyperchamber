import tensorflow as tf
tensors = {}
def set_tensor(name, tensor):
  tensors[name]=tensor

def get_tensor(name, graph=tf.get_default_graph(), isOperation=False):
    return tensors[name]# || graph.as_graph_element(name)


