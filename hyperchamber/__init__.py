import uuid

import hyperchamber.permute as permute
import hyperchamber.io as io

import random

store = {}
results = []

def set(key, value):
    """Sets a hyperparameter.  Can be used to set an array of hyperparameters."""
    store[key]=value
    return store

def count_configs():
    """Counts the total number of configs."""
    if(len(store)==0 and len(permute.store)==0):
      return 0
    
    count = 1

    for key in store:
        value = store[key]
        if(isinstance(value, list) and len(value) > count):
            count = len(value)

    return permute.count_configs(count)

def get_config_value(k, i):
    """Gets the ith config value for k.  e.g. get_config_value('x', 1)"""
    if(not isinstance(store[k], list)):
        return store[k]
    else:
        return store[k][i]

def configs(max_configs=1, offset=None, createUUID=True):
    """Generate max configs, each one a dictionary.  e.g. [{'x': 1}] 
      
      Will also add a config UUID, useful for tracking configs.  
      You can turn this off by passing createUUID=False.
    """
    if(len(store)==0 and len(permute.store)==0):
        return []

    configs = []
    total = count_configs()
    permute_configs = permute.count_configs(1)
    singular_configs = total // permute_configs

    for i in range(max_configs):
        if(offset == None):
          offset = max(0, random.randint(0, count_configs()))
          print("Offset: ", offset)
        # get an element to index over

        config = {}
        for k in store:
            config[k]=get_config_value(k, (offset)//permute_configs)

        more = permute.get_config((offset) % permute_configs)
        config.update(more)
        if(createUUID):
          config["uuid"]=uuid.uuid4().hex
        configs.append(config)
    return configs

def reset():
    """Reset the hyperchamber variables"""
    global store
    global results
    permute.store = {}
    store = {}
    results = []
    return

def top(sort_by):
    """Get the best results according to your custom sort method."""
    sort = sorted(results, key=sort_by)
    return sort

def record(config, result):
    """Record the results of a config."""
    results.append((config, result))
