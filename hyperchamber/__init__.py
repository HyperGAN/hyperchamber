import hyperchamber.io as io

from hyperchamber.selector import *

import importlib


default_selector = Selector()

def set(key, value):
    """Sets a hyperparameter.  Can be used to set an array of hyperparameters."""
    global default_selector
    return default_selector.set(key, value)

def count_configs():
    global default_selector
    return default_selector.count_configs()

def get_config_value(k, i):
    """Gets the ith config value for k.  e.g. get_config_value('x', 1)"""
    global default_selector
    return default_selector.get_config_value(k, i)

def configs(max_configs=1, offset=None, serial=False, create_uuid=True):
    """Generate max configs, each one a dictionary.  e.g. [{'x': 1}] 

      Will also add a config UUID, useful for tracking configs.  
      You can turn this off by passing create_uuid=False.
    """
    global default_selector
    return default_selector.configs(max_configs, offset, serial, create_uuid)

def config_at(i):
  """Gets the ith config"""
  global default_selector
  return default_selector.config_at(i)

def random_config():
  global default_selector
  return default_selector.random_config()

def reset():
    """Reset the hyperchamber variables"""
    global default_selector
    return default_selector.reset()

def top(sort_by):
    """Get the best results according to your custom sort method."""
    global default_selector
    return default_selector.top(sort_by)

def record(config, result):
    """Record the results of a config."""
    global default_selector
    return default_selector.record(config, result)

def load(filename):
    """Loads a config from disk"""
    global default_selector
    return default_selector.load(filename)

def load_or_create_config(filename, config=None):
    """Loads a config from disk.  Defaults to a random config if none is specified"""
    global default_selector
    return default_selector.load_or_create_config(filename, config)

def save(filename, config):
    """Loads a config from disk"""
    global default_selector
    return default_selector.save(filename, config)


# This looks up a function by name.
def get_function(name):
    if not isinstance(name, str):
        return name
    namespaced_method = name.split(":")[1]
    method = namespaced_method.split(".")[-1]
    namespace = ".".join(namespaced_method.split(".")[0:-1])
    return getattr(importlib.import_module(namespace),method)

# Take a config and replace any string starting with 'function:' with a function lookup.
def lookup_functions(config):
    for key, value in config.items():
        if(isinstance(value, str) and value.startswith("function:")):
            config[key]=get_function(value)
        if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
            config[key]=[get_function(v) for v in value]
            
    return config



