import uuid

store = {}
store_size=0
results = []

def set(key, value):
    global store_size
    if(isinstance(value, list)):
        store_size = len(value)
    store[key]=value
    return store


def get_config_value(k, i):
    if(not isinstance(store[k], list)):
        return store[k]
    else:
        return store[k][i]
def configs(max=1, offset=0):
    global store_size
    if(len(store)==0):
        return []
    configs = []
    for i in range(max):
        # get an element to index over
        if(offset+i >= store_size):
            break
        config = {}
        for k in store:
            config[k]=get_config_value(k, offset+i)
        configs.append(config)
    return configs

def reset():
    global store
    global store_size
    global results
    store = {}
    store_size=0
    results = []
    return

def top(n, sort_by):
    sort = sorted(results, key=sort_by)
    return sort

def record(config, result):
    results.append((config, result))
