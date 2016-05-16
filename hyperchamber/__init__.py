store = {}
storeSize=0
def set(key, value):
    global storeSize
    if(isinstance(value, list)):
        storeSize = len(value)
    store[key]=value
    return store


def get_config_value(k, i):
    if(not isinstance(store[k], list)):
        return store[k]
    else:
        return store[k][i]
def configs(max=1, offset=0):
    global storeSize
    if(len(store)==0):
        return []
    configs = []
    for i in range(max):
        # get an element to index over
        if(offset+i >= storeSize):
            break
        config = {}
        for k in store:
            config[k]=get_config_value(k, offset+i)
        configs.append(config)
    return configs

def reset():
    global store
    global storeSize
    store = {}
    storeSize=0
    return
