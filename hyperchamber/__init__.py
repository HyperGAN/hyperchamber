store = {}
def set(key, value):
    store[key]=value
    return store

def configs(max=1):
    if(len(store)==0):
        return []
    configs = []
    for i in range(max):
        config = {}
        # get an element to index over
        values = store[list(store)[0]]
        for k in store:
            config[k]=store[k][i]
        configs.append(config)
    return configs

def reset():
    store = {}
    return
