store = {}
def set(key, value):
    store[key]=value
    return store

def configs(max=1, offset=0):
    if(len(store)==0):
        return []
    configs = []
    for i in range(max):
        # get an element to index over
        values = store[list(store)[0]]
        if(offset+i >= len(values)):
            break
        config = {}
        for k in store:
            config[k]=store[k][offset+i]
        configs.append(config)
    return configs

def reset():
    global store
    store = {}
    return
