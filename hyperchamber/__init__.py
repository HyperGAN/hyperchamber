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
        for k in store:
            rand=store[k][0]
            config[k]=rand
        configs.append(config)
    return configs

def reset():
    store = {}
    return
