store = {}

def set(key, value):
    store[key]=value
    return store

def count_configs(prior_count):
    count = prior_count
    for key in store:
        value = store[key]
        count *= len(value)

    return count

def get_config(i):
  """Gets the ith config"""
  selections = {}
  count = count_configs(1)
  for key in store:
    value = store[key]
    selected = i % len(value)
    i = i // len(value)
    selections[key]= value[selected]

  return selections


