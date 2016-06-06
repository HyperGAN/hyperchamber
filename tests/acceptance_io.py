import unittest
import tensorflow as tf
import numpy as np

import hyperchamber as hc
import matplotlib.pyplot as plt

def test_graph(config, filename, n):
  plt.rcdefaults()

  people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
  y_pos = np.arange(len(people))
  performance = 3 + 10 * np.random.rand(len(people))
  error = np.random.rand(len(people))

  plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
  plt.yticks(y_pos, people)
  plt.xlabel("Step "+str(n))
  plt.title('Are intrinsic measurements working?')

  plt.savefig(filename)

hc.io.apikey("TESTAPIKEY")

hc.set("model", "255bits/acceptance_test")
hc.set("version", "0.0.1")
hc.set("test", "acceptance_io")

for config in hc.configs(1):
  filenames = [ "/tmp/acceptance_io_"+str(i)+".png" for i in range(10) ]
  graphs = [test_graph(config, filename, i) for i, filename in enumerate(filenames)]
  hc.io.sample(config, [{'image':f, 'label':f} for f in filenames])

  hc.io.record(config, {'ranking': 1})
  print("Stored config", config)
