import unittest
import tensorflow as tf
import numpy as np

import hyperchamber as hc
import matplotlib.pyplot as plt

def test_graph(config, filename):
  plt.rcdefaults()

  people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
  y_pos = np.arange(len(people))
  performance = 3 + 10 * np.random.rand(len(people))
  error = np.random.rand(len(people))

  plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
  plt.yticks(y_pos, people)
  plt.xlabel('Performance')
  plt.title('Are intrinsic measurements working?')

  plt.savefig(filename)

#hc.io.apikey("TODO")

hc.set("model", "255bits/acceptance_test")
hc.set("version", "0.0.1")
hc.set("test", "acceptance_io")

for config in hc.configs(1):
  filename = "/tmp/acceptance_io.png"
  hc.io.sample(config, [filename])

  hc.io.record(config, {'done': True})
  print("Stored config", config)
