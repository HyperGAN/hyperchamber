import unittest
import tensorflow as tf
import numpy as np

import hyperchamber as hc
import matplotlib.pyplot as plt
plt.rcdefaults()

people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
plt.yticks(y_pos, people)
plt.xlabel('Performance')
plt.title('Are intrinsic measurements working?')

plt.savefig("/tmp/acceptance_io.png")

#hc.io.apikey("TODO")

hc.set("model", "255bits/acceptance_test")
hc.set("version", "0.0.1")

config = {'test': 'acceptance_io'}
hc.io.sample(config, ["/tmp/acceptance_io.png"])

hc.io.record(config, {'done': True})
