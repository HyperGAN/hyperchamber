import unittest
import tensorflow as tf
import numpy as np

import hyperchamber as hc


class hyperchamber_test(unittest.TestCase):

  def test_gets_empty_config(self):
      hc.reset()
      self.assertEqual(hc.configs(), [])

if __name__ == '__main__':
    unittest.main()
