import unittest
import tensorflow as tf
import numpy as np

import hyperchamber as hc


class hyperchamber_test(unittest.TestCase):
    def test_set(self):
        hc.reset()
        hc.set('x', [1])
        self.assertEqual(hc.configs(1), [{'x':1}])

    def test_set2(self):
        hc.reset()
        hc.set('x', [1,2])
        self.assertEqual(hc.configs(2), [{'x':1},{'x':2}])

    def test_pagination(self):
        hc.reset()
        hc.set('x', [1,2])
        self.assertEqual(hc.configs(1), [{'x':1}])
        self.assertEqual(hc.configs(1, offset=1), [{'x':2}])
        self.assertEqual(hc.configs(1, offset=2), [])

    def test_constant_set(self):
        hc.reset()
        hc.set('x', 1)
        hc.set('y', [2,3])
        self.assertEqual(hc.configs(1), [{'x':1, 'y':2}])
        self.assertEqual(hc.configs(1, offset=1), [{'x':1, 'y':3}])
        self.assertEqual(hc.configs(1, offset=2), [])

    def test_set2_2vars(self):
        hc.reset()
        hc.set('x', [1,2])
        hc.set('y', [3,4])
        self.assertEqual(hc.configs(2), [{'x':1,'y':3},{'x':2,'y':4}])

    def test_configs(self):
        hc.reset()
        self.assertEqual(hc.configs(), [])


    def test_record(self):
        hc.reset()
        def do_nothing(x):
            return 0
        self.assertEqual(hc.top(1, sort_by=do_nothing), [])
        config = {'a':1}
        result = {'b':2}
        hc.record(config, result)
        self.assertEqual(hc.top(1, sort_by=do_nothing)[0], (config, result))

    def test_reset(self):
        hc.reset()
        self.assertEqual(hc.configs(), [])
        self.assertEqual(hc.results, [])
        self.assertEqual(hc.store_size, 0)

    def test_store_size(self):
        hc.reset()
        hc.set('x', [1,2])
        self.assertEqual(hc.store_size, 2)

    def test_top(self):
        hc.reset()
        for i in range(10):
            config = {"i": i}
            result = {"cost": 10-i}
            hc.record(config, result)

        def by_cost(x):
            config,result = x
            return result['cost']

        self.assertEqual(hc.top(1, sort_by=by_cost)[0], ({'i': 9}, {'cost': 1}))


if __name__ == '__main__':
    unittest.main()
