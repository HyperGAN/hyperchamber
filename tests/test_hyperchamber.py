import unittest
import tensorflow as tf
import numpy as np

import hyperchamber as hc


class hyperchamber_test(unittest.TestCase):
    def test_set(self):
        hc.reset()
        hc.set('x', [1])
        self.assertEqual(hc.configs(1, offset=0, serial=True,create_uuid=False), [{'x':1}])

    def test_set2(self):
        hc.reset()
        hc.set('x', [1,2])
        self.assertEqual(hc.configs(2, offset=0, serial=True,create_uuid=False), [{'x':1},{'x':2}])

    def test_Config_accessor(self):
        hc.reset()
        hc.set('x', [1])
        config = hc.configs(1, offset=0, serial=True,create_uuid=False)[0]
        self.assertEqual(config.x, 1)

    def test_createUUID(self):
        hc.reset()
        hc.set('x', [1])
        self.assertTrue(len(hc.configs(1)[0]["uuid"]) > 1)


    def test_pagination(self):
        hc.reset()
        hc.set('x', [1,2])
        self.assertEqual(hc.configs(1, create_uuid=False, serial=True,offset=0), [{'x':1}])
        self.assertEqual(hc.configs(1, create_uuid=False, serial=True,offset=1), [{'x':2}])
        self.assertEqual(hc.configs(1, create_uuid=False, serial=True,offset=2), [{'x':1}])

    def test_constant_set(self):
        hc.reset()
        hc.set('x', 1)
        hc.set('y', [2,3])
        self.assertEqual(hc.configs(1, create_uuid=False,serial=True, offset=0), [{'x':1, 'y':2}])
        print("--")
        self.assertEqual(hc.configs(1, create_uuid=False, serial=True,offset=1), [{'x':1, 'y':3}])
        self.assertEqual(hc.configs(1, create_uuid=False,serial=True, offset=2), [{'x':1, 'y':2}])

    def test_set2_2vars(self):
        hc.reset()
        hc.set('x', [1])
        hc.set('y', [3,4])
        self.assertEqual(hc.configs(2, create_uuid=False, serial=True,offset=0), [{'x':1,'y':3},{'x':1,'y':4}])

    def test_configs(self):
        hc.reset()
        self.assertEqual(hc.configs(create_uuid=False), [])


    def test_record(self):
        hc.reset()
        def do_nothing(x):
            return 0
        self.assertEqual(hc.top(sort_by=do_nothing), [])
        config = {'a':1}
        result = {'b':2}
        hc.record(config, result)
        self.assertEqual(hc.top(sort_by=do_nothing)[0], (config, result))

    def test_reset(self):
        hc.reset()
        self.assertEqual(hc.configs(), [])
        self.assertEqual(hc.count_configs(), 1)

    def test_store_size(self):
        hc.reset()
        hc.set('x', [1,2])
        self.assertEqual(hc.count_configs(), 2)

    def test_top(self):
        hc.reset()
        for i in range(10):
            config = {"i": i}
            result = {"cost": 10-i}
            hc.record(config, result)

        def by_cost(x):
            config,result = x
            return result['cost']

        self.assertEqual(hc.top(sort_by=by_cost)[0], ({'i': 9}, {'cost': 1}))


if __name__ == '__main__':
    unittest.main()
