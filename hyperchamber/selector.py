from json import JSONEncoder
import random
import os
import uuid
import json

from .config import Config


# for function serialization
class HCEncoder(JSONEncoder):
    def default(self, o):
        if hasattr(o, '__call__'):  # is function
            return "function:" + o.__module__ + "." + o.__name__
        else:
            try:
                return o.__dict__
            except AttributeError:
                try:
                    return str(o)
                except AttributeError:
                    return super(o)


# for enumerating possible hyperparameter value configurations (with each hyperparameter having possibly several values)
class Selector:
    def __init__(self, initial_store={}):
        """
        Initialize a hyperparameter configuration selector.
        """
        self.store = initial_store
        self.results = []

    def set(self, key, value):
        """
        Set the value (or list of possible values) of a hyperparameter for the selector.

        Can be used to set an array of hyperparameters at once.
        """
        self.store[key] = value
        return self.store

    def count_configs(self):
        """
        Return the number of possible selections of hyperparameter configurations for the selector.
        """
        config_count = 1

        for key in self.store:
            value = self.store[key]
            if isinstance(value, list):
                config_count *= len(value)

        return config_count

    def get_config_value(self, k, i):
        """
        Gets the ith config value for k.  e.g. get_config_value('x', 1).
        """
        if not isinstance(self.store[k], list):
            return self.store[k]
        else:
            return self.store[k][i]

    def configs(self, max_configs=1, offset=None, serial=False, create_uuid=True):
        """
        Generate max configs, each one a dictionary.  e.g. [{'x': 1}]

          Will also add a config UUID, useful for tracking configs.  
          You can turn this off by passing create_uuid=False.
        """
        if len(self.store) == 0:
            return []

        configs = []

        if offset is None:
            offset = max(0, random.randint(0, self.count_configs()))
        for i in range(max_configs):
            # get an element to index over

            config = self.config_at(offset)
            if create_uuid:
                config["uuid"] = uuid.uuid4().hex
            configs.append(config)
            if serial:
                offset += 1
            else:
                offset = max(0, random.randint(0, self.count_configs()))
        return configs

    def config_at(self, i):
        """
        Go through all the possible selections of configurations, and return the ith selection.
        """
        selection = {}
        for key in self.store:
            value = self.store[key]
            if isinstance(value, list):
                selected = i % len(value)
                i = i // len(value)
                selection[key] = value[selected]
            else:
                selection[key] = value

        return Config(selection)

    def random_config(self):
        """
        Get a random configuration selection.
        """
        offset = max(0, random.randint(0, self.count_configs()))
        return self.config_at(offset)

    def reset(self):
        """
        Reset the hyperparameter configuration selector.
        """
        self.store = {}
        self.results = []
        return

    def top(self, sort_by):
        """
        Get the best results according to your custom sort method.
        """
        sort = sorted(self.results, key=sort_by)
        return sort

    def record(self, config, result):
        """
        Record the results of a config.
        """
        self.results.append((config, result))

    # can be made into a function
    @staticmethod
    def load(filename):
        """
        Loads a configuration selector from disk.
        """
        content = open(filename)
        return Config(json.load(content))

    def load_or_create_config(self, filename, config=None):
        """
        Loads a config from disk.  Defaults to a random config if none is specified.
        """
        os.makedirs(os.path.dirname(os.path.expanduser(filename)), exist_ok=True)
        if os.path.exists(filename):
            return Selector.load(filename)

        if config is None:
            config = self.random_config()

        Selector.save(filename, config)
        return config

    # can be made into a function
    @staticmethod
    def save(filename, config):
        """
        Save a configuration selector to disk.
        """
        with open(os.path.expanduser(filename), 'w') as file:
            file.write(json.dumps(config, cls=HCEncoder, sort_keys=True, indent=2, separators=(',', ': ')))
