# hyperchamber
Random search your hyper parameters.

## Changelog

### 0.2

* local save/load

### 0.1

* initial pip release

You set a list of options that define your hyperparams:
```python
import hyperchamber as hc

hc.set('learning_rate', [0.1, 0.2, 0.5])
config = hc.random_config() # => { 'learning_rate' : 0.2 }
```

## Examples

* logistic regression classifier on MNIST [code](examples/track.py)

  Based on a simple tensorflow example. We find the best learning rate from a small set of options.

* Finding a better network architecture for MNIST [code](examples/mnist.py)

  Uses hyperparameter tuning to find the best performing MNIST fully connected deep network configuration.

  Our search space of options here is now 720 options.  Note we only have 2 variables.  This search space expands exponentially with new options to search.


This is currently in development and not ready for use (yet).

# Installation

## Developer mode

```
  python setup.py develop
```

# API

```python
  import hyperchamber as hc
```
```python
  hc.set(name, values)
```

Sets a hyperparameter to values.  

* If values is an array, config[name] will be set to one element in that array.
* If values is a scalar, config[name] will always be set to that scalar

```python
  hc.configs(n)
```
Returns up to n configs of the form {name:value} for each hyperparameter.


```python
	hc.save(config, filename)
```
Saves the config to a file.

```python
	hc.load(filename)
```
Load a configuration from file

```python
	hc.load_or_create_config(filename, config)
```
Load a configuration from file if that file exists.  Otherwise save `config` to that file.  `config` is assumed to be a Dictionary.



```python
  hc.record(filename, config)
```
Store the cost of a config's training results. 


```python
  hc.top(sort_by)
```

Return the top results across all recorded results

Example:

```python
  def by_cost(x):
    config, result =x
    return result['cost']
  for config, result in hc.top(by_cost): 
    print(config, result)
```

## hyperchamber.io

Hyperchamber.io allows you to save and share your hyperparameters across runs and across organizations.

It is currently in an open alpha.  You can sign up at https://hyperchamber.255bits.com

---

```python
  hc.set('model', modelandtag)
```

Models organize your results.  All networks trained on the same model are ranked against each other.

Please use the same test and training set across configs for more accurate model comparisons. 

example: 

```python
  hc.set('model', 'mnist:0.1')
```
---

```python
  hc.io.sample(config, sample)
```

Send a sample to your hyperchamber.io account.  Samples are intrinsic aspects of your model.

---

```python
  hc.io.measure(config, measurement)
```

Saves measurements for your model.  measurement is a dictionary with numerical values.

example:
```
  measurement = { 'loss': 0.2, 'seconds_taken': 30, 'my_custom_measurement' : 10 }
```

