# hyperchamber
Track and optimize your hyperparameters.

You set a list of options that define your hyperparams:
```python
import hyperchamber as hc

hc.set('learning_rate', [0.1, 0.2, 0.5])
config = hc.random_config() # => { 'learning_rate' : 0.2 }
```

Optionally integrates with [Hyperchamber](https://hyperchamber.255bits.com) for saving configuration results.

As you train your model you can send samples, and configuration results to hyperchamber.io.

## A quick note from the author

Designing working neural network configurations currently involves a large amount of trial and error.  
A single network parameter can single handedly break a network and can be hard to debug.
With hyperchamber, you can take the hacker approach of:

* test random combinations
* see what works
* repeat as necessary

Then you do not need to worry as much about debugging networks, instead you just are searching for one that works
according to your numbers.

A lot of our work on hyperchambers focuses around GAN (generative adversarial networks).  This is because GANs involve
many networks working with each other.  They are also known for being hard to get working well.

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
  hc.record(config, result)
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

```python
  hc.io.model(mymodel)
```

Models organize your results.  All networks trained on the same model are ranked against each other.

Please use the same test and training set across configs for more accurate model comparisons. 

example: 

```python
  hc.io.model('hypergan')
```
---

```python
  hc.io.sample(config, sample)
```

Send a sample to your hyperchamber.io account.  Samples are intrinsic measurements of your model.

---

```python
  hc.io.record(config, result)
```

Saves the results of your model.  result is a freeform dictionary.


