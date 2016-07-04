# hyperchamber
Track and optimize your tensorflow hyperparameters.

Integrates with [Hyperchamber](https://hyperchamber.255bits.com)

# Examples

Use hyperchamber for:

* Track your results across experiments:

[logistic regression classifier on MNIST](examples/track.py)

Based on a simple tensorflow example. We find the best learning rate from a small set of options.

[Finding a better network architecture for MNIST](examples/mnist.py)

Uses hyperparameter tuning to find the best performing MNIST fully connected deep network configuration.

Our search space of options here is 

* Evolve a network that fits MNIST:

examples/evolve-mnist/

* Evolve RNN types:

examples/evolve-rnn/

* Evolve GAN (HyperGAN):

examples/evolve-gan/

* Report your trainings with hyperchamber.report:

examples/report

* Run multiple experiments in parallel

Train multiple neural networks at once, exploiting the parallelism of your GPU

# Running in parallel
```python
  x = x[:hc.get('batch_size')]

  # sess is your tensorflow session
  #runs 5 experiments in parallel
  _, costs = hc.run(sess, train_step, {hc.getTensor('x'):x}, parallel=5)
```

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

It is currently in alpha state.

```python
  hc.io.apikey(apikey)
```

Set the apikey you will use.

```python
  hc.io.model(mymodel)
```

Models organize your results.  All networks trained on the same model are ranked against each other.

Please use the same test and training set across configs for more accurate model comparisons. 

example: hc.io.model('255bits/hypergan.hc')

```python
  hc.io.sample(config, sample)
```

Send a sample to your hyperchamber.io account.  Samples are intrinsic measurements of your model.

Note:  this issues a rate limited HTTP request.  Samples more frequent than the rate limit are not synced. 

```python
  hc.io.record(config, result)
```

Saves the results of your model.  Call this instead of hc.record.


