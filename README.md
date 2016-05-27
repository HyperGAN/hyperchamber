# hyperchamber
Track and optimize your tensorflow hyperparameters.

# Examples

Use hyperchamber for:

* Run more experiments

Train multiple neural networks at once, exploiting the parallelism of your GPU

* Track your results across experiments:

examples/track

Use hyperchamber.evolve for:

* Finding a good learning rate for MNIST:

examples/hypertune-mnist/

* Evolve a network that fits MNIST:

examples/evolve-mnist/

* Evolve RNN types:

examples/evolve-rnn/

* Evolve GAN (HyperGAN):

examples/evolve-gan/

* Report your trainings with hyperchamber.report:

examples/report

* Run multiple experiments in parallel


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
  hc.set(name, value)
```

Set a series of hyperparameters.  Note, value must be a vector of length n, where each call to set has length n.

```python
  hc.permute.set(name, values)
```

Permute over a series of hyperparameters.

```python
  hc.configs(n)
```
Returns up to n configs of the form {name:value} for each set parameter.


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


