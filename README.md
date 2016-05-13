# hyperchamber
Track and optimize your tensorflow hyperparameters.

To run multiple experiments in parallel:

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

# Running in parallel
```python
  x = x[:hc.get('batch_size')]

  # sess is your tensorflow session
  #runs 5 experiments in parallel
  _, costs = hc.run(sess, train_step, {hc.getTensor('x'):x}, parallel=5)
```

This is currently in development and not ready for use (yet).


# API

```python
hyperchamber.set(name, value)
```

Set a series of hyperparameters.  Note, value must be a vector of length n, where each call to set has length n.

```python
hyperchamber.configs(n)
```
Returns up to n configs of the form {name:value} for each set parameter.


```python
hyperchamber.cost(config, cost)
```
Store the cost of a config's training results. 


