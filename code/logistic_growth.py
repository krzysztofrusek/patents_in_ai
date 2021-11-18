from jax.config import config
config.update("jax_enable_x64", True)

import functools

import optax
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import jax.scipy.optimize as opt

from typing import NamedTuple

from absl import flags, app, logging
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
import haiku as hk
import data

flags.DEFINE_string("pickle", "../dane/clean.pickle", "Input file")
flags.DEFINE_integer("nkl", 8, "num samples for kl")
flags.DEFINE_integer("steps", 1000, "num samples for kl")

FLAGS= flags.FLAGS

#%%
# https://github.com/deepmind/deepmind-research/blob/master/counterfactual_fairness/causal_network.py

class NormalPosterior(hk.Module):
    def __init__(self,prior:tfd.Distribution,num_kl=1,bijector=None,initial=None, name=None):
        super().__init__(name=name)

        self.prior = prior
        self.num_kl = num_kl
        self.bijector = bijector
        self.initial = initial


    def __call__(self):
        if self.initial:
            init = lambda  *args: self.initial + jnp.zeros(*args)
        else:
            init = lambda  *args: jnp.mean(self.prior.mean()) + jnp.zeros(*args)
        scale_init = lambda  *args: 2 + jnp.zeros(*args)
        loc = hk.get_parameter('loc', shape=self.prior.event_shape, init=init)
        log_var = hk.get_parameter('log_var', shape=self.prior.event_shape, init=scale_init)
        scale = jnp.sqrt(jnp.exp(log_var))
        #scale=0.00001

        posterior = tfd.Normal(loc=loc, scale=scale)
        posterior = tfd.Independent(posterior, 1 if self.prior.event_shape!=[] else None)

        if self.bijector:
            posterior = tfd.TransformedDistribution(posterior, self.bijector)

        param = posterior.sample(self.num_kl, seed=hk.next_rng_key())
        kl = jnp.mean(posterior.log_prob(param)-self.prior.log_prob(param), axis=0)
        #kl = jnp.mean(self.prior.log_prob(param)-posterior.log_prob(param) , axis=0)
        hk.set_state('kl',kl)

        return param

class SofplusNormalPosterior(NormalPosterior):
    def __init__(self,prior:tfd.Distribution,num_kl=1,initial=None, name=None):
        super().__init__(prior=prior,num_kl=num_kl,bijector=tfp.bijectors.Softplus(),initial=initial,name=name)


class InhomogeneousPoissonProcess(NamedTuple):
    maximum:jnp.ndarray
    midpoints:jnp.ndarray
    rates:jnp.ndarray
    mix:jnp.ndarray

    @property
    def capacity(self):
        return 1e4*self.maximum

    @property
    def distribution(self):
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.mix),
            components_distribution=tfd.Logistic(loc=self.midpoints*13e3+7e3, scale=self.rates*13e3)
        )

    def log_rate(self,events:jnp.ndarray):
        return jnp.log(self.capacity)+ self.distribution.log_prob(events)

    def cumulative_rate(self,x:jnp.ndarray):
        return self.capacity * self.distribution.cdf(x)

    @functools.partial(jax.vmap, in_axes=(0,None))
    def log_prob(self,events:jnp.ndarray):
        return jnp.sum(self.log_rate(events)) - \
               (self.cumulative_rate(events[-1]) - self.cumulative_rate(events[0]))


class LogisticGrowthSuperposition(hk.Module):
    def __init__(self, num_kl:int=4,name=None):
        super().__init__(name=name)

        self.maximum = SofplusNormalPosterior(
            prior=tfd.LogNormal(loc=[0.5,], scale=2.),num_kl=num_kl,
            initial=2.,
            name = 'maximum'
        )
        self.midpoints= NormalPosterior(
            prior=tfd.Sample( tfd.Normal(1.,1.),2 ),num_kl=num_kl,
            name='midpoints'
        )
        self.rates = SofplusNormalPosterior(
            prior=tfd.Sample( tfd.Exponential(3.0),2 ),num_kl=num_kl,
            name='rates'
        )
        self.mix = NormalPosterior(
            prior=tfd.Sample(tfd.Normal(0, 1.), 2),
            num_kl=num_kl,
            name='mix'
        )

    def __call__(self):

        return InhomogeneousPoissonProcess(
            maximum=self.maximum(),
            midpoints=self.midpoints(),
            rates=self.rates(),
            mix = self.mix()
        )



def main(_):

    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
    events=day_events.astype(np.float64)

    events = events[:-300]

    counts = np.cumsum(jnp.ones_like(events))

    @hk.transform_with_state
    def model():
        m = LogisticGrowthSuperposition(num_kl=FLAGS.nkl)
        return m()

    rng = jax.random.PRNGKey(44)

    new_key, rng = jax.random.split(rng,2)
    params, state = model.init(rng)


    def elbo_loss(params, state, rng, t):
        dist, state = model.apply(params, state, rng)
        nll = -jnp.mean(dist.log_prob(t))
        total_kl = sum(kl for kl in  jax.tree_leaves(
            hk.data_structures.filter(lambda module_name, name, value: name == 'kl', state)))
        return (nll + total_kl)

    grad_fn = jax.jit(jax.grad(elbo_loss))
    loss_fn = jax.jit(elbo_loss)

    opt = optax.adam(0.08)
    opt_state = opt.init(params)

    for i in range(FLAGS.steps):
        new_key, rng = jax.random.split(rng, 2)
        grads = grad_fn(params, state, new_key, events)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 10 ==0:
            logging.info(f'step {i}, loss {loss_fn(params, state, new_key, events)}')
            #print(params)

    dist, _ = model.apply(params, state, rng)
    logging.info(dist)


    return 0

if __name__ == '__main__':
    app.run(main)