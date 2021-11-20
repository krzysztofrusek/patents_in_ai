import os.path

from jax.config import config
config.update("jax_enable_x64", True)

import functools

import optax
import numpy as np
import matplotlib as mpl
import seaborn as sns
from  matplotlib import pyplot as plt

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
flags.DEFINE_integer("seed", 1000, "initial seed")
flags.DEFINE_string("out", "out", "Output directory")

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
            prior=tfd.LogNormal(loc=[5.0,], scale=2.),num_kl=num_kl,
            initial=2.,
            name = 'maximum'
        )
        self.midpoints= NormalPosterior(
            prior=tfd.Sample( tfd.Normal(1.0,10.),2 ),num_kl=num_kl,
            name='midpoints'
        )
        self.rates = SofplusNormalPosterior(
            prior=tfd.Sample( tfd.Exponential(0.5),2 ),num_kl=num_kl,
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
            mix = jnp.cumsum(self.mix(), axis=-1)
        )



def main(_):
    # try:
    #     mpl.use('MacOSX')
    # except:
    #mpl.use('Agg')

    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
    events=day_events.astype(np.float64)

    train_test_date = np.datetime64('2020-09-01')
    last_idx = np.where(day_events<train_test_date)[0][-1]

    train_events = events[:last_idx]

    counts = np.cumsum(jnp.ones_like(events))

    @hk.transform_with_state
    def model():
        m = LogisticGrowthSuperposition(num_kl=FLAGS.nkl)
        return m()

    rng = jax.random.PRNGKey(144)

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
        grads = grad_fn(params, state, new_key, train_events)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 10 ==0:
            logging.info(f'step {i}, loss {loss_fn(params, state, new_key, train_events)}')
            #print(params)

    dist, _ = model.apply(params, state, rng)
    logging.info(dist)

    @functools.partial(jax.vmap, in_axes=(0,None))
    def v_cum_rate(dist, t):
        cum_rate = dist.cumulative_rate(t)
        return cum_rate-cum_rate[0]

    v_log_rate = jax.vmap(InhomogeneousPoissonProcess.log_rate, in_axes=(0,None))

    cum_rates = v_cum_rate(dist, events)
    rates = jnp.exp(v_log_rate(dist, events))

    mean_cum_rate = jnp.mean(cum_rates, axis=0)
    cl,ch = np.quantile(cum_rates,q=[0.025,0.975], axis=0)
    f = plt.figure()
    plt.plot(day_events, mean_cum_rate, label='cumulative rate')
    plt.plot(day_events,counts,label='counts')

    plt.fill_between(day_events,cl,ch,alpha=0.5, label=r'95 % interval')

    plt.axvline(train_test_date, linestyle=':', color='k',label='train end')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FLAGS.out,'cumrate.pdf'))

    alexnet_date = np.datetime64('2012-09-10')

    @functools.partial(jax.vmap, in_axes=(0, None,None))
    def rate_component(dist:InhomogeneousPoissonProcess,events,i):
        return dist.distribution.mixture_distribution.probs_parameter()[ i]*\
               dist.distribution.components_distribution[i].prob(events)*dist.capacity
    plt.close(f)
    f = plt.figure()

    for i in range(2):
        rates = rate_component(dist,events,i)
        plt.plot(events, np.mean(rates, axis=0),label=f'$\lambda_{i+1}(t)$')
        cl, ch = np.quantile(rates, q=[0.025, 0.975], axis=0)
        plt.fill_between(day_events, cl, ch, alpha=0.3, label=r'95 % interval')


    plt.axvline(alexnet_date, linestyle=':', color='grey',label='alexnet')
    plt.axvline(train_test_date, linestyle=':', color='k', label='train end')

    plt.legend()

    plt.savefig(os.path.join(FLAGS.out,'rate.pdf'))

    return 0

if __name__ == '__main__':
    app.run(main)