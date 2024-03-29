import os.path
import pickle

import pandas as pd
from jax.config import config

from util import plot_config

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
tfb = tfp.bijectors

import haiku as hk
import data

flags.DEFINE_string("pickle", "../dane/clean.pickle", "Input file")
flags.DEFINE_integer("nkl", 8, "num samples for kl")
flags.DEFINE_integer("steps", 100, "num samples for kl")
flags.DEFINE_integer("seed", 1000, "initial seed")
flags.DEFINE_string("out", "out", "Output directory")
flags.DEFINE_bool("coldstart", True, "train or restore files and plot results")
flags.DEFINE_string("train_test_date", '2020-09-01', 'date for train test spit')

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
        scale_init = lambda  *args: _ta(2) + jnp.zeros(*args)
        loc = hk.get_parameter('loc', shape=self.prior.event_shape, init=init,dtype=jnp.float64)
        log_var = hk.get_parameter('log_var', shape=self.prior.event_shape, init=scale_init,dtype=jnp.float64)
        scale = jnp.sqrt(jnp.exp(log_var))
        #scale=0.00001

        posterior = tfd.Normal(loc=loc, scale=scale)
        posterior = tfd.Independent(posterior, 1 if self.prior.event_shape!=[] else None)

        if self.bijector:
            posterior = tfd.TransformedDistribution(posterior, self.bijector)

        param = posterior.sample(self.num_kl, seed=hk.next_rng_key())
        kl = jnp.mean(posterior.log_prob(param)-self.prior.log_prob(param), axis=0)
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
        return self.maximum

    @property
    def distribution(self):
        #mix = jnp.sort(self.mix)
        #mix = jnp.cumsum(self.mix,axis=-1)
        mix = self.mix
        zero = jnp.zeros_like(mix)
        mix = jnp.concatenate([mix,zero], axis=-1)

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=mix),
            components_distribution=tfd.Logistic(loc=self.midpoints, scale=self.rates)
        )

    def log_rate(self,events:jnp.ndarray):
        return jnp.log(self.capacity)+ self.distribution.log_prob(events)

    def cumulative_rate(self,x:jnp.ndarray):
        return self.capacity * self.distribution.cdf(x)

    @functools.partial(jax.vmap, in_axes=(0,None))
    def log_prob(self,events:jnp.ndarray):
        return jnp.sum(self.log_rate(events)) - \
               (self.cumulative_rate(events[-1]) - self.cumulative_rate(events[0]))

def _ta(x):
    '''Typed array'''
    a = jnp.asarray(x)
    return a.astype(jnp.float64)

class LogisticGrowthSuperposition(hk.Module):
    def __init__(self, num_kl:int=4,name=None):
        super().__init__(name=name)

        self.maximum = NormalPosterior(
            #prior=(tfd.LogNormal(loc=_ta(26),scale=_ta(4.))),
            prior=tfd.Pareto(_ta(1.3), _ta(11e3)),
            num_kl=num_kl,
            initial=500.,
            #bijector=tfb.Chain([tfb.Scale(_ta(1e2)),tfb.Softplus()]),
            bijector=tfb.Chain([tfb.Shift(_ta(11e3)),tfb.Softplus(),tfb.Scale(_ta(1e2))]),
            name = 'maximum'
        )
        self.midpoints= NormalPosterior(
            prior=tfd.Sample( tfd.Normal(_ta(1.5),_ta(1.)),2 ),num_kl=num_kl,
            initial=1.5,
            #bijector=tfb.Scale(_ta(1e4)),
            name='midpoints'
        )
        self.rates = NormalPosterior(
            prior=tfd.Sample( tfd.Exponential(_ta(0.2)),2 ),num_kl=num_kl,
            bijector=tfb.Chain([tfb.Softplus()]),
            initial=5.,
            name='rates'
        )
        self.mix = NormalPosterior(
            prior=tfd.Sample(tfd.Normal(_ta(1), _ta(1.)), 1),
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
    plot_config()

    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
    events=day_events.astype(np.float64)

    train_test_date = np.datetime64(FLAGS.train_test_date)
    last_idx = np.where(day_events<train_test_date)[0][-1]

    TIME_SCALE=1e4
    train_events = events[:last_idx]/TIME_SCALE
    test_events = events[last_idx:]/TIME_SCALE

    counts = np.cumsum(jnp.ones_like(events))

    @hk.transform_with_state
    def model():
        m = LogisticGrowthSuperposition(num_kl=FLAGS.nkl)
        return m()

    rng = jax.random.PRNGKey(FLAGS.seed)

    new_key, rng = jax.random.split(rng,2)
    params, state = model.init(rng)
    print(state)


    def elbo_loss(params, state, rng, t):
        dist, state = model.apply(params, state, rng)
        nll = -jnp.mean(dist.log_prob(t))
        total_kl = sum(kl for kl in  jax.tree_leaves(
            hk.data_structures.filter(lambda module_name, name, value: name == 'kl', state)))
        return (nll + total_kl)

    @jax.jit
    def enll(params, state, rng, t):
        dist, state = model.apply(params, state, rng)
        nll = -jnp.mean(dist.log_prob(t))

        return nll

    grad_fn = jax.jit(jax.grad(elbo_loss))
    loss_fn = jax.jit(elbo_loss)

    opt = optax.adam(0.08)
    opt_state = opt.init(params)
    eval_stats=[]

    if FLAGS.coldstart:
        for i in range(FLAGS.steps):
            new_key, rng = jax.random.split(rng, 2)
            grads = grad_fn(params, state, new_key, train_events)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if i % 10 ==0:
                logging.info(f'step {i}, loss {loss_fn(params, state, new_key, train_events)}')
                expected_train_nll = enll(params, state, new_key, train_events)
                expected_test_nll = enll(params, state, new_key, test_events)
                logging.info(f'step {i}, train nll {expected_train_nll}, test nll {expected_test_nll}')
                eval_stats.append(dict(step=i,expected_train_nll=expected_train_nll,expected_test_nll=expected_test_nll))
                #print(params)

        with open(os.path.join(FLAGS.out, 'stat_params.pickle'), 'wb') as f:
            pickle.dump((params, state), f)

        df = pd.DataFrame(eval_stats)
        f=plt.figure()
        plt.plot(df['step'],df['expected_train_nll'],label='expected_train_nll')
        plt.plot(df['step'], df['expected_test_nll'], label='expected_test_nll')
        plt.legend()
        plt.savefig(os.path.join(FLAGS.out, 'nll.pdf'))
        plt.close(f)
        df.to_csv(os.path.join(FLAGS.out, 'nll.csv'))
    else:
        with open(os.path.join(FLAGS.out,'stat_params.pickle'),'rb') as f:
            params, state = pickle.load(f)

    dist, _ = model.apply(params, state, rng)
    logging.info(dist)
    df = pd.DataFrame(np.concatenate(dist[1:], axis=1),columns=[r'$t_{0,1}$',r'$t_{0,2}$',r'$s_{1}$',r'$s_{2}$',r'$p_{1}$'])
    df['$C$']=dist.capacity
    df[r'$p_{1}$'] = jax.nn.sigmoid(df[r'$p_{1}$'].to_numpy())
    df.iloc[:, :4] = (TIME_SCALE * df.iloc[:, :4])
    df.iloc[:, :2] = df.iloc[:, :2].astype('datetime64[D]')
    df.iloc[:, 2:4] = df.iloc[:, 2:4].astype('timedelta64[D]')

    try:
        means = df.mean(axis=0, numeric_only=False)
        stds=df.std(axis=0, numeric_only=False)

        means['stat']='mean'
        stds['stat'] = 'std'
        tex = pd.concat([means, stds], axis=1).T.to_latex(escape=False)

        with open(os.path.join(FLAGS.out,'params.tex'), 'wt') as f:
            f.write(tex)
    except:
        pass


    logging.info('midpoints')
    logging.info(np.asarray(np.mean(dist.midpoints, axis=0)*1e4).astype('datetime64[D]'))

    logging.info('+-')
    logging.info(2*1e4*np.std(dist.midpoints, axis=0))


    extra_time = np.linspace(events[-1],np.datetime64('2035-01-01').astype(np.float64),100)
    plot_events = np.concatenate((events,extra_time))
    plotx = plot_events.astype('datetime64[D]')


    @functools.partial(jax.vmap, in_axes=(0,None))
    def v_cum_rate(dist, t):
        cum_rate = dist.cumulative_rate(t)
        return cum_rate-cum_rate[0]

    v_log_rate = jax.vmap(InhomogeneousPoissonProcess.log_rate, in_axes=(0,None))

    cum_rates = v_cum_rate(dist, plot_events/TIME_SCALE)
    rates = jnp.exp(v_log_rate(dist, plot_events))

    mean_cum_rate = jnp.mean(cum_rates, axis=0)
    cl,ch = np.quantile(cum_rates,q=[0.025,0.975], axis=0)
    f = plt.figure()
    plt.plot(plotx, mean_cum_rate, label='cumulative rate')
    plt.plot(day_events,counts,label='counts')

    plt.fill_between(plotx,cl,ch,alpha=0.5, label=r'95 % interval')

    plt.axvline(train_test_date, linestyle=':', color='k',label='train end')
    plt.ylabel('Patents')

    plt.legend()
    plt.yscale('log')
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
        rates = rate_component(dist,plot_events/TIME_SCALE,i)/TIME_SCALE
        plt.plot(plot_events, np.mean(rates, axis=0),label=f'$\lambda_{i+1}(t)$')
        cl, ch = np.quantile(rates, q=[0.025, 0.975], axis=0)
        plt.fill_between(plotx, cl, ch, alpha=0.3, label=r'95 % interval')


    plt.axvline(alexnet_date, linestyle=':', color='grey',label='alexnet')
    plt.axvline(train_test_date, linestyle=':', color='k', label='train end')
    plt.axvline(day_events[-1],linestyle='-.', color='grey', label='data end')
    plt.ylabel('Rate [patents per day]')
    plt.legend()

    plt.savefig(os.path.join(FLAGS.out,'rate.pdf'))



    return 0

if __name__ == '__main__':
    app.run(main)