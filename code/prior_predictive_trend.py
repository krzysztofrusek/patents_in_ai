import os.path
import pickle

import chex
import pandas as pd
from jax.config import config

import util
from logistic_growth import LogisticGrowthSuperposition, _ta, InhomogeneousPoissonProcess
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

# flags.DEFINE_string("pickle", "../dane/clean.pickle", "Input file")
# flags.DEFINE_integer("nkl", 8, "num samples for kl")
# flags.DEFINE_integer("steps", 1000, "num samples for kl")
# flags.DEFINE_integer("seed", 1000, "initial seed")
# flags.DEFINE_string("out", "out", "Output directory")
# flags.DEFINE_bool("coldstart", True, "train or restore files and plot results")
flags.DEFINE_string("paperdir", "../gen/paper", "out dir")
FLAGS= flags.FLAGS


def main(_):
    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
    events=day_events.astype(np.float64)

    train_test_date = np.datetime64('2005-01-01')
    last_idx = np.where(day_events<train_test_date)[0][-1]

    TIME_SCALE=1e4
    events1 = events[:last_idx]/TIME_SCALE
    events2 = events[last_idx:]/TIME_SCALE

    time_boundaries = events[[0,last_idx,-1]]/TIME_SCALE

    counts = np.cumsum(jnp.ones_like(events))

    @hk.transform_with_state
    def model():
        m = LogisticGrowthSuperposition(num_kl=FLAGS.nkl)
        return m()

    rng = jax.random.PRNGKey(FLAGS.seed)

    new_key, rng = jax.random.split(rng,2)
    params, state = model.init(rng)

    prior=tfd.JointDistributionNamed({
        'maximum':tfd.Pareto(1.3, 10759.),#tfd.LogNormal(loc=_ta(26),scale=_ta(4.)),
        'midpoints':tfd.Sample( tfd.Normal(_ta(1.5),_ta(1.)),2 ),
        'rates':tfd.Sample( tfd.Exponential(_ta(0.2)),2 ),
        'mix':tfd.Sample(tfd.Normal(_ta(1), _ta(1.)), 1)
    })

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None))
    def T_stat(ipp:InhomogeneousPoissonProcess, rng:jax.random.PRNGKey)->chex.Array:
        rates = jnp.diff(jax.vmap(InhomogeneousPoissonProcess.cumulative_rate, in_axes=(None, 0))(ipp, time_boundaries))
        return tfd.Poisson(rate=rates).sample(seed=rng)

    ipp = InhomogeneousPoissonProcess(**prior.sample(300000, seed=rng))

    t_stat = T_stat(ipp,new_key)
    #
    # plt.hist(np.log1p(np.sum(t_stat,1)),bins=20, density=True,log=True)
    # plt.axvline(np.log1p(len(events)))

    util.plot_config()
    plt.hist2d(np.log1p(t_stat[:, 0]), np.log1p(t_stat[:, 1]), bins=10, density=True,norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.plot(np.log1p(last_idx), np.log1p(len(events)-last_idx), 'X',color="c", label='data')
    plt.legend(loc='upper left')
    plt.xlabel(r'$T_1$')
    plt.ylabel(r'$T_2$')
    plt.tight_layout()
    plt.savefig(os.path.join(FLAGS.paperdir,'PriorPredictiveChecksTrend.pdf'))
    return 0

if __name__ == '__main__':
    app.run(main)