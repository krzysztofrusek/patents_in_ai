import os.path
import pickle

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
flags.DEFINE_bool("coldstart", True, "train or restore files and plot results")

FLAGS= flags.FLAGS

Root = tfd.JointDistributionCoroutine.Root








def main(_):
    try:
        mpl.use('MacOSX')
    except:
        mpl.use('Agg')

    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
    events=day_events.astype(np.float64)
    counts = np.cumsum(jnp.ones_like(events))

    @tfd.JointDistributionCoroutine
    def model():
        maximum = yield Root(tfd.LogNormal(loc=[5, ], scale=8.,name='maximum'))
        midpoints = yield Root(tfd.Sample( tfd.Normal(1.0,10.),2,name='midpoints' ))
        rates = yield Root(tfd.Sample( tfd.Exponential(0.5),2,name='rates' ))
        mix = yield Root(tfd.Sample(tfd.Normal(0, 1.), 2,name='mix'))
        # maximum = yield Root(tfd.LogNormal(loc=[0.5, ], scale=5.,name='maximum'))
        # midpoints = yield Root(tfd.Sample( tfd.Normal(1.0,10.),2,name='midpoints' ))
        # rates = yield Root(tfd.Sample( tfd.Exponential(1.3),2,name='rates' ))
        # mix = yield Root(tfd.Sample(tfd.Normal(0, 1.), 2,name='mix'))


    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0,None))
    def ratefn(sample,events):
        maximum, midpoints, rates, mix = sample
        mixdist =  tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=jnp.cumsum(mix, axis=-1)),
                components_distribution=tfd.Logistic(loc=midpoints*13e3+7e3, scale=rates*13e3),name='rate'
            )
        cumrate = 1e3*maximum*mixdist.cdf(events)

        return  jnp.stack((jnp.mean(cumrate, axis=0), jnp.std(cumrate, axis=0)))


    rng = jax.random.PRNGKey(32137)

    s = model.sample(20000,seed=rng)
    #plt.plot(events, s[4][:,0,0])
    T = ratefn(s, events)

    Tl = np.log(T)
    safe_idx = np.any(Tl> -10, axis=1)
    Tl = Tl[safe_idx,:]
    sns.jointplot(x=Tl[:,0],y=Tl[:,1],kind='hex')
    plt.plot(
        np.log(np.mean(counts)),
        np.log(np.std(counts)),'rX'
    )
    plt.show()


    return 0

if __name__ == '__main__':
    app.run(main)