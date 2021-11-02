from jax.config import config
config.update("jax_enable_x64", True)

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

import data

flags.DEFINE_string("pickle", "../dane/clean.pickle", "Input file")

FLAGS= flags.FLAGS



class LogisticGrowth(NamedTuple):
    '''
    https://en.wikipedia.org/wiki/Logistic_function

    '''
    L:jnp.array
    k:jnp.array
    x0:jnp.array

    @property
    def _L(self):
        return 1000.*self.L
        #return  self.L
    def __call__(self, x:jnp.array)->jnp.array:
        return self._L/(1+ jnp.exp(-self.k*(x-self.x0)))

    def logistic_integral(self,x):
        '''
        https://www.wolframalpha.com/input/?i=integrate+L%2F%281%2Be%5E%28-k+%28x-x0%29%29%29
        :param x:
        :return:
        '''
        #integral = self.L*jnp.log(jnp.exp(self.k*(x-self.x0))+1)/self.k + self.L * x
        integral = self._L * jax.nn.softplus(self.k * (self.x0-x)) / self.k + self._L * x
        return integral

    def pack(self):
        return jnp.stack(self)

    @staticmethod
    def unpack(dense:jnp.array)->'LogisticGrowth':
        return LogisticGrowth(*jnp.split(dense,3))

class LogisticGrowthV2(LogisticGrowth):

    def __call__(self, x):
        f =  jax.jit(jax.vmap(jax.grad(super(LogisticGrowthV2, self).__call__)))
        return f(x)

    def logistic_integral(self,x):
        return super(LogisticGrowthV2, self).__call__(x)


@jax.jit
def fit(events:jnp.array,initial:LogisticGrowth,t0:float, T:float):
    def nll(theta:jnp.array)->jnp.array:
        m = LogisticGrowth.unpack(theta)
        ll = jnp.sum(jnp.log(m(events))) - jnp.sum(m.logistic_integral(T)-m.logistic_integral(t0)) # remove dim of size 1
        return -ll/1e4

    return opt.minimize(nll,
                        x0=initial.pack(),
                        method="BFGS",
                        options=dict(maxiter=500000, line_search_maxiter=1000)
                        )

@jax.jit
def fit2(events:jnp.array,initial:LogisticGrowthV2,t0:float, T:float):
    def nll(theta:jnp.array)->jnp.array:
        m = LogisticGrowthV2.unpack(theta)
        ll = jnp.sum(jnp.log(m(events))) - jnp.sum(m.logistic_integral(T)-m.logistic_integral(t0)) # remove dim of size 1
        return -ll/1e6

    return opt.minimize(nll,
                        x0=initial.pack(),
                        method="BFGS",
                        options=dict(maxiter=500000, line_search_maxiter=1000)
                        )


class LogisticGrowthV3(NamedTuple):
    '''
    https://en.wikipedia.org/wiki/Logistic_function

    '''
    a:jnp.array
    loc:jnp.array
    scale:jnp.array
    mix:jnp.array

    @property
    def dist(self):
        dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.mix),
            components_distribution=tfd.Logistic(loc=self.loc, scale=self.scale)
        )
        return dist


    def __call__(self, x:jnp.array)->jnp.array:

        return self.a*self.dist.cdf(x)

def main(_):

    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
    #day_events = day_events[50:]
    day_events = day_events[50:]
    events=day_events.astype(np.float64)
    #events = events[:-2000]/1000
    #events = events[5000:]/1000
    counts = np.cumsum(jnp.ones_like(events))
    model = LogisticGrowthV3(
        a=20000,
        loc=jnp.array([1e4,18e3]),
        scale=jnp.array([3e3,4e3]),
        mix = jnp.array([-2.,1.])
    )

    hat = model(events)

    t0 = events[0]
    T = events[-1]

    #events = (events - t0)/(T-t0)

    model =LogisticGrowthV2(
        L=15.,
        k=1.5,
        x0=18.
    )
    model =LogisticGrowthV2(
        L=6365.,
        k=0.46,
        x0=34.42
    )
    opt_result = fit2(events,initial=model,t0=t0,T=T)
    fited = LogisticGrowthV2.unpack(opt_result.x)
    fited=LogisticGrowthV2(*map(lambda x: float(x),fited))




    slambda = 1/(pd.DataFrame(np.diff(events),index=day_events[:-1] ).ewm(alpha=0.02).mean())
    slambda.plot()
    _lambda = fited(events)
    plt.plot(day_events,1/_lambda)
    plt.yscale('log')
    #plt.plot(day_events, np.cumsum(np.ones(day_events.shape)))

    plt.show()

    plt.plot(day_events, np.cumsum(np.ones(day_events.shape)))
    cum_lambda = model.logistic_integral(events)
    plt.plot(day_events, cum_lambda)
    plt.show()

    slambda = 1 / (pd.DataFrame(np.diff(events), index=day_events[:-1]).ewm(alpha=0.02).mean())
    slambda.plot()
    plt.plot(day_events, np.cumsum(np.ones(day_events.shape)))
    cum_lambda = fited.logistic_integral(events)
    plt.plot(day_events, cum_lambda)
    cum_lambda = fited(events)
    plt.plot(day_events, cum_lambda)
    plt.show()

    cum_lambda = fited.logistic_integral(events)
    plt.plot(day_events, cum_lambda)
    cum_lambda = fited(events)
    plt.plot(day_events, cum_lambda)
    plt.show()




    return 0

if __name__ == '__main__':
    app.run(main)