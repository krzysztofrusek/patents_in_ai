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


def main(_):

    clean_df = data.load_clean(FLAGS.pickle)
    day_events = clean_df.application_date.sort_values().to_numpy().astype('datetime64[D]')
    day_events = day_events[50:]
    events=day_events.astype(np.float64)
    #events = events[:-2000]/1000
    events = events/1000
    t0 = events[0]
    T = events[-1]

    #events = (events - t0)/(T-t0)

    model =LogisticGrowth(
        L=15.,
        k=1.5,
        x0=18.
    )
    opt_result = fit(events,initial=model,t0=t0,T=T)
    fited = LogisticGrowth.unpack(opt_result.x)




    slambda = (pd.DataFrame(np.diff(events),index=day_events[:-1] ).ewm(alpha=0.5).mean())
    slambda.plot()
    _lambda = fited(events)
    plt.plot(day_events,1/_lambda)
    plt.yscale('log')

    plt.show()





    return 0

if __name__ == '__main__':
    app.run(main)