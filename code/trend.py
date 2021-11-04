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

#%%

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

    def pack(self)->jnp.array:
        return jnp.concatenate(self)

    @classmethod
    def unpack(cls,dense:jnp.array):
        return cls(*jnp.split(dense,(1,3,5)))

class InhomogeneousPoissonProcess(LogisticGrowthV3):

    def __call__(self, x:jnp.array)->jnp.array:

        return 1e4*self.a*self.dist.prob((x-7000.)/13000.)

    def log_call(self, x:jnp.array)->jnp.array:
        return jnp.log(1e4) + jnp.log(self.a)+ self.dist.log_prob((x - 7000.) / 13000.)

    def integral(self,x):
        '''
        :param x:
        :return:
        '''

        return 1e4*self.a*self.dist.cdf((x-7000.)/13000.)



#%%
clean_df = data.load_clean("dane/clean.pickle")
day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
#day_events = day_events[50:]
#day_events = day_events[50:]
events=day_events.astype(np.float64)
#%%
counts = np.cumsum(jnp.ones_like(events))
plt.plot(events,counts)
plt.yscale('log')

plt.show()
#%%
x = (events-7000)/13000
y= counts/10000
plt.plot(x,y)
plt.show()
#%%
model = LogisticGrowthV3(
    a=jnp.array([1.]),
    loc=jnp.array([0.4,0.9]),
    scale=jnp.array([0.4,0.1]),
    mix = jnp.array([-0.5,2.])
)
t = np.linspace(np.min(x),np.max(x),500)
hat = model(t)
plt.plot(x,y)
plt.plot(t,hat)
plt.show()

#%%
@jax.jit
def fit(x:jnp.array,y:jnp.array,initial:LogisticGrowthV3):
    def nll(theta:jnp.array)->jnp.array:
        m = LogisticGrowthV3.unpack(theta)
        ll = jnp.mean(jnp.square(y-m(x))) # remove dim of size 1
        return ll

    return opt.minimize(nll,
                        x0=initial.pack(),
                        method="BFGS",
                        options=dict(maxiter=500000, line_search_maxiter=1000)
                        )

@jax.jit
def fit_poison(x:jnp.array, initial:InhomogeneousPoissonProcess):
    def nll(theta:jnp.array)->jnp.array:
        m = InhomogeneousPoissonProcess.unpack(theta)
        #ll = jnp.mean(jnp.log(m(events))) - jnp.sum(m.integral(x[-1])-m.integral(x[0]))/x.shape[0] # remove dim of size 1
        ll = jnp.mean(m.log_call(events)) - jnp.sum(m.integral(x[-1]) - m.integral(x[0])) / x.shape[
            0]  # remove dim of size 1
        return -ll
    return opt.minimize(nll,
                        x0=initial.pack(),
                        method="BFGS",
                        options=dict(maxiter=500000, line_search_maxiter=1000)
                        )


#%%
ipp = InhomogeneousPoissonProcess(
    a=jnp.array([1.68]),
    loc=jnp.array([0.72078344, 0.90014877]),
    scale=jnp.array([0.19851322, 0.03673386]),
    mix = jnp.array([0.40917747, 1.09082253])
)

opt_result = fit_poison(events,ipp)
fitted_ipp = InhomogeneousPoissonProcess.unpack(opt_result.x)
print(opt_result.success)
#%%
hat = fitted_ipp.integral(events)
plt.plot(events,counts)
plt.plot(events,hat)
plt.show()

#%%

td = (t*13000+7000).astype('datetime64[D]')
alexnet_date=np.datetime64('2012-09-10')
for i in range(2):

    plt.plot(td,fitted_ipp.dist.components_distribution[i].prob(t))

plt.axvline(alexnet_date,linestyle=':',color='k')


plt.show()


#%%
opt_result=fit(x[:-400],y[:-400],model)
fitted = LogisticGrowthV3.unpack(opt_result.x)
#  LogisticGrowthV3(a=DeviceArray([1.68389389], dtype=float64), loc=DeviceArray([0.72078344, 0.90014877], dtype=float64), scale=DeviceArray([0.19851322, 0.03673386], dtype=float64), mix=DeviceArray([0.40917747, 1.09082253], dtype=float64))

#%%
hat = fitted(t)
plt.plot(x,y)
plt.plot(t,hat)
plt.show()
#%%
f = jax.jit(jax.vmap(jax.grad(lambda x: fitted(x)[0] )))

plt.plot(t,f(t))
plt.show()

#%%
td = (t*13000+7000).astype('datetime64[D]')
alexnet_date=np.datetime64('2012-09-10')
for i in range(2):

    plt.plot(td,10000*fitted.a*fitted.dist.components_distribution[i].prob(t))

plt.axvline(alexnet_date,linestyle=':',color='k')


plt.show()


#%%

#events = events[:-2000]/1000
#events = events[5000:]/1000

model = LogisticGrowthV3(
    a=20000,
    loc=jnp.array([14e3,20e3]),
    scale=jnp.array([5e3,1e3]),
    mix = jnp.array([-0.5,2.])
)

t = np.linspace(8e3,20e3,500)
hat = model(t)
plt.plot(events,counts)
plt.plot(t,hat)
plt.show()