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
            #components_distribution=tfd.Logistic(loc=self.loc*13e3+7e3, scale=self.scale*13e3)
            components_distribution = tfd.Logistic(loc=self.loc, scale=self.scale)
        )
        return dist


    def __call__(self, x:jnp.array)->jnp.array:

        return self.a*self.dist.cdf(x)

    def pack(self)->jnp.array:
        return jnp.concatenate(self)

    @classmethod
    def unpack(cls,dense:jnp.array):
        return cls(*jnp.split(dense,(1,3,5)))

#%%

class InhomogeneousPoissonProcess(LogisticGrowthV3):

    @property
    def dist(self):
        dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.mix),
            components_distribution=tfd.Logistic(loc=self.loc*13e3+7e3, scale=self.scale*13e3)
            #components_distribution = tfd.Logistic(loc=self.loc, scale=self.scale)
        )
        return dist
    def __call__(self, x:jnp.array)->jnp.array:

        return 1e4*self.a*self.dist.prob(x)

    def log_call(self, x:jnp.array)->jnp.array:
        return jnp.log(1e4) + jnp.log(self.a)+ self.dist.log_prob(x)

    def integral(self,x):
        '''
        :param x:
        :return:
        '''

        return 1e4*self.a*self.dist.cdf(x)



#%%
ipp = InhomogeneousPoissonProcess(
    a=jnp.array([1.68]),
    loc=jnp.array([0.72078344, 0.90014877]),
    scale=jnp.array([0.19851322, 0.03673386]),
    mix = jnp.array([0.40917747, 1.09082253])
)
hat = ipp.integral(events)
plt.plot(events,counts)
plt.plot(events,hat)
plt.show()

hat = ipp(events)
plt.plot(events,hat)
plt.show()
#%%
td = (t*13000+7000).astype('datetime64[D]')
alexnet_date=np.datetime64('2012-09-10')
for i in range(2):

    plt.plot(events,ipp.a*1e4*ipp.dist.components_distribution[i].prob(events))

plt.axvline(alexnet_date,linestyle=':',color='k')


plt.show()
#%%
x= np.linspace(0,10,100)
y=tfd.LogNormal(loc=[1,], scale=0.5).prob(x)
plt.plot(x,y)
plt.show()
#%%
clean_df = data.load_clean("dane/clean.pickle")
day_events = clean_df.publication_date.sort_values().to_numpy().astype('datetime64[D]')
#day_events = day_events[50:]
#day_events = day_events[50:]
events=day_events.astype(np.float64)

#%% nowe

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


wyznaczone = InhomogeneousPoissonProcess(
    maximum=5.68,
    midpoints=jnp.array([0.9968559 , 1.8940382 ]),
    rates=jnp.array([0.05962872, 0.59377897]),
    mix=jnp.array([ 0.08877622, -0.1729958 ])

)

wyznaczone = InhomogeneousPoissonProcess(
    maximum=jnp.array([4.118548]),
    midpoints=jnp.array([1.084058  , 0.9827988 ]),
    rates=jnp.array([0.3611422 , 0.05136992]),
    mix=jnp.array([ [-0.46248153,  0.36811638]  ])

)

hat = wyznaczone.cumulative_rate(events)-wyznaczone.cumulative_rate(events[0])
plt.plot(events,counts)
plt.plot(events,hat)
plt.show()

hat = np.exp(wyznaczone.log_rate(events))
plt.plot(events,hat)
plt.show()
#%%
alexnet_date=np.datetime64('2012-09-10')
for i in range(2):

    plt.plot(events,wyznaczone.capacity*wyznaczone.distribution.components_distribution[i].prob(events))

plt.axvline(alexnet_date,linestyle=':',color='k')
plt.show()

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
        ll = jnp.mean(m.log_call(events)) - jnp.sum(m.integral(x[-1]) -m.integral(x[0])) / x.shape[
            0]  # remove dim of size 1
        return -ll/2e3
    return opt.minimize(nll,
                        x0=initial.pack(),
                        method="BFGS",
                        options=dict(maxiter=500000, line_search_maxiter=1000)
                        )

@jax.jit
def ols_poison(x:jnp.array, initial:InhomogeneousPoissonProcess):
    y = jnp.cumsum(jnp.ones_like(x))
    def nll(theta:jnp.array)->jnp.array:
        m = InhomogeneousPoissonProcess.unpack(theta)
        #ll = jnp.mean(jnp.log(m(events))) - jnp.sum(m.integral(x[-1])-m.integral(x[0]))/x.shape[0] # remove dim of size 1
        ll = jnp.mean(jnp.square(y-m.integral(x))) /1e3
        return ll
    #return nll
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
hat = ipp.integral(events)
plt.plot(events,counts)
plt.plot(events,hat)
plt.show()

#%%
opt_result = ols_poison(events,ipp)
opt_result = ols_poison(events,ipp)
#%%
opt_result = fit_poison(events,ipp)
fitted_ipp = InhomogeneousPoissonProcess.unpack(opt_result.x)
print(opt_result.success)
#%%
hat = fitted_ipp.integral(events)
plt.plot(events,counts)
plt.plot(events,hat)
plt.show()

#%%
opt_result = ols_poison(events,ipp)
fitted_ipp = InhomogeneousPoissonProcess.unpack(opt_result.x)
print(opt_result.success)
#%%
hat = fitted_ipp.integral(events)-fitted_ipp.integral(events[0])
plt.plot(events,counts)
plt.plot(events,hat)
plt.show()

#%%
td = (events).astype('datetime64[D]')
alexnet_date=np.datetime64('2012-09-10')
for i in range(2):

    plt.plot(td,fitted_ipp.dist.components_distribution[i].prob(events))

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


