import dataclasses
import os.path
import pickle

import scipy.stats.contingency

import data
import gravity
import bayes
import util
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from absl import flags, app, logging

import util
from util import plot_config
from scipy.stats import norm
from scipy.stats.contingency import crosstab
plot_config()

#%%
def reshaper(x):
    s = x.shape
    return np.reshape(x, (-1,) + s[-2:])

def load_samples(path:str)->np.ndarray:
    with open(path, 'rb') as f:
        s = pickle.load(f)
        all_long = np.reshape(np.concatenate(s, axis=-1), (-1, 8))
        return all_long


#%%
s05 = load_samples('plg/mcmc3_s05/samples.pkl')
s20 = load_samples('plg/mcmc3_s20/samples.pkl')
s10 = load_samples('plg/mcmc3/samples.pkl')

s = [ s05,s20,s10]
labels=[r'$0.5\sigma$',r'$2.0\sigma$',r'$1.0\sigma$ (used)']



@dataclasses.dataclass
class Prior:
    priors:list

    @staticmethod
    def make():
        return Prior(priors=2*[norm(0.5,0.5)]+3*[norm(-8., 3.)]+3*[norm(0,2)])

    def __call__(self, i:int, scale:float):
        p = self.priors[i]
        return norm(p.mean(), scale*p.std())


#%%
plt.hist2d(s20[:,1],s20[:,4],50,density=True,norm=mpl.colors.LogNorm())
crosstab(s20[:,1]<0.8,s20[:,4]>0)
#plt.hist(s20[:,4],100)
plt.show()
#%%
priors = Prior.make()

fig, axs = plt.subplots(3,3, sharex=False, sharey=False,constrained_layout=True,figsize=(4.7,4.7))
columns=[r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\beta_0$', r'$l_0$', r'$l_1$', r'$l_2$']


scales=[0.5,1.0,2.0]
axs_bac = axs
axs = axs.ravel()
for i in range(8):
    for j in range(3):
        axs[i].hist(s[j][:,i], bins=20, density=True,histtype='step', label=labels[j])
        axs[i].set_xlabel(columns[i])

for ax in axs_bac[:,0]:
    ax.set_ylabel('density')
# for i in range(8):
#     axs[i].set_prop_cycle(None)
#     for j in range(3):
#
#         lims = axs[i].get_xlim()
#         center = np.mean(lims)
#         width = lims[1]-lims[0]
#         x = np.linspace(center-0.5*width, center+0.5*width, 100)
#         x = np.linspace(lims[0],lims[1], 100)
#         axs[i].plot(x, priors(i, scales[j]).pdf(x),':')



handles, labels = axs[0].get_legend_handles_labels()

leg = axs[-1].legend(handles, labels)
axs[-1].set_axis_off()

plt.savefig('gen/paper/priorImpact.pdf')
plt.show()

#%% inventors

s05 = load_samples('plg/mcmc_inv_s05/samples.pkl')
s20 = load_samples('plg/mcmc_inv_s20/samples.pkl')
s10 = load_samples('plg/mcmc_inv/samples.pkl')

s = [ s05,s20,s10]
labels=[r'$0.5\sigma$',r'$2.0\sigma$',r'$1.0\sigma$ (used)']


fig, axs = plt.subplots(3,3, sharex=False, sharey=False,constrained_layout=True,figsize=(4.7,4.7))
columns=[r'$\alpha_1$', r'$\alpha_2$', r'$\beta_1$', r'$\beta_2$', r'$\beta_0$', r'$l_0$', r'$l_1$', r'$l_2$']


scales=[0.5,1.0,2.0]
axs_bac = axs
axs = axs.ravel()
for i in range(8):
    for j in range(3):
        axs[i].hist(s[j][:,i], bins=20, density=True,histtype='step', label=labels[j])
        axs[i].set_xlabel(columns[i])

for ax in axs_bac[:,0]:
    ax.set_ylabel('density')


handles, labels = axs[0].get_legend_handles_labels()

leg = axs[-1].legend(handles, labels)
axs[-1].set_axis_off()

plt.savefig('gen/paper/priorImpactInventor.pdf')
plt.show()
