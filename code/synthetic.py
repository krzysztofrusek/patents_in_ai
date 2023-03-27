import numpy

import bayes
from scipy.stats import bernoulli, norm, multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data
import gravity
#%%
clean = data.load_clean('dane/clean.pickle')
df = data.fractions_countries(clean, with_others=True)
p=np.sign(df).mean().to_numpy()
#%%
from tensorflow_probability.substrates import numpy as tfp
pat = pd.DataFrame(tfp.distributions.Bernoulli(probs=p).sample(500000))
#%%
ds = bayes.Dataset.from_pandas(pat, gravity.CountryFeaturesType.ALL)
plt.scatter(ds.x,ds.y)
plt.yscale('function', functions=(np.log1p, np.expm1))
plt.show()

#%%
idx = ds.x>15

pol = np.polyfit(ds.x[idx], np.log(ds.y[idx]),1)
pol

#%%

A=norm(-200,180).rvs((pp.shape[0],pp.shape[0]))
# tri=np.linalg.cholesky(A@A.T)
#
# aa=A@A.T
# aa= aa/np.diag(aa) + 1e-8*np.eye(aa.shape[0])
ss= norm().rvs((2000,pp.shape[0]))@A



#%% korelacje

pp = np.delete(p,-5)
raw = pat.to_numpy()
raw = np.delete(raw, -5, 1)
c = np.corrcoef(raw.T)


c = np.corrcoef(ss.T)

b=norm().isf(pp)

z=multivariate_normal(cov=c)
synth = z.rvs(200000)>b

#%%
ds = bayes.Dataset.from_pandas(pd.DataFrame(synth.astype(np.float32)), gravity.CountryFeaturesType.ALL)
plt.scatter(ds.x,ds.y)
plt.yscale('function', functions=(np.log1p, np.expm1))
plt.show()

#%%
idx = ds.x>18

pol = np.polyfit(ds.x[idx], np.log(ds.y[idx]),1)
pol
