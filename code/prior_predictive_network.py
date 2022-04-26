import os.path
import pickle

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

FLAGS= flags.FLAGS
flags.DEFINE_string("mcmcpickle", "../gen/mcmc3/samples.pkl", "Input file")
flags.DEFINE_string("paperdir", "../gen/paper", "out dir")

def main(_):
    clean_df = data.load_clean(FLAGS.pickle)
    clean_df = clean_df[clean_df.publication_date.dt.year <= FLAGS.toyear]
    fractions = data.fractions_countries(clean_df, with_others=True)

    with open(FLAGS.mcmcpickle, 'br') as f:
        samples = pickle.load(f)

    dataset = bayes.Dataset.from_pandas(fractions, gravity.CountryFeaturesType.ALL)
    _x = dataset.x[..., np.newaxis]
    n_batch =1
    nnz=2
    model = bayes.poisson_mixture_regression(
        np.broadcast_to(_x,[n_batch]+list(_x.shape)),
        nnz,
        prior_samples=None,
        llz=-8.
    )

    prior_predictive_sample = model.sample(100000)
    counts = prior_predictive_sample[-1]

    patents_features = np.broadcast_to(_x.T,counts.shape).flatten()
    sampled_counts = counts.numpy().flatten()

    # plt.hist2d(x=patents_features,y=np.log1p(sampled_counts))
    # plt.scatter(dataset.x, np.log1p(dataset.y))
    # plt.ylim(0,np.log1p(5000))
    # kwargs = dict(range=(0,np.log1p(2000)),histtype='step',log=False,cumulative=True,bins=20)
    # plt.hist(np.log1p(sampled_counts),density=True,label='prior',**kwargs)
    # plt.hist(np.log1p(dataset.y),density=True,label='data',**kwargs)
    # plt.legend()
    # plt.show()
    util.plot_config()

    plt.hist2d(np.mean(np.log1p(counts), axis=1),np.std(np.log1p(counts), axis=1), bins=10, density=True,norm=mpl.colors.LogNorm())
    plt.plot(np.mean(np.log1p(dataset.y)),np.std(np.log1p(dataset.y)),'X',color="c",label='data')
    plt.legend(loc='upper left')
    plt.xlabel(r'$\mathrm{mean}\quad\log(C_{ij}+1)$')
    plt.ylabel(r'$\mathrm{std}\quad\log(C_{ij}+1)$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(FLAGS.paperdir,'PriorPredictiveChecks.pdf'))
    return 0

if __name__ == '__main__':
    app.run(main)