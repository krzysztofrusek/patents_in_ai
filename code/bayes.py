from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd
from absl import flags, app, logging

import tensorflow as tf
import tensorflow_probability as tfp

import data

tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

import gravity
FLAGS = flags.FLAGS

Root = tfd.JointDistributionCoroutine.Root


def poisson_mixture_regression(x:Any,nnz:int=2, ):
    @tfd.JointDistributionCoroutine
    def model():
        _x = tf.convert_to_tensor(x)

        w = yield  Root(tfd.Sample(tfd.Normal(loc=0.5, scale=1.),(1,nnz),name='w'))
        c = yield Root(tfd.Sample(tfd.Normal(loc=-8., scale=3.), (1,2),name='c'))
        c0 = yield Root(tfd.Sample(tfd.Normal(loc=-3., scale=3.), (1, 1), name='c0'))
        logits = yield Root(tfd.Sample(tfd.Normal(loc=0, scale=2.),(1,nnz+1),name='logits'))

        log_rate_nnz = _x@w+c
        log_rate0 = tf.broadcast_to(c0,log_rate_nnz.shape[:-1]+[1])
        log_rate = tf.concat([log_rate0, log_rate_nnz], axis=-1)

        yield tfd.Independent(
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=tf.broadcast_to(logits, log_rate.shape)),
                components_distribution=tfd.Poisson(log_rate=log_rate),
                name='y'
            ),1,name='y',
        )
        pass
    return model

# tmp

@dataclass
class Dataset:
    x:np.array
    y:np.array
    flat_idx:Tuple[np.array,np.array]
    names: List[str]

    @staticmethod
    def from_pandas(df:pd.DataFrame, mass_type:gravity.CountryFeaturesType,dt=np.float32)->'Dataset':
        A = np.array(df)
        C = np.zeros((A.shape[1], A.shape[1]))
        flat_idx = np.triu_indices(df.shape[1],k = 1)

        for v in A:
            v = np.sign(v)
            M = v[..., np.newaxis] @ v[np.newaxis, ...]
            M -= np.diag(np.diag(M))
            C += M

        if mass_type == gravity.CountryFeaturesType.ALL:
            S = np.sum(A,axis=0)[..., np.newaxis]
        elif mass_type == gravity.CountryFeaturesType.ONLY_COOPERATION:
            S = A[(np.all(A != 1, axis=1)), :].sum(axis=0)[..., np.newaxis]
        elif mass_type == gravity.CountryFeaturesType.INDIVIDUAL:
            S = A[np.any(A == 1, axis=1), :].sum(axis=0)[..., np.newaxis]

        y = C[flat_idx]
        sst = S@S.T
        x = sst[flat_idx]

        where_x, = np.where(x)
        dataset = Dataset(
            x = np.log(x[where_x]),
            y = y[where_x],
            flat_idx = tuple(fi[where_x] for fi in flat_idx),
            names=list(df.columns)
        )
        return dataset




def main(_):
    clean_df = data.load_clean(FLAGS.pickle)
    df = data.fractions_countries(clean_df, with_others=FLAGS.others)
    dataset = Dataset.from_pandas(df,gravity.CountryFeaturesType.ALL)
    _x = dataset.x.astype(np.float32)[..., np.newaxis]
    n_batch =4
    model = poisson_mixture_regression(
        np.broadcast_to(_x,[n_batch]+list(_x.shape)),
        2)

    s = model.sample(n_batch)
    @tf.function(jit_compile=True)
    def f(x):
        return model.log_prob(x)
    print(f(s))
    return 0

if __name__ == '__main__':
    app.run(main)