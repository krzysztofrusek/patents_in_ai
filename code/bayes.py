import pickle
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

flags.DEFINE_integer("num_results", 5, "...")
flags.DEFINE_integer("num_chains", 2, "...")
flags.DEFINE_integer("num_adaptation", 10, "...")
flags.DEFINE_integer("num_burnin_steps", 2, "...")
#flags.DEFINE_string("datafile",'Dane2019_koreta_miernika.xlsx',"data excell")
flags.DEFINE_string('tag', "samples.pkl", "Name of the run and output file")


Root = tfd.JointDistributionCoroutine.Root


def poisson_mixture_regression(x:Any,nnz:int=2, ):
    @tfd.JointDistributionCoroutine
    def model():
        _x = tf.convert_to_tensor(x)
        tensor = lambda tx: tf.convert_to_tensor(tx, dtype=_x.dtype)
        w = yield  Root(tfd.Sample(tfd.Normal(loc=tensor(0.5), scale=tensor(1.)), (1,nnz),name='w'))
        c = yield Root(tfd.Sample(tfd.Normal(loc=tensor(-8.), scale=tensor(3.)), (1,nnz),name='c'))
        c0 = yield Root(tfd.Sample(tfd.Normal(loc=tensor(-3.), scale=tensor(3.)), (1, 1), name='c0'))
        logits = yield Root(tfd.Sample(tfd.Normal(loc=tensor(0), scale=tensor(2.)),(1,nnz+1),name='logits'))

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
    _x = dataset.x[..., np.newaxis]

    num_adaptation=FLAGS.num_adaptation
    num_chains=FLAGS.num_chains
    num_results=FLAGS.num_results
    num_burnin_steps=FLAGS.num_burnin_steps

    n_batch = num_chains
    nnz = FLAGS.nnz

    model = poisson_mixture_regression(
        np.broadcast_to(_x,[n_batch]+list(_x.shape)),
        nnz)

    _y = np.broadcast_to(dataset.y,[n_batch]+list(dataset.y.shape))

    def target_log_prob(w,c,c0,logits):
        return model.log_prob(w,c,c0,logits, _y)


    hmc = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob,
        step_size=[.05,0.1,0.1,0.05])

    hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc,
        bijector=[
            tfp.bijectors.Identity(),
            tfp.bijectors.Identity(),
            tfp.bijectors.Identity(),
            tfp.bijectors.Identity()
        ])

    hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc,
        num_adaptation_steps=num_adaptation,
        target_accept_prob=.8)


    initial_state = [
            0.8*tf.ones([num_chains,1, nnz], name='init_w', dtype=tf.float64),
            -8.*tf.ones([num_chains,1,nnz], name='init_c', dtype=tf.float64),
            -3.+tf.zeros([num_chains,1,1], name='init_c0', dtype=tf.float64),
            tf.zeros([num_chains,1,nnz+1], name='init_logits', dtype=tf.float64),
        ]

    @tf.function(autograph=False, jit_compile=True)
    def run():
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=initial_state,
            kernel=hmc,
            num_burnin_steps=num_burnin_steps,
            trace_fn=lambda _, kr: kr
        )

    samples, traces = run()
    print('R-hat diagnostics: ', tfp.mcmc.potential_scale_reduction(samples))
    with open(FLAGS.tag, 'wb') as f:
        pickle.dump([t.numpy() for t in samples], f)


    return 0

if __name__ == '__main__':
    app.run(main)