import functools
import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Tuple, NamedTuple

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
flags.DEFINE_integer('toyear', 2021, "Run analysis up to year")
flags.DEFINE_string('priorsample', None, "prior samples from other experiment")

Root = tfd.JointDistributionCoroutine.Root

class PoissonMixtureRegression(NamedTuple):
    w:tf.Tensor
    c:tf.Tensor
    c0:tf.Tensor
    logits:tf.Tensor

    def __call__(self, x:tf.Tensor)->tfd.Distribution:
        log_rate_nnz = x@self.w+self.c
        log_rate0 = tf.broadcast_to(self.c0,log_rate_nnz.shape[:-1]+(1,))
        log_rate = tf.concat([log_rate0, log_rate_nnz], axis=-1)

        return tfd.Independent( distribution=tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=tf.broadcast_to(self.logits, log_rate.shape)),
                    components_distribution=tfd.Poisson(log_rate=log_rate)
            ),
            reinterpreted_batch_ndims=1,
            name='y'
        )
    def Z(self,x:tf.Tensor,y:tf.Tensor)->tfd.Distribution:
        '''
        Rozklad warunkowy `P(Z|x,y,theta)` na element mieszanki
        :param x:
        :param y:
        :return:
        '''
        ydist = self(x)
        inner_dist = ydist.distribution
        lp = inner_dist.components_distribution.log_prob(y) + inner_dist.mixture_distribution.logits_parameter()
        return tfd.Categorical(logits=lp)

    @staticmethod
    def prior(nnz:int,dtype:tf.DType):
        tensor = lambda tx: tf.convert_to_tensor(tx, dtype=dtype)
        return PoissonMixtureRegression(
            w = tfd.Sample(tfd.Normal(loc=tensor(0.5), scale=tensor(1.)), (1,nnz),name='w'),
            c = tfd.Sample(tfd.Normal(loc=tensor(-8.), scale=tensor(3.)), (1,nnz),name='c'),
            c0 = tfd.Sample(tfd.Normal(loc=tensor(-3.), scale=tensor(3.)), (1, 1), name='c0'),
            logits = tfd.Sample(tfd.Normal(loc=tensor(0), scale=tensor(2.)),(1,nnz+1),name='logits')
        )


def load_mcmc(path:str)->np.array:
    if path:
        with open(path, 'br') as f:
            samples = pickle.load(f)

        return tf.nest.map_structure(lambda x: np.reshape(x,(-1,)+x.shape[2:]), samples)
    return None


def poisson_mixture_regression(x:Any,nnz:int=2,prior_samples:Any=None ):
    @tfd.JointDistributionCoroutine
    def model():
        _x = tf.convert_to_tensor(x)
        tensor = lambda tx: tf.convert_to_tensor(tx, dtype=_x.dtype)
        if prior_samples:
            mean_ax0 = functools.partial(np.mean, axis=0)
            std_ax0 = functools.partial(np.std, axis=0)
            means = tf.nest.map_structure(mean_ax0, prior_samples)
            stds  = tf.nest.map_structure(std_ax0, prior_samples)
            w = yield Root(tfd.Independent(tfd.Normal(loc=tensor(means[0]), scale=tensor(stds[0])), 2, name='w'))
            c = yield Root(tfd.Independent(tfd.Normal(loc=tensor(means[1]), scale=tensor(stds[1])), 2, name='c'))
            c0 = yield Root(tfd.Independent(tfd.Normal(loc=tensor(means[2]), scale=tensor(stds[2])), 2, name='c0'))
            logits = yield Root(tfd.Independent(tfd.Normal(loc=tensor(means[3]), scale=tensor(stds[3])), 2, name='logits'))
        else:
            w = yield  Root(tfd.Sample(tfd.Normal(loc=tensor(0.5), scale=tensor(0.5)), (1,nnz),name='w'))
            c = yield Root(tfd.Sample(tfd.Normal(loc=tensor(-8.), scale=tensor(3.)), (1,nnz),name='c'))
            c0 = yield Root(tfd.Sample(tfd.Normal(loc=tensor(-8.), scale=tensor(3.)), (1, 1), name='c0'))
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
    clean_df_t = clean_df[clean_df.publication_date.dt.year <=FLAGS.toyear]

    df = data.fractions_countries(clean_df_t, with_others=FLAGS.others)
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
        nnz,
        prior_samples=load_mcmc(FLAGS.priorsample)
    )

    _y = np.broadcast_to(dataset.y,[n_batch]+list(dataset.y.shape))

    def target_log_prob(w,c,c0,logits):
        return model.log_prob(w,c,c0,logits, _y)


    hmc = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob,
        step_size=[.03,0.05,0.05,0.05])

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
            tf.broadcast_to(tf.convert_to_tensor([0.5,1.],dtype=tf.float64),[num_chains,1,2], name='init_w'),
            -8.*tf.ones([num_chains,1,nnz], name='init_c', dtype=tf.float64),
            -8.+tf.zeros([num_chains,1,1], name='init_c0', dtype=tf.float64),
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
    with open(os.path.join(FLAGS.out,FLAGS.tag), 'wb') as f:
        pickle.dump([t.numpy() for t in samples], f)


    return 0

if __name__ == '__main__':
    app.run(main)