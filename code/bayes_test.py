import unittest
from functools import partial
import numpy as np
import bayes
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

class BayesTestCase(unittest.TestCase):
    def test_batch(self):

        n=4
        x = np.random.normal(loc=10, scale=2, size=(n,351,1)).astype(np.float32)
        model = bayes.poisson_mixture_regression(x, 2)

        s = model.sample(n)

        @tf.function(jit_compile=True)
        def f(x):
            return model.log_prob(x)

        f(s)
        return 0
    def test_no_batch(self):

        n=4
        x = np.random.normal(loc=10, scale=2, size=(351,1)).astype(np.float32)
        model = bayes.poisson_mixture_regression(x, 2)

        s = model.sample()

        @tf.function(jit_compile=True)
        def f(x):
            return model.log_prob(x)

        f(s)
        return 0


class TupleModelTestCase(unittest.TestCase):

    def test_prior(self):
        prior = bayes.PoissonMixtureRegression.prior(2, tf.float64)

        n = 4
        sample_model = bayes.PoissonMixtureRegression(*map(partial(tfd.Distribution.sample,sample_shape=n), prior))
        x = np.random.normal(loc=10, scale=2, size=(n,351, 1))
        sample_model(x)
        ...

    def test_jdc(self):
        n = 4
        x = np.random.normal(loc=10, scale=2, size=(n, 351, 1))
        Root = tfd.JointDistributionCoroutine.Root
        def makejdc():
            @tfd.JointDistributionCoroutine
            def model():
                params=[]
                for p in bayes.PoissonMixtureRegression.prior(2,tf.float64):
                    param = yield Root(p)
                    params.append(param)
                yield bayes.PoissonMixtureRegression(*params)(x)
            return model

        model = makejdc()
        model.log_prob(model.sample())
        ...


if __name__ == '__main__':
    unittest.main()
