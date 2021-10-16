import unittest
import numpy as np
import bayes
import tensorflow as tf

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


if __name__ == '__main__':
    unittest.main()
