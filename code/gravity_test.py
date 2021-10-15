import unittest

import data
import gravity
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class GravityTestCase(unittest.TestCase):
    def test_2(self):
        x = np.zeros(20)
        m = gravity.PoissonGravitationalModel()
        ydist = m(x)
        self.assertEqual(len(ydist.components),2)

    def test_3(self):
        x = np.zeros(20)
        m = gravity.PoissonGravitationalModel(nnz=2)
        ydist = m(x)
        self.assertEqual(len(ydist.components),3)

class ZTest(unittest.TestCase):
    def test_z_dist(self):
        mix = tfd.Mixture(
            cat=tfd.Categorical(logits=np.zeros(2)),
            components=[tfd.Normal(loc=np.asarray(l), scale=1.) for l in [1.,3.]]
        )
        x=2.
        lp=tf.stack([c.log_prob(x) for c in mix.components])+mix.cat.logits_parameter()
        d = tfd.Categorical(logits=lp)
        self.assertTrue(np.isclose(d.prob(0).numpy(), 0.5))

    def test_real_data_z(self):
        clean_df = data.load_clean('../dane/clean.pickle')
        df = data.fractions_countries(clean_df, with_others=True)
        e = gravity.Estimator(
            data=df,
            model=gravity.PoissonGravitationalModel(
                nnz=2,
                trainable_lograte=True
            ),
            bootstrap=False,
            mass=gravity.CountryFeaturesType.ALL
        )
        e.load('../gen/bootstrap/0')
        Z = e.model.Z_dist(e.x, e.y)
        mix = tfd.Mixture(
            cat=tfd.Categorical(logits=np.zeros((2))),
            components=2*[tfd.Independent(Z,1)]
        )


if __name__ == '__main__':
    unittest.main()
