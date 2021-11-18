import functools
import unittest
from typing import NamedTuple
import functools

import logistic_growth as lg
import haiku as hk
import jax
import jax.numpy as jnp
tfd = lg.tfd


class PosteriorTestCase(unittest.TestCase):
    def test_trainable(self):

        @hk.transform_with_state
        def trainable_normal():
            return lg.NormalPosterior(prior=tfd.Normal(loc=0., scale=2.),name="test",num_kl=2)()

        rng = jax.random.PRNGKey(42)
        p,s = trainable_normal.init(rng)
        trainable_normal.apply(p,s,rng)

    def test_trainable_plus(self):

        @hk.transform_with_state
        def trainable_normal():
            return lg.SofplusNormalPosterior(prior=tfd.LogNormal(loc=0., scale=2.),name="test", num_kl=2)()

        rng = jax.random.PRNGKey(42)
        p,s = trainable_normal.init(rng)
        trainable_normal.apply(p,s,rng)

class IPPTestCase(unittest.TestCase):
    def test_vmap(self):
        dist = lg.InhomogeneousPoissonProcess(
            maximum=jnp.array(2*[4.]),
            rates=jnp.ones((2,2)),
            midpoints= jnp.ones((2, 2)),
            mix = jnp.zeros((2,2))
        )
        t = jnp.array([1, 2, 3])

        dist.log_prob(t)

    # def test_vmap_distrax(self):
    #     d = distrax.Normal(loc=jnp.array([0.,1,]), scale=jnp.array([2.,1,]))
    #     #d = tfd.Normal(loc=jnp.array([0.,1,]), scale=jnp.array([2.,1,]))
    #     d = distrax.Normal(loc=jnp.array([0. ]), scale=jnp.array([2.]))
    #
    #     x = jnp.array([1.,2,3])
    #
    #     def _lp(dist, x):
    #         return dist.log_prob(x)
    #     lp = jax.vmap(_lp, in_axes=(0,None),out_axes=(2))
    #
    #     lp(d,x)
    #
    #     class TMP(NamedTuple):
    #         a:jnp.ndarray
    #
    #         @functools.partial(jax.vmap,in_axes=(0,None))
    #         def f(self,x):
    #             #return self.a+x
    #             return distrax.Normal(loc=self.a, scale=2.).log_prob(x)
    #
    #     tmp=TMP(a=jnp.array([1,2]))
    #
    #     tmp.f(jnp.array([1,2,3]))

class LogisticGrowthTestCase(unittest.TestCase):
    def test_model(self):

        @hk.transform_with_state
        def model():
            m = lg.LogisticGrowthSuperposition()
            return m()

        rng = jax.random.PRNGKey(42)
        params, state = model.init(rng)
        dist, state = model.apply(params, state,rng)
        t = jnp.array([1, 2, 3.])
        lp=dist.log_prob(t)
        ...


if __name__ == '__main__':
    unittest.main()
