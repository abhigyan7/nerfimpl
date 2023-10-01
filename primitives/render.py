#!/usr/bin/env python3

import jax
import jax.numpy as jnp

from jaxtyping import Float, Int

import equinox as eqx

class Integrator(eqx.Module):

    def __init__(self):
        pass

    def __call__(self, points, samples):
        return

class HierarchicalSampler():
    nc: Int
    nf: Int

    def __init__(self, key, nc, nf):
        self.nc = nc
        self.nf = nf
        self.key = key

    def get_coarse_points(self):
        self.key, new_key = jax.random.split(self.key)
        coarse_points = jnp.arange(self.nc) + jax.random.uniform(new_key, (self.nc,))
        coarse_points = coarse_points / self.nc
        return coarse_points

    def get_fine_points(self, coarse_t, coarse_p):
        coarse_pmf = coarse_p / coarse_p.sum()
        coarse_cdf = jnp.cumsum(coarse_p)

        self.key, new_key = jax.random.split(self.key)
        uniform = jax.random.uniform(new_key, (self.nf,))


        coarse_bins = jnp.arange(self.nc)

        self.coarse_bins = self.near + coarse_bins * (self.far - self.near)
        # TODO sample course_points from above
        self.deltas_fine = None
        self.fine_points = NotImplemented
        return self.fine_points
