#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import flax
from flax import linen

class HierarchicalSampling(linen.Module):
    def setup(self, near, far, nc, nf):
        self.near = near
        self.far = far
        self.nc = nc
        self.nf = nf


    def get_coarse_points(self):
        # first sample set
        course_bins = jnp.arange(nc+1)
        course_bins = course_bins / nc
        self.course_bins = self.near + course_bins * (self.far - self.near)
        # TODO sample course_points from above
        self.deltas_course = None
        return course_points

    def get_find_points(self, coarse_points, coarse_samples):
        # second sample set
        course_bins = jnp.arange(nf)
        course_bins = course_bins / nc
        self.course_bins = self.near + course_bins * (self.far - self.near)
        self.deltas_course = jnp.ones(nc) / nc
        # TODO sample course_points from above
        return course_points
        return

    def __call__(self, density_function):
        pass
