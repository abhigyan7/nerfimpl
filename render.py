#!/usr/bin/env python3

import jax
import jax.numpy as jnp

import equinox as eqx

class Integrator(eqx.Module):

    def setup(self):
        pass

    def __call__(self, points, samples):
        return

class HierarchicalSampler(eqx.Module):
    def setup(self, near, far, nc, nf):
        self.near = near
        self.far = far
        self.nc = nc
        self.nf = nf

    def get_coarse_points(self):
        # first sample set
        course_bins = jnp.arange(self.nc+1)
        course_bins = course_bins / self.nc
        self.course_bins = self.near + course_bins * (self.far - self.near)

        # TODO sample course_points from above
        self.deltas_course = None
        course_points = None
        return course_points

    def get_fine_points(self, coarse_points, coarse_samples):
        # second sample set
        course_bins = jnp.arange(self.nf)
        course_bins = course_bins / self.nc
        self.course_bins = self.near + course_bins * (self.far - self.near)
        # TODO sample course_points from above
        self.deltas_fine = None
        self.fine_points = NotImplemented
        return self.fine_points
        return

