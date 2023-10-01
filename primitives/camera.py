#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx

class PinholeCamera(eqx.Module):
    f: jax.Array
    d: jax.Array
    c: jax.Array

    def __init__(self, f, d, c=(0.0, 0.0)):
        self.f = f
        self.d = d
        self.c = c

    def get_ray(self, p):
        # should return centers and directions for the specified pixel
        raise NotImplementedError

    def get_ray_bundle(self):
        # should return center and directions for all rays coming out of the camera
        pixels = None # get all pixels
        # can do this? return jax.vmap(self.get_ray, pixels)
        raise NotImplementedError
