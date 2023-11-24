#!/usr/bin/env python3

from functools import partial
import jax.numpy as jnp
import equinox as eqx
import jax

from jaxlie import SE3

from jaxtyping import Float, Array

class Ray(eqx.Module):
    origin: Float[Array, "3"]
    direction: Float[Array, "3"]

    def __call__(self, t):
        return self.origin + self.direction * t

class PinholeCamera(eqx.Module):
    f: Float
    w: Float
    h: Float
    c: Float[Array, "2"]
    n: Float
    pose: SE3

    def __init__(self, f, h, w, pose, n, c=(0.0, 0.0)):
        self.f = f
        self.h = h
        self.w = w
        self.pose = pose
        self.n = n
        self.c = c

    def get_ray(self, u: Float, v: Float) -> Ray:
        # should return centers and directions for the specified pixel
        o = jnp.array([0.0, 0.0, 0.0])
        d = jnp.array([u-self.w/(2*self.f), v-self.h/(2*self.f), -1.0])

        on = self.pose.translation() + o
        d = self.pose.rotation().inverse() @ d

        tn = -(self.n+o[2]) / d[2]
        on = on + tn * d

        o_prime = jnp.array([
            -(self.f * on[0])/((self.w/2)*on[2]),
            -(self.f * on[1])/((self.h/2)*on[2]),
            1+(2*self.n)/(on[2]),
        ])

        d_prime = jnp.array([
            - (self.f / (self.w/2)) * ((d[0]/d[2]) - (on[0]/on[2])),
            - (self.f / (self.h/2)) * ((d[1]/d[2]) - (on[1]/on[2])),
            - (2*self.n)/(on[2]),
        ])

        return Ray(o_prime, d_prime)

    def get_rays(self):
        us = jnp.arange(self.w)
        vs = jnp.arange(self.h)
        us, vs = jnp.meshgrid(us, vs)
        mapped = eqx.filter_vmap(
            eqx.filter_vmap(self.get_ray, in_axes=(0, 0)),
            in_axes=(0, 0))
        return mapped(us, vs)
