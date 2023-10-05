#!/usr/bin/env python3

import jax.numpy as jnp
import equinox as eqx

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
        o = jnp.array([0, 0, 0.0])
        d = jnp.array([u, v, 1.0])

        o = self.pose.translation() + o
        d = self.pose @ d

        tn = -(self.n+o[2]) / d[2]
        on = o + tn * d

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

