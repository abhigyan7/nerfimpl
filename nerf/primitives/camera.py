#!/usr/bin/env python3

import equinox as eqx
from jaxlie import SE3, SO3
import jax.numpy as jnp
from jaxtyping import Float, Array


def inverse_transform(t):
    return t / (1.0 - t)


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
    near: Float
    far: Float
    pose: SE3

    def __init__(self, f, h, w, pose, c, near=1.0, far=2.0):
        self.f = f
        self.h = h
        self.w = w
        self.pose = pose
        self.near = near
        self.far = far
        self.c = c

    def get_ray(self, u, v):
        o = jnp.array([0.0, 0.0, 0.0])
        d = jnp.array(
            [
                (u - 0.5 - c[0]) / self.f,
                -(v - 0.5 - c[1]) / self.f,
                -1.0,
            ]
        )

        tn = -(self.near + o[2]) / d[2]
        o = o + tn * d

        distance = jnp.abs(self.near - self.far)
        d = distance * d / jnp.abs(d[2])

        d = self.pose.rotation() @ d
        o = self.pose @ o

        return Ray(o, d)

    def get_rays(self):
        us = jnp.arange(self.w)
        vs = jnp.arange(self.h)
        us, vs = jnp.meshgrid(us, vs)
        mapped = eqx.filter_vmap(
            eqx.filter_vmap(self.get_ray_exp, in_axes=(0, 0)), in_axes=(0, 0)
        )
        return mapped(us, vs)


if __name__ == "__main__":
    pose = SO3.from_matrix(jnp.eye(3))
    pose = SE3.from_rotation_and_translation(pose, jnp.zeros((3,)))

    camera = PinholeCamera(100.0, 99.0, 99.0, pose)

    rays = camera.get_rays()
    c = camera
    ray = c.get_ray(0, 0)
    print(f"{ray.origin=}")
    print(f"{ray.direction=}")
    print(f"{ray(0.0)=}")
    print(f"{ray(1.0)=}")
    ray = c.get_ray(50, 50)
    print(f"{ray.origin=}")
    print(f"{ray.direction=}")
    print(f"{ray(0.0)=}")
    print(f"{ray(0.25)=}")
    print(f"{ray(0.5)=}")
    print(f"{ray(0.75)=}")
    print(f"{ray(1.0)=}")
    breakpoint()
