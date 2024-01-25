#!/usr/bin/env python3

import jax
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

    def __init__(self, f, h, w, pose, near=2.0, far=6.0, c=(0.0, 0.0)):
        self.f = f
        self.h = h
        self.w = w
        self.pose = pose
        self.near = near
        self.far = far
        self.c = c

    @jax.jit
    def get_ray(self, u: Float, v: Float) -> Ray:
        # should return centers and directions for the specified pixel
        o = jnp.array([0.0, 0.0, 0.0])
        d = jnp.array(
            [
                (u - 0.5 - self.w / 2.0) / self.f,
                -(v - 0.5 - self.h / 2.0) / self.f,
                -1.0,
            ]
        )

        d = self.pose.rotation() @ d
        o = self.pose.translation() + o

        tn = -(self.near + o[2]) / d[2]
        o = o + tn * d

        # o_prime = jnp.array([
        #     -(self.f * o[0])/((self.w/2)*o[2]),
        #     -(self.f * o[1])/((self.h/2)*o[2]),
        #     1.0+(2.0*self.near)/(o[2]),
        # ])

        # d_prime = jnp.array([
        #     - (self.f / (self.w/2)) * ((d[0]/d[2]) - (o[0]/o[2])),
        #     - (self.f / (self.h/2)) * ((d[1]/d[2]) - (o[1]/o[2])),
        #     - (2.0*self.near)/(o[2]),
        # ])

        return Ray(o, d)

    def get_ray_exp(self, u, v):
        o = jnp.array([0.0, 0.0, 0.0])
        d = jnp.array(
            [
                (u + 0.5 - self.w / 2.0)
                / self.f,  # TODO make sure that +0.5 is correct
                -(v + 0.5 - self.h / 2.0) / self.f,
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
            eqx.filter_vmap(self.get_ray, in_axes=(0, 0)), in_axes=(0, 0)
        )
        return mapped(us, vs)

    def get_rays_experimental(self):
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
    ray = c.get_ray_exp(50, 50)
    print(f"{ray.origin=}")
    print(f"{ray.direction=}")
    print(f"{ray(0.0)=}")
    print(f"{ray(0.25)=}")
    print(f"{ray(0.5)=}")
    print(f"{ray(0.75)=}")
    print(f"{ray(1.0)=}")
    breakpoint()
