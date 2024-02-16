#!/usr/bin/env python3

import jax
import jaxtyping
import equinox as eqx
from jax.nn import relu
import jax.numpy as jnp
from equinox.nn import Linear

from nerf.primitives.encoding import PositionalEncoding


class ImageFuncMLP(eqx.Module):
    layers: jax.Array

    def __init__(self, key: jaxtyping.PRNGKeyArray, pos_dim=40):
        keys = jax.random.split(key, 7)
        self.layers: jax.Array = [
            Linear(pos_dim, 256, key=keys[0]),
            relu,
            Linear(256, 256, key=keys[1]),
            relu,
            Linear(256, 256, key=keys[2]),
            relu,
            Linear(256, 256, key=keys[3]),
            relu,
            Linear(256, 256, key=keys[4]),
            relu,
            Linear(256, 256, key=keys[5]),
            relu,
            Linear(256, 3, key=keys[6]),
        ]

    def __call__(self, xyz: jax.Array) -> jax.Array:
        x = xyz
        for layer in self.layers:
            x = layer(x)
        return x


class BasicNeRF(eqx.Module):
    layers: jax.Array

    def __init__(self, key: jaxtyping.PRNGKeyArray, encoding_scale, pos_dim=60):
        keys = jax.random.split(key, 5)
        self.layers: jax.Array = [
            PositionalEncoding(pos_dim, encoding_scale),
            Linear(pos_dim, 256, key=keys[0]),
            relu,
            Linear(256, 256, key=keys[1]),
            relu,
            Linear(256, 256, key=keys[2]),
            relu,
            Linear(256, 128, key=keys[3]),
            relu,
            Linear(128, 4, key=keys[4]),
        ]

    def __call__(self, xyz: jax.Array, _: jax.Array) -> jax.Array:
        x = xyz
        for layer in self.layers:
            x = layer(x)

        density = x[0]
        rgb = x[1:]

        return density, rgb


class MhallMLP(eqx.Module):
    layers_first_half: jax.Array
    layers_second_half: jax.Array
    rgb_head: jax.Array

    def __init__(
        self,
        key: jaxtyping.PRNGKeyArray,
        pos_dim=60,
        dir_dim=24,
        pos_encoding_scale=1.0,
    ):
        keys = jax.random.split(key, 10)
        self.layers_first_half: jax.Array = [
            PositionalEncoding(pos_dim, pos_encoding_scale),
            Linear(pos_dim, 256, key=keys[0]),
            relu,
            Linear(256, 256, key=keys[1]),
            relu,
            Linear(256, 256, key=keys[2]),
            relu,
            Linear(256, 256, key=keys[3]),
            relu,
        ]

        self.layers_second_half = [
            Linear(256 + pos_dim, 256, key=keys[4]),
            relu,
            Linear(256, 256, key=keys[5]),
            relu,
            Linear(256, 256, key=keys[6]),
            relu,
            Linear(256, 256, key=keys[7]),
            relu,
            Linear(256, 256 + 1, key=keys[8]),
        ]

        self.rgb_head = [
            Linear(256 + dir_dim, 128, key=keys[9]),
            relu,
            Linear(128, 3, key=keys[10]),
        ]

    def __call__(self, xyz: jax.Array, view_dir: jax.Array) -> jax.Array:
        x = xyz

        for layer in self.layers_first_half:
            x = layer(x)

        x = jnp.concatenate([x, xyz])

        for layer in self.layers_second_half:
            x = layer(x)

        density = x[0]

        x = x[1:]
        x = jnp.concatenate([x, view_dir])

        for layer in self.rgb_head:
            x = layer(x)
        rgb = x

        return density, rgb


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    mlp = MhallMLP(key)
    grad_mlp = jax.grad(mlp)
    pos = jax.random.normal(key, (60,))
    dirs = jax.random.normal(key, (24,))
    vpos = jax.random.normal(key, (11, 60))
    vdirs = jax.random.normal(key, (11, 24))
    print(mlp)
    print(mlp(pos, dirs))
    print(jax.vmap(mlp, (vpos, vdirs)))
