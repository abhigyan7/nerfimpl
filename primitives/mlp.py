#!/usr/bin/env python3

import jax
import jax.numpy as jnp

from jax.nn import relu, sigmoid
import equinox as eqx
from equinox.nn import Linear

import jaxtyping


class MhallMLP(eqx.Module):
    layers_first_half: jax.Array
    layers_second_half: jax.Array
    rgb_head: jax.Array

    def __init__(self, key : jaxtyping.PRNGKeyArray, pos_dim=60, dir_dim=24):
        keys = jax.random.split(key, 10)
        self.layers_first_half : jax.Array = [
            Linear(pos_dim, 256, key=keys[0]), relu,
            Linear(256, 256, key=keys[1]), relu,
            Linear(256, 256, key=keys[2]), relu,
            Linear(256, 256, key=keys[3]), relu,
        ]

        self.layers_second_half = [
            Linear(256+pos_dim, 256, key=keys[4]), relu,
            Linear(256, 256, key=keys[5]), relu,
            Linear(256, 256, key=keys[6]), relu,
            Linear(256, 256, key=keys[7]), relu,
            Linear(256, 256+1, key=keys[8]),
        ]

        self.rgb_head = [
            Linear(256+dir_dim, 128, key=keys[9]), relu,
            Linear(128, 3, key=keys[10]), sigmoid,
        ]


    def __call__(self, xyz: jax.Array, dirs: jax.Array) -> jax.Array:

        x = xyz

        for layer in self.layers_first_half:
            x = layer(x)

        x = jnp.concatenate([x, xyz], axis=1)

        for layer in self.layers_second_half:
            x = layer(x)

        density = jax.nn.relu(x[..., 0])

        x = x[..., 1:]
        x = jnp.concatenate([x, dirs], axis=1)

        for layer in self.rgb_head:
            x = layer(x)
        rgb = x

        return density, rgb

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    mlp = MhallMLP(key)
    grad_mlp = jax.grad(mlp)
    print(mlp)
