#!/usr/bin/env python3

import abc

from flax import linen
import jax
import jax.numpy as jnp

class SceneRepresentation(abc.ABC):

    def __init__(self):
        pass

    def forward(self):
        pass

class MhallMLP(linen.Module):

    def setup(self):
        self.dense1 = linen.Dense(256)
        self.dense2 = linen.Dense(256)
        self.dense3 = linen.Dense(256)
        self.dense4 = linen.Dense(256)
        self.dense5 = linen.Dense(256)
        self.dense6 = linen.Dense(256)
        self.dense7 = linen.Dense(256)
        self.dense8 = linen.Dense(257)
        self.dense9 = linen.Dense(128)
        self.dense10 = linen.Dense(3)

    def __call__(self, x, d):

        x = linen.relu(self.dense1(x))
        x = linen.relu(self.dense2(x))
        x = linen.relu(self.dense3(x))
        x = linen.relu(self.dense4(x))

        x = jnp.concatenate([x, d], axis=1)
        x = linen.relu(self.dense5(x))
        x = linen.relu(self.dense6(x))
        x = linen.relu(self.dense7(x))
        x = linen.relu(self.dense8(x))

        density = x[..., 0]

        x = linen.relu(self.dense9(x[..., 1:]))
        rgb = linen.sigmoid(self.dense10(x))

        return density, rgb

if __name__ == "__main__":
    mlp = MhallMLP()
    print(mlp.tabulate(jax.random.key(0), jnp.ones((10, 3)), jnp.ones((10, 2))))
