#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class Dataloader:
    def __init__(self, key, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key
        self.W = dataset.W
        self.H = dataset.H

    def __len__(self):
        return len(self.dataset)

    def get_batch(self):
        self.key, key_b, key_u, key_v = jax.random.split(self.key, 4)
        batch_idx = jax.random.choice(key_b, len(self.dataset), (self.batch_size,))
        us = jax.vmap(lambda x, k: jax.random.uniform(k) * x)(
            self.W[batch_idx],
            jax.random.split(key_u, self.batch_size),
        )

        vs = jax.vmap(lambda x, k: jax.random.uniform(k) * x)(
            self.H[batch_idx],
            jax.random.split(key_v, self.batch_size),
        )
        us = jnp.int32(us)
        vs = jnp.int32(vs)

        rgb_ground_truths = jax.vmap(
            lambda images, batch_id, u, v: images[batch_id, v, u],
            in_axes=[None, 0, 0, 0],
        )(self.dataset.images, batch_idx, us, vs)
        cameras = jax.tree_map(lambda x: x[batch_idx], self.dataset.cameras)
        rays = jax.vmap(lambda x, u, v: x.get_ray(u, v))(cameras, us, vs)
        return rgb_ground_truths, rays
