#!/usr/bin/env python3

from primitives import camera, mlp, render
from data.nerfdata import NerfDataloader, NerfDataset
import jax.numpy as jnp
import equinox as eqx
import optax
import jax

from pathlib import Path

dataset_path = "/media/data/lego-20231005T103337Z-001/lego/"

BATCH_SIZE = 128


@eqx.filter_jit
def optimize_one_batch(nerf, rays, rgb_ground_truths, key, optimizer, optimizer_state):

        @eqx.filter_value_and_grad
        def loss_fn(nerf, rays, rgb_ground_truths, key):
            keys = jax.random.split(key, rays.origin.shape[0])
            (coarse_rgbs, fine_rgbs) = eqx.filter_vmap(
                render.hierarchical_render_single_ray,
                in_axes=(0, 0, None, None)
            ) (keys, rays, nerf, True)
            coarse_loss = jnp.mean((coarse_rgbs - rgb_ground_truths)**2.0)
            fine_loss = jnp.mean((fine_rgbs - rgb_ground_truths)**2.0)
            return coarse_loss + fine_loss

        loss, grad = loss_fn(nerf, rays, rgb_ground_truths, key)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        nerf = eqx.apply_updates(nerf, updates)

        return nerf, optimizer_state, loss


def main():

    key = jax.random.PRNGKey(0)
    nerf_key, dataloader_key, sampler_key = jax.random.split(key, 3)

    nerf = mlp.MhallMLP(nerf_key)

    nerfdataset = NerfDataset(Path(dataset_path))
    dataloader = NerfDataloader(dataloader_key, nerfdataset, BATCH_SIZE)

    optimizer = optax.adabelief(3e-4)
    optimizer_state = optimizer.init(eqx.filter(nerf, eqx.is_inexact_array))

    for step, (rgb_ground_truths, rays) in zip(range(100), dataloader):
        key, sampler_key = jax.random.split(sampler_key)
        nerf, optimizer_state, loss = optimize_one_batch(nerf, rays, rgb_ground_truths, key, optimizer, optimizer_state)
        print(f"{step=}: {loss=}")
    return

if __name__ == "__main__":
    main()
