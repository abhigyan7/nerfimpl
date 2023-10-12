#!/usr/bin/env python3

from primitives import camera, mlp, render
from data.nerfdata import NerfDataloader, NerfDataset
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import jaxlie
import jax

from pathlib import Path

dataset_path = "/media/data/lego-20231005T103337Z-001/lego/"

BATCH_SIZE = 32

def main():
    nerfdataset = NerfDataset(Path(dataset_path))
    key = jax.random.PRNGKey(0)
    nerf_key, sampler_key = jax.random.split(key, 2)
    pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0, 0.1, 0),
        jnp.array([0,0,0]),
    )
    nerf = mlp.MhallMLP(nerf_key)


    dataloader_key, sampler_key = jax.random.split(sampler_key, 2)
    dataloader = NerfDataloader(dataloader_key, nerfdataset, BATCH_SIZE)
    rgb_ground_truths, rays = next(dataloader)

    sampler_keys = jax.random.split(sampler_key, BATCH_SIZE)
    coarse_rgbs, fine_rgbs = eqx.filter_vmap(
        render.hierarchical_render_single_ray,
        in_axes=(0, 0, None)
    ) (sampler_keys, rays, nerf)

    diffs = jnp.abs(fine_rgbs - rgb_ground_truths)

    print(f"{diffs=}")
    print(f"{jnp.mean(diffs)=}")
    print(f"{coarse_rgbs=}")
    print(f"{fine_rgbs=}")

    return

if __name__ == "__main__":
    main()
