#!/usr/bin/env python3

import os
import click
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from datetime import datetime

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from nerf.primitives.mlp import MhallMLP
from nerf.datasets.nerfdata import Dataloader
from nerf.datasets.blender import BlenderDataset
from nerf.render import (
    render_frame,
    hierarchical_render_single_ray,
    sample_coarse,
    sample_fine,
    inverse_transform,
)
from nerf.utils import timing, serialize, deserialize, jax_to_PIL, PSNR, mse_to_psnr


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dataset-path", type=Path, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--scale", type=float, default=1.0)
@click.option("--t-sampling", type=str, default="linear")
@click.option("--num-fine-samples", type=int, default=16)
@click.option("--num-coarse-samples", type=int, default=16)
@timing
def test(**conf):
    key = jax.random.PRNGKey(conf["seed"])
    dl_key, loc_key, sampler_key = jax.random.split(key, 3)

    nerfdataset = BlenderDataset(
        conf["dataset_path"], "transforms_train.json", conf["scale"], 64
    )

    batch_size = 256
    dataloader = Dataloader(dl_key, nerfdataset, batch_size)

    camera_centers = nerfdataset.cameras.pose.translation()

    first_camera = jax.tree_map(lambda x: x[0], nerfdataset.cameras)
    rays = first_camera.get_rays_experimental()
    _, rays = dataloader.get_batch()
    rays = jax.tree_map(lambda x: x.reshape(-1, 3), rays)
    keys = jax.random.split(loc_key, rays.origin.shape[0])

    def get_samples_for_ray(ray, key):
        ts = sample_coarse(key, conf["num_coarse_samples"])
        if conf["t_sampling"] == "inverse":
            ts = inverse_transform(ts)
        xyzs = eqx.filter_vmap(ray)(ts)
        return xyzs

    xyzs = jax.vmap(get_samples_for_ray)(rays, keys)
    xyzs = xyzs.reshape(-1, 3)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection="3d")

    xs = xyzs[:, 0]
    ys = xyzs[:, 1]
    zs = xyzs[:, 2]
    ax.scatter(xs, ys, zs, c="r")

    xs = camera_centers[:, 0]
    ys = camera_centers[:, 1]
    zs = camera_centers[:, 2]
    ax.scatter(xs, ys, zs)

    xs = rays.origin[:, 0]
    ys = rays.origin[:, 1]
    zs = rays.origin[:, 2]
    ax.scatter(xs, ys, zs)

    xs = xs + rays.direction[:, 0]
    ys = ys + rays.direction[:, 1]
    zs = zs + rays.direction[:, 2]
    ax.scatter(xs, ys, zs, c=np.arange(xs.shape[0]))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    xs = rays.origin[:, 0]
    ys = rays.origin[:, 1]
    zs = rays.origin[:, 2]

    xd = rays.direction[:, 0]
    yd = rays.direction[:, 1]
    zd = rays.direction[:, 2]

    print(f"{rays.origin.shape=}")

    # ax.quiver(xs, ys, zs, xd, yd, zd, length=1.0)

    plt.show()

    print(xyzs.shape)
    print("Test done!")
    return


if __name__ == "__main__":
    cli()
