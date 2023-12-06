#!/usr/bin/env python3

import tqdm
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from tensorboardX import SummaryWriter

from nerf.primitives.mlp import MhallMLP
from nerf import utils
from nerf.render import render_frame, hierarchical_render_single_ray
from nerf.datasets.nerfdata import Dataloader
from nerf.datasets.blender import BlenderDataset


@eqx.filter_jit
def optimize_one_batch(nerfs, rays, rgb_ground_truths, key, optimizer, optimizer_state):
    @eqx.filter_value_and_grad
    def loss_fn(nerfs, rays, rgb_ground_truths, key):
        keys = jax.random.split(key, rays.origin.shape[0])
        coarse_rgbs, fine_rgbs = eqx.filter_vmap(
            hierarchical_render_single_ray, in_axes=(0, 0, None, None)
        )(keys, rays, nerfs, True)
        loss = jnp.mean((fine_rgbs - rgb_ground_truths) ** 2.0) + jnp.mean(
            (coarse_rgbs - rgb_ground_truths) ** 2.0
        )
        return loss

    loss, grad = loss_fn(nerfs, rays, rgb_ground_truths, key)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, nerfs)
    nerfs = eqx.apply_updates(nerfs, updates)

    return nerfs, optimizer_state, loss


def main(conf):
    key = jax.random.PRNGKey(conf.seed)
    coarse_nerf_key, fine_nerf_key, dataloader_key, sampler_key = jax.random.split(
        key, 4
    )

    writer = SummaryWriter()

    coarse_nerf = MhallMLP(coarse_nerf_key)
    fine_nerf = MhallMLP(fine_nerf_key)
    nerfs = (coarse_nerf, fine_nerf)

    nerfdataset = BlenderDataset(conf.dataset_path, "transforms_train.json", conf.scale)
    nerfdataset_test = nerfdataset
    dataloader = Dataloader(dataloader_key, nerfdataset, conf.batch_size)

    gt_idx = 23
    ground_truth_image = nerfdataset_test.images[gt_idx]
    camera = jax.tree_map(lambda x: x[gt_idx], nerfdataset_test.cameras)

    image = utils.jax_to_PIL(ground_truth_image)
    writer.add_image("ground-truth", np.array(image), 0, dataformats="HWC")
    image.save(f"runs/gt.png")

    optimizer = optax.adam(5e-4)
    optimizer_state = optimizer.init(eqx.filter(nerfs, eqx.is_array))

    psnr = 0.0

    for step in (pbar := tqdm.trange(1000000)):
        rgb_ground_truths, rays = dataloader.get_batch()

        if step % 200 == 0:

            img = render_frame(nerfs, camera, key, conf.chunk_size)

            image = utils.jax_to_PIL(img)
            image.save(f"runs/output_{step:09}.png")
            psnr = utils.PSNR(ground_truth_image, img)

            writer.add_scalar("psnr", psnr, step)
            writer.add_image("render", np.array(image), step, dataformats="HWC")

        key, sampler_key = jax.random.split(key)
        nerfs, optimizer_state, loss = optimize_one_batch(
            nerfs, rays, rgb_ground_truths, sampler_key, optimizer, optimizer_state
        )

        pbar.set_description(f"Loss={loss.item():.4f}, utils.PSNR={psnr:.4f}")
        writer.add_scalar("loss", loss.item(), step)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--scale", type=float, default=1.0)
    conf = parser.parse_args()
    main(conf)
