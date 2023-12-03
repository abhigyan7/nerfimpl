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

from nerf.primitives.mlp import MhallMLP
from nerf.utils import PSNR
from nerf.render import render_frame, hierarchical_render_single_ray
from nerf.datasets.nerfdata import Dataloader
from nerf.datasets.blender import BlenderDataset


dataset_path = "/media/data/lego-20231005T103337Z-001/lego/"

@eqx.filter_jit
def optimize_one_batch(nerf, rays, rgb_ground_truths, key, optimizer, optimizer_state):
    @eqx.filter_value_and_grad
    def loss_fn(nerf, rays, rgb_ground_truths, key):
        keys = jax.random.split(key, rays.origin.shape[0])
        _, fine_rgbs = eqx.filter_vmap(
            hierarchical_render_single_ray, in_axes=(0, 0, None, None)
        )(keys, rays, nerf, True)
        loss = jnp.mean((fine_rgbs - rgb_ground_truths) ** 2.0)
        return loss

    loss, grad = loss_fn(nerf, rays, rgb_ground_truths, key)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, nerf)
    nerf = eqx.apply_updates(nerf, updates)

    return nerf, optimizer_state, loss


def main(conf):
    key = jax.random.PRNGKey(conf.seed)
    nerf_key, dataloader_key, sampler_key = jax.random.split(key, 3)

    nerf = MhallMLP(nerf_key)

    nerfdataset = BlenderDataset(conf.dataset_path, "transforms_train.json", conf.scale)
    nerfdataset_test = nerfdataset
    dataloader = Dataloader(dataloader_key, nerfdataset, conf.batch_size)

    gt_idx = 23
    ground_truth_image = nerfdataset_test.images[gt_idx]
    camera = jax.tree_map(lambda x: x[gt_idx], nerfdataset_test.cameras)

    img = np.uint8(np.array(ground_truth_image) * 255.0)
    image = Image.fromarray(img)
    image.save(f"runs/gt.png")

    optimizer = optax.adam(5e-4)
    optimizer_state = optimizer.init(eqx.filter(nerf, eqx.is_array))

    psnr = -1.0

    for step in (pbar := tqdm.trange(1000000)):
        rgb_ground_truths, rays = dataloader.get_batch()

        if step % 2000 == 0 and step != 0:
            img = render_frame(nerf, camera, key)

            image = np.array(img)
            image = np.uint8(np.clip(image * 255.0, 0, 255))
            image = Image.fromarray(image)
            image.save(f"runs/output_{step:09}.png")
            psnr = PSNR(ground_truth_image, img)

        key, sampler_key = jax.random.split(key)
        nerf, optimizer_state, loss = optimize_one_batch(
            nerf, rays, rgb_ground_truths, sampler_key, optimizer, optimizer_state
        )

        pbar.set_description(f"Loss={loss.item():.4f}, PSNR={psnr:.4f}")
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
