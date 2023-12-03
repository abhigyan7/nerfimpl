#!/usr/bin/env python3

from primitives import mlp, render
from data.nerfdata import NerfDataloader, NerfDataset
import jax.numpy as jnp
from PIL import Image
import numpy as np
import equinox as eqx
import optax
import tqdm
import jax

from pathlib import Path

dataset_path = "/media/data/lego-20231005T103337Z-001/lego/"

BATCH_SIZE = 256
SCALE = 1.0

@eqx.filter_jit
def render_line(nerf, rays, key):

    keys = jax.random.split(key, rays.origin.shape[0])
    _, fine_rgbs = eqx.filter_vmap(
                render.hierarchical_render_single_ray,
                in_axes=(0, 0, None, None)) (keys, rays, nerf, False)
    return fine_rgbs


def render_frame(nerf, camera, key, n_rays_per_chunk=400):
    rays = camera.get_rays()
    rays_orig_shape = rays.origin.shape
    total_n_rays = rays_orig_shape[0] * rays_orig_shape[1]
    n_chunks = int(total_n_rays / n_rays_per_chunk)
    assert n_chunks * n_rays_per_chunk == total_n_rays
    keys = jax.random.split(key, n_chunks)

    rays = jax.tree_map(
        lambda x: x.reshape((n_chunks, n_rays_per_chunk, 3)),
        rays
    )

    fine_rgbs = jax.lax.map(
            lambda ray_key : render_line(nerf, ray_key[0], ray_key[1]), (rays, keys))

    fine_rgbs = jax.tree_map(
        lambda x: x.reshape((*rays_orig_shape[:-1], 3)),
        fine_rgbs
    )

    return fine_rgbs


@eqx.filter_jit
def optimize_one_batch(nerf, rays, rgb_ground_truths, key, optimizer, optimizer_state):

    @eqx.filter_value_and_grad
    def loss_fn(nerf, rays, rgb_ground_truths, key):
        keys = jax.random.split(key, rays.origin.shape[0])
        _, fine_rgbs = eqx.filter_vmap(
            render.hierarchical_render_single_ray,
            in_axes=(0, 0, None, None)
        ) (keys, rays, nerf, True)
        loss = jnp.mean((fine_rgbs - rgb_ground_truths)**2.0)
        return loss

    loss, grad = loss_fn(nerf, rays, rgb_ground_truths, key)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, nerf)
    nerf = eqx.apply_updates(nerf, updates)

    return nerf, optimizer_state, loss


def PSNR(ground_truth, pred, max_intensity=1.0):
    MSE = jnp.mean((ground_truth - pred) ** 2.0)
    psnr = 20 * jnp.log10(max_intensity) - 10 * jnp.log10(MSE)
    return psnr

def main():

    key = jax.random.PRNGKey(42)
    nerf_key, dataloader_key, sampler_key = jax.random.split(key, 3)

    nerf = mlp.MhallMLP(nerf_key)

    nerfdataset = NerfDataset(Path(dataset_path), "transforms_train.json", SCALE)
    #nerfdataset_test = NerfDataset(Path(dataset_path), "transforms_test.json", SCALE)
    nerfdataset_test = nerfdataset
    dataloader = NerfDataloader(dataloader_key, nerfdataset, BATCH_SIZE)

    gt_idx = 23
    ground_truth_image = nerfdataset_test.images[gt_idx]
    camera = jax.tree_map(lambda x: x[gt_idx], nerfdataset_test.cameras)

    img = np.uint8(np.array(ground_truth_image)*255.0)
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
            image = np.uint8(np.clip(image*255.0, 0, 255))
            image = Image.fromarray(image)
            image.save(f"runs/output_{step:09}.png")
            psnr = PSNR(ground_truth_image, img)

        key, sampler_key = jax.random.split(key)
        nerf, optimizer_state, loss = optimize_one_batch(
            nerf, rays, rgb_ground_truths, sampler_key, optimizer, optimizer_state)

        pbar.set_description(f"Loss={loss.item():.4f}, PSNR={psnr:.4f}")
    return

if __name__ == "__main__":
    main()
