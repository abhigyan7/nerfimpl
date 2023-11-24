#!/usr/bin/env python3

from primitives import mlp, render
from data.nerfdata import NerfDataloader, NerfDataset
from primitives.camera import PinholeCamera
import jax.numpy as jnp
from PIL import Image
import numpy as np
import equinox as eqx
import jaxlie
import optax
import tqdm
import jax

from pathlib import Path

dataset_path = "/media/data/lego-20231005T103337Z-001/lego/"

BATCH_SIZE = 32

@eqx.filter_jit
def render_line(nerf, rays, key):

    keys = jax.random.split(key, rays.origin.shape[0])
    _, fine_rgbs = eqx.filter_vmap(
                render.hierarchical_render_single_ray,
                in_axes=(0, 0, None, None)) (keys, rays, nerf, False)
    return fine_rgbs


#@eqx.filter_jit
def render_frame(nerf, camera, key):
    rays = camera.get_rays()
    keys = jax.random.split(key, rays.origin.shape[0])
    fine_rgbs = jax.lax.map(
            lambda ray_key : render_line(nerf, ray_key[0], ray_key[1]), (rays, keys))
    return fine_rgbs

@eqx.filter_jit
def optimize_one_batch(nerf, rays, rgb_ground_truths, key, optimizer, optimizer_state):

    @eqx.filter_value_and_grad
    def loss_fn(nerf, rays, rgb_ground_truths, key):
        keys = jax.random.split(key, rays.origin.shape[0])
        coarse_rgbs, fine_rgbs = eqx.filter_vmap(
            render.hierarchical_render_single_ray,
            in_axes=(0, 0, None, None)
        ) (keys, rays, nerf, True)
        coarse_loss = jnp.mean((coarse_rgbs - rgb_ground_truths)**2.0)
        fine_loss = jnp.mean((fine_rgbs - rgb_ground_truths)**2.0)
        return coarse_loss + fine_loss

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

    nerfdataset = NerfDataset(Path(dataset_path), "transforms_train.json", 8.0)
    dataloader = NerfDataloader(dataloader_key, nerfdataset, BATCH_SIZE)

    ground_truth_image = nerfdataset.images[0]
    pose = jaxlie.SO3.from_matrix(nerfdataset.rotations[0])
    pose = jaxlie.SE3.from_rotation_and_translation(pose, nerfdataset.translations[0])
    camera = PinholeCamera(100.0, nerfdataset.H[0], nerfdataset.W[0], pose, 1.0)

    img = np.uint8(np.array(ground_truth_image)*255.0)
    image = Image.fromarray(img)
    image.save(f"runs/gt.png")

    optimizer = optax.adam(5e-5)
    optimizer_state = optimizer.init(eqx.filter(nerf, eqx.is_array))

    psnr = -1.0

    for step in (pbar := tqdm.trange(1000000)):
        rgb_ground_truths, rays = next(dataloader)

        if step % 200 == 0:
            img = render_frame(nerf, camera, key)

            image = np.array(img)
            image = (image - image.min())/(image.max() - image.min()+0.0001)
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
