#!/usr/bin/env python3

from nerf.primitives import mlp
from nerf.primitives.encoding import positional_encoding
from nerf.datasets.blender import BlenderDataset
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


def visualize_encoding(encodings, savedir="runs/encoding"):
    encodings = np.array(encodings)
    for i in range(encodings.shape[-1]):
        encoding = encodings[..., i]
        encoding = (encoding - encoding.min()) / (encoding.max() - encoding.min())
        encoding = np.uint8(encoding * 255.0)
        image = Image.fromarray(encoding)
        image.save(f"{savedir}/encoding_{i}.png")


@eqx.filter_jit
def render_image(nerf, pixel_locs):
    locations = jax.vmap(jax.vmap(lambda x: positional_encoding(x, 10, scale=128.0)))(
        pixel_locs
    )
    rgbs = eqx.filter_vmap(eqx.filter_vmap(nerf))(locations)
    return rgbs


@eqx.filter_jit
def optimize_one_batch(nerf, pixel_locs, rgb_ground_truths, optimizer, optimizer_state):
    @eqx.filter_value_and_grad
    def loss_fn(nerf, pixel_locs, rgb_ground_truths):
        locations = jax.vmap(lambda x: positional_encoding(x, 10, scale=128.0))(
            pixel_locs
        )
        rgbs = eqx.filter_vmap(nerf)(locations)
        loss = jnp.mean((rgbs - rgb_ground_truths) ** 2.0)
        return loss

    loss, grad = loss_fn(nerf, pixel_locs, rgb_ground_truths)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, nerf)
    nerf = eqx.apply_updates(nerf, updates)

    return nerf, optimizer_state, loss


def PSNR(ground_truth, pred, max_intensity=1.0):
    MSE = jnp.mean((ground_truth - pred) ** 2.0)
    psnr = 20 * jnp.log10(max_intensity) - 10 * jnp.log10(MSE)
    return psnr


def main():
    key = jax.random.PRNGKey(7)
    nerf_key, sampler_key = jax.random.split(key, 2)

    nerf = mlp.ImageFuncMLP(nerf_key)

    nerfdataset = BlenderDataset(Path(dataset_path), "transforms_train.json", 8.0)

    ground_truth_image = nerfdataset.images[0]

    img = np.uint8(np.array(ground_truth_image) * 255.0)
    image = Image.fromarray(img)
    image.save("runs/gt.png")

    jax_img = jnp.array(np.float32(img)) / 255.0

    optimizer = optax.adam(5e-5)
    optimizer_state = optimizer.init(eqx.filter(nerf, eqx.is_array))

    pixel_locs = (np.arange(jax_img.shape[0]), np.arange(jax_img.shape[1]))
    xx, yy = np.meshgrid(*pixel_locs)
    pixel_locs = np.stack([xx, yy]).T

    psnr = -1.0

    for step in (pbar := tqdm.trange(1000000)):
        key, sampler_key = jax.random.split(sampler_key)

        samples = jax.random.choice(
            key, jax_img.shape[0] * jax_img.shape[1], (BATCH_SIZE,), axis=-1
        )
        b_pixel_locs = pixel_locs.reshape(-1, 2)[samples]
        b_rgb_gts = jax_img.reshape(-1, 3)[samples]

        if step % 200 == 0:
            img = render_image(nerf, pixel_locs)
            image = np.array(img)
            image = np.uint8(np.clip(image * 255.0, 0, 255))
            image = Image.fromarray(image)
            image.save(f"runs/output_{step:09}.png")
            psnr = PSNR(ground_truth_image, img)

        nerf, optimizer_state, loss = optimize_one_batch(
            nerf, b_pixel_locs, b_rgb_gts, optimizer, optimizer_state
        )

        pbar.set_description(f"Loss={loss.item():.4f}, PSNR={psnr:.4f}")
    return


if __name__ == "__main__":
    main()
