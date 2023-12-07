#!/usr/bin/env python3

import os
import json
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

from nerf import utils
from nerf.primitives.mlp import MhallMLP
from nerf.datasets.nerfdata import Dataloader
from nerf.datasets.blender import BlenderDataset
from nerf.render import render_frame, hierarchical_render_single_ray


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


@click.command()
@click.option('--dataset-path', type=Path, required=True)
@click.option('--seed', type=int, default=42)
@click.option('--batch-size', type=int, default=256)
@click.option('--chunk-size', type=int, default=400)
@click.option('--scale', type=float, default=1.0)
@click.option('--num-epochs', type=int, default=1e7)
@click.option('--render-every', type=int, default=200)
@click.option('--checkpoint-every', type=int, default=5000)
@click.option('--resume-from', type=Path)
@click.option('--runs-dir', type=Path, default=Path("runs"))
def train(**conf):
    key = jax.random.PRNGKey(conf["seed"])
    coarse_nerf_key, fine_nerf_key, dataloader_key, sampler_key = jax.random.split(
        key, 4
    )

    logdir = conf["runs_dir"] / datetime.now().strftime('%b%d_%H-%M-%S')
    if conf["resume_from"] is not None:
        logdir = conf["resume_from"]
    writer = SummaryWriter(logdir)
    ckpt_dir = logdir / "ckpts"
    ckpt_file = ckpt_dir / "nerf.eqx"
    render_dir = logdir / "renders"
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    coarse_nerf = MhallMLP(coarse_nerf_key)
    fine_nerf = MhallMLP(fine_nerf_key)
    nerfs = [coarse_nerf, fine_nerf]
    start_step = 0
    if conf["resume_from"] is not None:
        assert ckpt_file.exists(), "NeRF checkpoint not found"
        with open(ckpt_file, "rb") as f:
            train_state = json.loads(f.readline().decode())
            start_step = train_state["step"]
            nerfs = eqx.tree_deserialise_leaves(f, nerfs)
        print(f"Loaded checkpoints from {ckpt_file}.\nResuming training from step {start_step}.")

    nerfdataset = BlenderDataset(conf["dataset_path"], "transforms_train.json", conf["scale"])
    nerfdataset_test = nerfdataset
    dataloader = Dataloader(dataloader_key, nerfdataset, conf["batch_size"])

    gt_ids = jnp.array([0, 23, 50, 75, 90], dtype=jnp.int32)
    ground_truth_images = jax.vmap(lambda i: nerfdataset_test.images[i]) (gt_ids)

    cameras = []
    for gt_id in gt_ids:
        cameras.append(jax.tree_map(
            lambda tensor: tensor[gt_id],
            nerfdataset_test.cameras,
        ))

    for i, gt_id in enumerate(gt_ids):
        image = utils.jax_to_PIL(ground_truth_images[i])
        writer.add_image(f"ground-truth/{gt_id}", np.array(image), 0, dataformats="HWC")
        image.save(logdir/f"gt_{gt_id}.png")
        os.makedirs(render_dir/f"{gt_id}", exist_ok=True)

    optimizer = optax.adam(5e-4)
    optimizer_state = optimizer.init(eqx.filter(nerfs, eqx.is_array))

    psnr = 0.0
    pbar = trange(start_step, conf["num_epochs"], initial=start_step, total=conf["num_epochs"])

    for step in (pbar):
        rgb_ground_truths, rays = dataloader.get_batch()
        key, sampler_key = jax.random.split(key)

        nerfs, optimizer_state, loss = optimize_one_batch(
            nerfs, rays, rgb_ground_truths, sampler_key, optimizer, optimizer_state
        )

        if step % conf["render_every"] == 0:
            imgs = map(
                lambda c: render_frame(nerfs, c, key, conf["chunk_size"]),
                tqdm(cameras, desc="Rendering test images: ", leave=False)
            )
            imgs = list(imgs)
            for i, img in zip(gt_ids, imgs):
                image = utils.jax_to_PIL(img)
                image.save(render_dir / f"{i}" / f"output_{step:09}.png")
                writer.add_image(f"render/{i}", np.array(image), step, dataformats="HWC")
            imgs = jnp.stack(imgs)
            psnr = jnp.mean(jax.vmap(utils.PSNR) (ground_truth_images, imgs))
            writer.add_scalar("psnr", psnr, step)

        if step % conf["checkpoint_every"] == 0 and step > 0:
            train_state = json.dumps({
                "step": step,
            }) + '\n'
            with open(ckpt_file, "wb") as f:
                f.write(train_state.encode())
                eqx.tree_serialise_leaves(f, nerfs)

        pbar.set_description(f"Loss={loss.item():.4f}, utils.PSNR={psnr:.4f}")
        writer.add_scalar("loss", loss.item(), step)
    print("Training done!")
    return


if __name__ == "__main__":
    train()
