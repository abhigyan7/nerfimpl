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

from nerf.primitives.mlp import MhallMLP
from nerf.datasets.nerfdata import Dataloader
from nerf.datasets.blender import BlenderDataset
from nerf.render import render_frame, hierarchical_render_single_ray
from nerf.utils import timing, serialize, deserialize, jax_to_PIL, PSNR


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
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    nerfs = eqx.apply_updates(nerfs, updates)

    return nerfs, optimizer_state, loss


@click.command()
@click.option('--dataset-path', type=Path, required=True)
@click.option('--seed', type=int, default=42)
@click.option('--batch-size', type=int, default=256)
@click.option('--lr', type=float, default=5e-4)
@click.option('--chunk-size', type=int, default=400)
@click.option('--scale', type=float, default=1.0)
@click.option('--num-steps', type=int, default=1e7)
@click.option('--render-every', type=int, default=200)
@click.option('--checkpoint-every', type=int, default=5000)
@click.option('--resume-from', type=Path)
@click.option('--runs-dir', type=Path, default=Path("runs"))
@timing
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
    optstate_file = ckpt_dir / "optstate.eqx"
    render_dir = logdir / "renders"
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    coarse_nerf = MhallMLP(coarse_nerf_key)
    fine_nerf = MhallMLP(fine_nerf_key)
    nerfs = [coarse_nerf, fine_nerf]
    start_step = 0
    lr_sched = optax.cosine_decay_schedule(conf["lr"], conf["num_steps"])
    optimizer = optax.adam(lr_sched)
    optimizer_state = optimizer.init(eqx.filter(nerfs, eqx.is_array))

    if conf["resume_from"] is not None:
        assert ckpt_file.exists(), "NeRF checkpoint not found"
        nerfs, train_state = deserialize(
            nerfs, ckpt_file)
        start_step = train_state["step"]
        print(f"Loaded checkpoints from {ckpt_file}.")
        optimizer_state = deserialize(
            optimizer_state, optstate_file, has_metadata=False)
        print(f"Loaded optimizer state from {optstate_file}.")
        print(f"Resuming training from step {start_step}.")

    nerfdataset = BlenderDataset(conf["dataset_path"], "transforms_train.json", conf["scale"])
    nerfdataset_test = BlenderDataset(
        conf["dataset_path"], 
        "transforms_test.json", 
        conf["scale"], 
        nerfdataset.t_min, nerfdataset.t_max,
        N=30)
    dataloader = Dataloader(dataloader_key, nerfdataset, conf["batch_size"])

    gt_ids = jnp.array([0, 5, 10, 15, 20, 25, 29], dtype=jnp.int32)
    ground_truth_images = jax.vmap(lambda i: nerfdataset_test.images[i]) (gt_ids)

    cameras = []
    for gt_id in gt_ids:
        cameras.append(jax.tree_map(
            lambda tensor: tensor[gt_id],
            nerfdataset_test.cameras,
        ))

    for i, gt_id in enumerate(gt_ids):
        image = jax_to_PIL(ground_truth_images[i])
        writer.add_image(f"ground-truth/{gt_id}", np.array(image), 0, dataformats="HWC")
        image.save(logdir/f"gt_{gt_id}.png")
        os.makedirs(render_dir/f"{gt_id}", exist_ok=True)


    psnr = 0.0
    pbar = trange(start_step, conf["num_steps"], initial=start_step, total=conf["num_steps"])

    for step in (pbar):
        rgb_ground_truths, rays = dataloader.get_batch()
        key, sampler_key = jax.random.split(key)

        nerfs, optimizer_state, loss = optimize_one_batch(
            nerfs, rays, rgb_ground_truths, sampler_key, optimizer, optimizer_state
        )

        if step % conf["render_every"] == 0 or (step+1) == conf["num_steps"]:
            rendered_imgs = list(map(
                lambda c: render_frame(nerfs, c, key, conf["chunk_size"]),
                tqdm(cameras, desc="Rendering test images: ", leave=False)
            ))
            for i, (coarse_img, fine_img) in zip(gt_ids, rendered_imgs):
                image = jax_to_PIL(coarse_img)
                image.save(render_dir / f"{i}" / f"coarse_output_{step:09}.png")
                writer.add_image(f"coarse_render/{i}", np.array(image), step, dataformats="HWC")
                image = jax_to_PIL(fine_img)
                image.save(render_dir / f"{i}" / f"fine_output_{step:09}.png")
                writer.add_image(f"fine_render/{i}", np.array(image), step, dataformats="HWC")
            imgs = jnp.stack([c_i for (c_i, f_i) in rendered_imgs])
            psnr = jnp.mean(jax.vmap(PSNR) (ground_truth_images, imgs))
            writer.add_scalar("coarse_psnr", psnr, step)
            imgs = jnp.stack([f_i for (c_i, f_i) in rendered_imgs])
            psnr = jnp.mean(jax.vmap(PSNR) (ground_truth_images, imgs))
            writer.add_scalar("fine_psnr", psnr, step)

        if step % conf["checkpoint_every"] == 0 and step > 0:
            serialize(nerfs, ckpt_file, {"step":step})
            serialize(optimizer_state, optstate_file, metadata=None)

        pbar.set_description(f"Loss={loss.item():.4f}, utils.PSNR={psnr:.4f}")
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("learning rate", 
            lr_sched(optimizer_state[1].count.item()), step )


    print("Training done!")
    return


if __name__ == "__main__":
    train()
