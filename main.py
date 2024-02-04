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
from nerf.utils import timing, serialize, deserialize, jax_to_PIL, PSNR, mse_to_psnr


@eqx.filter_jit
def optimize_one_batch(
    nerfs,
    rays,
    rgb_ground_truths,
    key,
    optimizer,
    optimizer_state,
    renderer_settings,
):
    @eqx.filter_value_and_grad
    def loss_fn(nerfs, rays, rgb_ground_truths, key, renderer_settings):
        keys = jax.random.split(key, rays.origin.shape[0])
        coarse_rgbs, fine_rgbs = eqx.filter_vmap(
            hierarchical_render_single_ray, in_axes=(0, 0, None, None, None)
        )(keys, rays, nerfs, True, renderer_settings)
        loss = jnp.mean((fine_rgbs - rgb_ground_truths) ** 2.0) + jnp.mean(
            (coarse_rgbs - rgb_ground_truths) ** 2.0
        )
        return loss

    loss, grad = loss_fn(nerfs, rays, rgb_ground_truths, key, renderer_settings)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    nerfs = eqx.apply_updates(nerfs, updates)

    return nerfs, optimizer_state, loss


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dataset-path", type=Path, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--chunk-size", type=int, default=400)
@click.option("--scale", type=float, default=1.0)
@click.option("--t-sampling", type=str, default="linear")
@click.option("--num-fine-samples", type=int, default=64)
@click.option("--num-coarse-samples", type=int, default=128)
@click.option("--output-dir", type=Path, default=Path("runs"))
@click.option("--nerf-weights", type=Path, required=True)
def evaluate(**conf):
    key = jax.random.PRNGKey(conf["seed"])
    coarse_nerf_key, fine_nerf_key, sampler_key = jax.random.split(key, 3)
    output_dir = conf["output_dir"] / datetime.now().strftime("%b%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)

    coarse_nerf = MhallMLP(coarse_nerf_key)
    fine_nerf = MhallMLP(fine_nerf_key)
    nerfs = [coarse_nerf, fine_nerf]
    nerfs, metadata = deserialize(nerfs, conf["nerf_weights"])
    loc_encoding_scale = metadata["loc_encoding_scale"]
    print(f"Loaded checkpoints from {conf['nerf_weights']}.")

    nerfdataset_test = BlenderDataset(
        conf["dataset_path"], "transforms_test.json", conf["scale"]
    )

    renderer_settings = {
        "loc_encoding_scale": loc_encoding_scale,
        "t_sampling": conf["t_sampling"],
        "num_coarse_samples": conf["num_coarse_samples"],
        "num_fine_samples": conf["num_fine_samples"],
    }

    ground_truth_images = nerfdataset_test.images

    cameras = nerfdataset_test.cameras

    rendered_imgs = list(
        map(
            lambda c: render_frame(
                nerfs, c, sampler_key, conf["chunk_size"], renderer_settings
            ),
            tqdm(cameras, desc="Rendering test images: ", leave=False),
        )
    )

    psnrs = []

    for i, ground_truth_image, (coarse_img, fine_img) in enumerate(
        zip(ground_truth_images, rendered_imgs)
    ):
        image = jax_to_PIL(ground_truth_image)
        image.save(output_dir / f"gt_{i}.png")
        image = jax_to_PIL(coarse_img)
        image.save(output_dir / f"coarse_{i}.png")
        image = jax_to_PIL(fine_img)
        image.save(output_dir / f"fine_{i}.png")

        psnr = PSNR(ground_truth_image, fine_img)
        print(f"Image {i:03}, psnr={psnr:.02}")
        psnrs.append(psnr)

    print(f"{np.mean(psnrs)=}")
    print("Evaluation done!")
    return


@cli.command()
@click.option("--dataset-path", type=Path, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--chunk-size", type=int, default=400)
@click.option("--scale", type=float, default=1.0)
@click.option("--t-sampling", type=str, default="linear")
@click.option("--num-fine-samples", type=int, default=64)
@click.option("--num-coarse-samples", type=int, default=128)
@click.option("--output-dir", type=Path, default=Path("runs"))
@click.option("--nerf-weights", type=Path, required=True)
@click.option("--pose-ids", type=str, default="-1")
@timing
def render(**conf):
    key = jax.random.PRNGKey(conf["seed"])
    coarse_nerf_key, fine_nerf_key, sampler_key = jax.random.split(key, 3)
    pose_ids = [int(x) for x in conf["pose_ids"].split(",")]
    output_dir = conf["output_dir"] / datetime.now().strftime("%b%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)

    coarse_nerf = MhallMLP(coarse_nerf_key)
    fine_nerf = MhallMLP(fine_nerf_key)
    nerfs = [coarse_nerf, fine_nerf]
    nerfs, metadata = deserialize(nerfs, conf["nerf_weights"])
    loc_encoding_scale = metadata["loc_encoding_scale"]
    print(f"Loaded checkpoints from {conf['nerf_weights']}.")

    nerfdataset_test = BlenderDataset(
        conf["dataset_path"], "transforms_test.json", conf["scale"]
    )

    renderer_settings = {
        "loc_encoding_scale": loc_encoding_scale,
        "t_sampling": conf["t_sampling"],
        "num_coarse_samples": conf["num_coarse_samples"],
        "num_fine_samples": conf["num_fine_samples"],
    }

    gt_ids = jnp.array(pose_ids, dtype=jnp.int32)
    ground_truth_images = jax.vmap(lambda i: nerfdataset_test.images[i])(gt_ids)

    cameras = []
    for gt_id in gt_ids:
        cameras.append(
            jax.tree_map(
                lambda tensor: tensor[gt_id],
                nerfdataset_test.cameras,
            )
        )

    rendered_imgs = list(
        map(
            lambda c: render_frame(
                nerfs, c, sampler_key, conf["chunk_size"], renderer_settings
            ),
            tqdm(cameras, desc="Rendering test images: ", leave=False),
        )
    )

    for i, ground_truth_image, (coarse_img, fine_img) in zip(
        gt_ids, ground_truth_images, rendered_imgs
    ):
        image = jax_to_PIL(ground_truth_image)
        image.save(output_dir / f"gt_{i}.png")
        image = jax_to_PIL(coarse_img)
        image.save(output_dir / f"coarse_{i}.png")
        image = jax_to_PIL(fine_img)
        image.save(output_dir / f"fine_{i}.png")
    print("Render done!")

    return


@cli.command()
@click.option("--dataset-path", type=Path, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--batch-size", type=int, default=256)
@click.option("--lr", type=float, default=5e-4)
@click.option("--chunk-size", type=int, default=400)
@click.option("--scale", type=float, default=1.0)
@click.option("--num-steps", type=int, default=1e7)
@click.option("--render-every", type=int, default=200)
@click.option("--checkpoint-every", type=int, default=5000)
@click.option("--t-sampling", type=str, default="linear")
@click.option("--num-fine-samples", type=int, default=64)
@click.option("--num-coarse-samples", type=int, default=128)
@click.option("--resume-from", type=Path)
@click.option("--runs-dir", type=Path, default=Path("runs"))
@timing
def train(**conf):
    key = jax.random.PRNGKey(conf["seed"])
    coarse_nerf_key, fine_nerf_key, dataloader_key, sampler_key = jax.random.split(
        key, 4
    )

    logdir = conf["runs_dir"] / datetime.now().strftime("%b%d_%H-%M-%S")
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

    loc_encoding_scale = -1.0

    if conf["resume_from"] is not None:
        assert ckpt_file.exists(), "NeRF checkpoint not found"
        nerfs, train_state = deserialize(nerfs, ckpt_file)
        start_step = train_state["step"]
        loc_encoding_scale = train_state["loc_encoding_scale"]
        print(f"Loaded checkpoints from {ckpt_file}.")
        optimizer_state, _ = deserialize(
            optimizer_state, optstate_file, has_metadata=False
        )
        print(f"Loaded optimizer state from {optstate_file}.")
        print(f"Resuming training from step {start_step}.")

    nerfdataset = BlenderDataset(
        conf["dataset_path"], "transforms_train.json", conf["scale"]
    )
    nerfdataset_test = BlenderDataset(
        conf["dataset_path"], "transforms_test.json", conf["scale"], N=30
    )

    dataloader = Dataloader(dataloader_key, nerfdataset, conf["batch_size"])

    t_min = nerfdataset.translations.min()
    t_max = nerfdataset.translations.max()
    if loc_encoding_scale < 0.0:
        loc_encoding_scale = (t_max - t_min) * 1.2
        loc_encoding_scale = loc_encoding_scale.item()

    renderer_settings = {
        "loc_encoding_scale": loc_encoding_scale,
        "t_sampling": conf["t_sampling"],
        "num_coarse_samples": conf["num_coarse_samples"],
        "num_fine_samples": conf["num_fine_samples"],
    }

    gt_ids = jnp.array([0, 15, 20, 29], dtype=jnp.int32)
    ground_truth_images = jax.vmap(lambda i: nerfdataset_test.images[i])(gt_ids)

    cameras = []
    for gt_id in gt_ids:
        cameras.append(
            jax.tree_map(
                lambda tensor: tensor[gt_id],
                nerfdataset_test.cameras,
            )
        )

    for i, gt_id in enumerate(gt_ids):
        image = jax_to_PIL(ground_truth_images[i])
        writer.add_image(f"ground-truth/{gt_id}", np.array(image), 0, dataformats="HWC")
        image.save(logdir / f"gt_{gt_id}.png")
        os.makedirs(render_dir / f"{gt_id}", exist_ok=True)

    psnr = 0.0
    pbar = trange(
        start_step, conf["num_steps"], initial=start_step, total=conf["num_steps"]
    )

    for step in pbar:
        rgb_ground_truths, rays = dataloader.get_batch()
        key, sampler_key = jax.random.split(key)

        nerfs, optimizer_state, loss = optimize_one_batch(
            nerfs,
            rays,
            rgb_ground_truths,
            sampler_key,
            optimizer,
            optimizer_state,
            renderer_settings,
        )

        if step % conf["render_every"] == 0 or (step + 1) == conf["num_steps"]:
            rendered_imgs = list(
                map(
                    lambda c: render_frame(
                        nerfs, c, key, conf["chunk_size"], renderer_settings
                    ),
                    tqdm(cameras, desc="Rendering test images: ", leave=False),
                )
            )
            for i, (coarse_img, fine_img) in zip(gt_ids, rendered_imgs):
                image = jax_to_PIL(coarse_img)
                image.save(render_dir / f"{i}" / f"coarse_output_{step:09}.png")
                writer.add_image(
                    f"coarse_render/{i}", np.array(image), step, dataformats="HWC"
                )
                image = jax_to_PIL(fine_img)
                image.save(render_dir / f"{i}" / f"fine_output_{step:09}.png")
                writer.add_image(
                    f"fine_render/{i}", np.array(image), step, dataformats="HWC"
                )
            imgs = jnp.stack([c_i for (c_i, _) in rendered_imgs])
            psnr = jnp.mean(jax.vmap(PSNR)(ground_truth_images, imgs))
            writer.add_scalar("coarse_psnr", psnr, step)
            imgs = jnp.stack([f_i for (_, f_i) in rendered_imgs])
            psnr = jnp.mean(jax.vmap(PSNR)(ground_truth_images, imgs))
            writer.add_scalar("fine_psnr", psnr, step)

        if step % conf["checkpoint_every"] == 0 and step > 0:
            serialize(
                nerfs,
                ckpt_file,
                {"step": step, "loc_encoding_scale": loc_encoding_scale},
            )
            serialize(optimizer_state, optstate_file, metadata=None)

        pbar.set_description(f"Loss={loss.item():.4f}, utils.PSNR={psnr:.4f}")
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("psnr", mse_to_psnr(loss.item() / 2.0), step)
        writer.add_scalar(
            "learning rate", lr_sched(optimizer_state[1].count.item()), step
        )

    print("Training done!")
    return


if __name__ == "__main__":
    cli()
