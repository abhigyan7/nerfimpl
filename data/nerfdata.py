#!/usr/bin/env python3

import jax
import json
import numpy as np
from PIL import Image
import equinox as eqx
import jax.numpy as jnp
from jaxlie import SE3, SO3

from primitives.camera import PinholeCamera

class NerfDataset():

    def __init__(self, scene_path):
        with open(scene_path/"transforms_train.json") as f:
            train_frames = json.load(f)
        with open(scene_path/"transforms_test.json") as f:
            test_frames = json.load(f)
        with open(scene_path/"transforms_val.json") as f:
            validation_frames = json.load(f)

        self.train_images = []
        self.train_poses = []
        self.train_fov = train_frames["camera_angle_x"]
        self.rotations = []
        self.translations = []
        for frame in train_frames["frames"]:
            img_path = scene_path / f"{frame['file_path']}.png"
            image = Image.open(img_path)
            self.train_images.append(np.asarray(image)[..., :3])

            transform = np.array(frame['transform_matrix'])
            self.rotations.append(transform[:3, :3])
            self.translations.append(transform[:3, 3].squeeze())
            t = SE3.from_rotation_and_translation(
                SO3.from_matrix(transform[:3, :3]),
                transform[:3, 3].squeeze(),
            )
            self.train_poses.append(t)

        self.train_images = jnp.stack(self.train_images, 0)
        self.rotations = jnp.stack(self.rotations, 0)
        self.translations = jnp.stack(self.translations, 0)

        self.rotations_SO3 = jax.vmap(SO3.from_matrix)(self.rotations)
        self.train_poses = jax.vmap(SE3.from_rotation_and_translation)(self.rotations_SO3, self.translations)

    def __getitem__(self, i):
        return self.train_images[i], self.train_poses[i]

    def __len__(self):
        return len(self.train_images)


class NerfDataloader:
    def __init__(self, key, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        key_b, key_u, key_v = jax.random.split(self.key, 3)
        batch_idx = jax.random.choice(key_b, len(self.dataset), (self.batch_size,))
        W, H = 800, 800
        us = jax.random.choice(key_u, W, (self.batch_size,))
        vs = jax.random.choice(key_v, H, (self.batch_size,))
        # rgb_ground_truths = jax.vmap(self.dataset.train_images.__getitem__)(batch_idx)
        rgb_ground_truths = self.dataset.train_images[batch_idx]
        breakpoint()
        poses = jax.tree_map(lambda x: self.dataset.train_poses[x], batch_idx)
        poses = self.dataset.train_poses[batch_idx]
        # poses = eqx.filter_vmap(lambda x: jnp.take(self.dataset.train_poses, x))(batch_idx)
        #poses = [self.dataset.train_poses[i] for i in batch_idx]
        rgb_ground_truths = jax.vmap(lambda x, u, v: x[u][v])(rgb_ground_truths, us, vs)

        breakpoint()
        #cameras = [PinholeCamera(100.0, H, W, pose, 1.0) for pose in poses]
        cameras = eqx.filter_vmap(lambda x: PinholeCamera(100.0, H, W, x, 1.0))(poses)
        cameras = eqx.filter_vmap(PinholeCamera, in_axes=(None, None, None, 0, None)
                           )(100.0, H, W, poses, 1.0)
        rays = jax.vmap(lambda x, u, v: x.get_ray(u,v))(cameras, us, vs)
        return rgb_ground_truths, rays
