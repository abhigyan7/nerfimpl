#!/usr/bin/env python3

import jax
import json
import numpy as np
from PIL import Image
import equinox as eqx
import jax.numpy as jnp
from jaxlie import SE3, SO3

from primitives.camera import PinholeCamera

def process_transforms_json(frames, scene_path, scale=1.0):

    images = []
    H = []
    W = []
    f = []
    rotations = []
    translations = []
    train_fov = frames["camera_angle_x"]
    for frame in frames["frames"]:
        img_path = scene_path / f"{frame['file_path']}.png"
        image = Image.open(img_path)
        image = image.resize((int(s/scale) for s in image.size))
        image = np.asarray(image, dtype=np.float32)[..., :3]
        images.append(image)
        H.append(image.shape[0])
        W.append(image.shape[1])
        f.append(image.shape[1] / (np.tan(train_fov / 2.0)))

        transform = np.array(frame['transform_matrix'])
        rotations.append(transform[:3, :3])
        translations.append(transform[:3, 3].squeeze())

    return images, H, W, f, rotations, translations

class NerfDataset():

    def __init__(self, scene_path, transforms_file, scale=1.0):
        with open(scene_path/transforms_file) as f:
            frames = json.load(f)
            outputs = process_transforms_json(frames, scene_path, scale)
            self.images, self.H, self.W, self.f, self.rotations, \
                self.translations = outputs

        self.images = jnp.stack(self.images, 0) / 255.0
        self.rotations = jnp.stack(self.rotations, 0)
        self.rotations_SO3 = jax.vmap(SO3.from_matrix)(self.rotations)
        self.translations = jnp.stack(self.translations, 0)
        self.poses = jax.vmap(SE3.from_rotation_and_translation
                              )(self.rotations_SO3, self.translations)
        self.H = np.array(self.H)
        self.W = np.array(self.W)
        self.f = np.array(self.f)

    def __getitem__(self, i):
        return self.images[i], self.poses[i]

    def __len__(self):
        return len(self.images)


class NerfDataloader:
    def __init__(self, key, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = key
        self.W = dataset.W
        self.H = dataset.H

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, key_b, key_u, key_v = jax.random.split(self.key, 4)
        batch_idx = jax.random.choice(key_b, len(self.dataset), (self.batch_size,))
        us = jax.vmap(lambda x, k: jax.random.uniform(k)*x) (
            self.W[batch_idx],
            jax.random.split(key_u, self.batch_size),
        )

        vs = jax.vmap(lambda x, k: jax.random.uniform(k)*x) (
            self.H[batch_idx],
            jax.random.split(key_v, self.batch_size),
        )
        us = jnp.int32(us)
        vs = jnp.int32(vs)

        rgb_ground_truths = self.dataset.images[batch_idx]
        rotations = self.dataset.rotations[batch_idx]
        translations = self.dataset.translations[batch_idx]
        rotations_SO3 = jax.vmap(SO3.from_matrix)(rotations)
        poses = jax.vmap(SE3.from_rotation_and_translation)(rotations_SO3, translations)
        rgb_ground_truths = jax.vmap(lambda x, u, v: x[v][u])(rgb_ground_truths, us, vs)

        cameras = jax.vmap(lambda x,h,w,f: PinholeCamera(f, h, w, x, 0.01)
                           ) (poses,
                              self.H[batch_idx],
                              self.W[batch_idx],
                              self.dataset.f[batch_idx])
        rays = jax.vmap(lambda x, u, v: x.get_ray(u,v))(cameras, us, vs)
        return rgb_ground_truths, rays
