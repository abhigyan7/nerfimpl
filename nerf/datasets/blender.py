#!/usr/bin/env python3

import jax
import json
import numpy as np
from PIL import Image
import jax.numpy as jnp
from jaxlie import SE3, SO3

from nerf.datasets.nerfdata import Dataset
from nerf.primitives.camera import PinholeCamera


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
        image = image.resize((int(s / scale) for s in image.size))
        image = np.asarray(image, dtype=np.float32)[..., :3]
        images.append(image)
        H.append(image.shape[0])
        W.append(image.shape[1])
        f.append(image.shape[1] / (np.tan(train_fov / 2.0)))

        transform = np.array(frame["transform_matrix"])
        rotations.append(transform[:3, :3])
        translations.append(transform[:3, 3].squeeze())

    return images, H, W, f, rotations, translations


class BlenderDataset(Dataset):
    def __init__(self, scene_path, transforms_file, scale=1.0):
        with open(scene_path / transforms_file) as f:
            frames = json.load(f)
            outputs = process_transforms_json(frames, scene_path, scale)
            (
                self.images,
                self.H,
                self.W,
                self.f,
                self.rotations,
                self.translations,
            ) = outputs

        self.images = jnp.stack(self.images, 0) / 255.0
        self.rotations = jnp.stack(self.rotations, 0)
        self.rotations_SO3 = jax.vmap(SO3.from_matrix)(self.rotations)
        self.translations = jnp.stack(self.translations, 0)
        self.poses = jax.vmap(SE3.from_rotation_and_translation)(
            self.rotations_SO3, self.translations
        )
        self.H = np.array(self.H)
        self.W = np.array(self.W)
        self.f = np.array(self.f)

        self.cameras = jax.vmap(lambda x, h, w, f: PinholeCamera(f, h, w, x))(
            self.poses, self.H, self.W, self.f
        )

    def __getitem__(self, i):
        return self.images[i], self.poses[i]

    def __len__(self):
        return len(self.images)
