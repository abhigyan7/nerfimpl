#!/usr/bin/env python3

import json
import numpy as np
from PIL import Image
import jax.numpy as jnp
from jaxlie import SE3, SO3

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
        for frame in train_frames["frames"]:
            img_path = scene_path / f"{frame['file_path']}.png"
            image = Image.open(img_path)
            self.train_images.append(np.asarray(image))

            transform = np.array(frame['transform_matrix'])
            t = SE3.from_rotation_and_translation(
                SO3.from_matrix(transform[:3, :3]),
                transform[:3, 3].squeeze(),
            )
            self.train_poses.append(t)

        self.train_images = jnp.stack(self.train_images, -1)

    def __getitem__(self, i):
        return self.train_images[i], self.train_poses[i]

    def __len__(self):
        return len(self.train_images)
