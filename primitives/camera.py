#!/usr/bin/env python3

class PinholeCamera():

    def __init__(self, f, w, h, cx = 0.0, cy = 0.0):
        if isinstance(f, tuple) or isinstance(f, list):
            self.f = (f[0], f[1])
        else:
            self.f = f
        self.w = w
        self.h = h
        return

    def get_ray(self, px, py):
        # should return centers and directions for the specified pixel
        raise NotImplementedError

    def get_ray_bundle(self):
        # should return center and directions for all rays coming out of the camera
        pixels = None # get all pixels
        # can do this? return jax.vmap(self.get_ray, pixels)
        raise NotImplementedError
