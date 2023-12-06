#!/usr/bin/env python3

import numpy as np
from PIL import Image
import jax.numpy as jnp


def PSNR(ground_truth, pred, max_intensity=1.0):
    MSE = jnp.mean((ground_truth - pred) ** 2.0)
    psnr = 20 * jnp.log10(max_intensity) - 10 * jnp.log10(MSE)
    return psnr

def jax_to_PIL(img):
    """# img is a 0-1 float array. anything outside is clipped.################"""
    img = np.uint8(np.array(img) * 255.0)
    img = np.clip(img, 0, 255)
    image = Image.fromarray(img)
    return image
