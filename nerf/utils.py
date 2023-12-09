#!/usr/bin/env python3

import json
import numpy as np
from PIL import Image
import equinox as eqx
from time import time
import jax.numpy as jnp
from functools import wraps


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


def serialize(parameters, checkpoint_filepath, metadata=None):
    """# Dump metadata as json to a file and append model parameters. #########"""
    with open(checkpoint_filepath, "wb") as f:
        if metadata is not None:
            metadata= json.dumps(metadata)
            f.write((metadata + "\n").encode())
        eqx.tree_serialise_leaves(f, parameters)


def deserialize(parameters, checkpoint_filepath, has_metadata=True):
    """# Load metadata and model parameters."""
    with open(checkpoint_filepath, "rb") as f:
        if has_metadata:
            metadata = json.loads(f.readline().decode())
        parameters = eqx.tree_deserialise_leaves(f, parameters)
    if has_metadata:
        return parameters, metadata
    return parameters


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap
