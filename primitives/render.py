#!/usr/bin/env python3

import jax
import jax.numpy as jnp

def sample_t(key, n_points, probs=None, ts = None):
    if probs is None:
        points = jnp.arange(n_points) + jax.random.uniform(key, (n_points,))
        points = points / n_points
        return points
    uniform_samples = jax.random.uniform(key, (n_points,))
    p_cdf = jnp.cumsum(probs)
    p_cdf = p_cdf / p_cdf.max()
    return jnp.interp(uniform_samples, p_cdf, ts)
