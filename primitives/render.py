#!/usr/bin/env python3

import jax
import jax.numpy as jnp

def sample_coarse(key, n_points):
    points = jnp.arange(n_points) + jax.random.uniform(key, (n_points,))
    points = points / n_points
    return points

def sample_fine(key, n_points, probs, ts):
    uniform_samples = jax.random.uniform(key, (n_points,))
    p_cdf = jnp.cumsum(probs)
    p_cdf = p_cdf / p_cdf.max()
    return jnp.interp(uniform_samples, p_cdf, ts)

def calc_T(density, delta):
    ret = density * delta
    ret = jnp.cumsum(ret)
    ret = jnp.exp(-ret)
    ret = jnp.roll(ret, 1)
    ret = ret.at[0].set(0)
    return ret

def calc_w(density, delta):
    T = calc_T(density, delta)
    ret = 1.0 - jnp.exp(-density * delta)
    ret = T * ret
    return ret

if __name__ == "__main__":
    test_density = jnp.array([0.4, 0.9, 0.5, 0.1])
    test_deltas  = jnp.array([1.0, 1.0, 1.0, 1.0])
    T = calc_T(test_density, test_deltas)
    w = calc_w(test_density, test_deltas)
    t = jnp.array([0.1, 0.2, 0.5, 0.9, 1.1, 1.8])
    delta = jnp.diff(t)
