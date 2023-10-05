#!/usr/bin/env python3

import jax
import jax.numpy as jnp

import equinox as eqx

from primitives.encoding import positional_encoding

def cart2sph(xyz):
    xy = xyz[0]**2 + xyz[1]**2
    theta = jnp.sqrt(xy + xyz[2]**2)
    si = jnp.arctan2(jnp.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    r = jnp.arctan2(xyz[1], xyz[0])
    return jnp.array([theta, si, r])

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

def render_single_ray(ray, ts, nerf):
    xyzs = jax.vmap(ray)(ts)
    location = jax.vmap(lambda x: positional_encoding(x,10))(xyzs)
    direction = jnp.tile(positional_encoding(ray.direction, 4), (location.shape[0], 1))
    nerf_densities, nerf_rgbs = jax.vmap(nerf)(location, direction)
    T = calc_T(nerf_densities[1:], jnp.diff(ts))
    w = calc_w(nerf_densities[1:], jnp.diff(ts))
    alpha = T * w
    rgb = (alpha @ nerf_rgbs[1:])
    return rgb, nerf_densities, nerf_rgbs

def hierarchical_render_single_ray(key, ray, nerf):
    coarse_key, fine_key = jax.random.split(key, 2)
    coarse_ts = sample_coarse(coarse_key, 64)
    coarse_rgb, coarse_densities, _ = render_single_ray(ray, coarse_ts, nerf)

    fine_ts = sample_fine(fine_key, 128, coarse_densities, coarse_ts)
    fine_ts = jnp.concatenate((coarse_ts, fine_ts))
    fine_rgb, _, _ = render_single_ray(ray, fine_ts, nerf)

    return coarse_rgb, fine_rgb

if __name__ == "__main__":
    test_density = jnp.array([0.4, 0.9, 0.5, 0.1])
    test_deltas  = jnp.array([1.0, 1.0, 1.0, 1.0])
    T = calc_T(test_density, test_deltas)
    w = calc_w(test_density, test_deltas)
    t = jnp.array([0.1, 0.2, 0.5, 0.9, 1.1, 1.8])
    delta = jnp.diff(t)
