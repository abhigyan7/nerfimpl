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
    probs = probs + jnp.ones_like(probs) * jnp.finfo(jnp.float32).eps
    p_cdf = jnp.cumsum(probs)
    p_cdf = p_cdf / p_cdf.max()
    return jnp.interp(uniform_samples, p_cdf, ts)

def calc_transmittance(density, delta):
    ret = density * delta
    ret = jnp.cumsum(ret)
    ret = jnp.exp(-ret)
    ret = jnp.roll(ret, 1)
    ret = ret.at[0].set(0)
    return ret

def calc_alpha(density, delta):
    return 1.0 - jnp.exp(-density * delta)

def calc_w(density, delta):
    alpha = calc_alpha(density, delta)
    T = calc_transmittance(density, delta)
    return T * alpha

def dists(ts):
    return jnp.diff(ts, append=1e10)

def render_single_ray(ray, ts, nerf, key, train=False):
    xyzs = eqx.filter_vmap(ray)(ts)
    locations = jax.vmap(lambda x: positional_encoding(x,10,scale=10.0))(xyzs)
    direction = positional_encoding(ray.direction, 4, scale=10.0)
    nerf_densities, nerf_rgbs = eqx.filter_vmap(nerf, in_axes=(0, None))(locations, direction)
    if train:
        nerf_densities = nerf_densities + jax.random.normal(key, nerf_densities.shape)
    nerf_densities = jax.nn.relu(nerf_densities)
    nerf_rgbs = jax.nn.sigmoid(nerf_rgbs)
    deltas = dists(ts)
    w = calc_w(nerf_densities, deltas)
    rgb = jnp.dot(w, nerf_rgbs)
    return rgb, nerf_densities, nerf_rgbs

def hierarchical_render_single_ray(key, ray, nerf, train=False):
    coarse_reg_key, fine_reg_key, key = jax.random.split(key, 3)
    coarse_key, fine_key = jax.random.split(key, 2)
    coarse_ts = sample_coarse(coarse_key, 32)
    coarse_rgb, coarse_densities, _ = render_single_ray(ray, coarse_ts, nerf, coarse_reg_key, train)

    return coarse_rgb, coarse_rgb

    coarse_densities = jax.lax.stop_gradient(coarse_densities)

    # TODO this isnt correct (see mhall p8 (Normalizing these ...))
    fine_ts = sample_fine(fine_key, 128, coarse_densities, coarse_ts)
    fine_ts = jnp.concatenate((coarse_ts, fine_ts))
    fine_rgb, _, _ = render_single_ray(ray, fine_ts, nerf, fine_reg_key, train)

    return coarse_rgb, fine_rgb

if __name__ == "__main__":
    sample_coarse(jax.random.PRNGKey(0), 16)
    test_density = jnp.array([0.4, 0.9, 0.5, 0.1])
    test_deltas  = jnp.array([1.0, 1.0, 1.0, 1.0])
    T = calc_transmittance(test_density, test_deltas)
    w = calc_w(test_density, test_deltas)
    t = jnp.array([0.1, 0.2, 0.5, 0.9, 1.1, 1.8])
    delta = jnp.diff(t)
    print(T, w)
    print(delta)
    print(dists(t))
