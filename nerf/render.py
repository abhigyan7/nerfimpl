#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import equinox as eqx
from nerf.primitives.camera import inverse_transform


def cart2sph(xyz):
    xy = xyz[0] ** 2 + xyz[1] ** 2
    theta = jnp.sqrt(xy + xyz[2] ** 2)
    si = jnp.arctan2(jnp.sqrt(xy), xyz[2])
    r = jnp.arctan2(xyz[1], xyz[0])
    return jnp.array([theta, si, r])


def sample_coarse(key, n_points):
    points = jnp.arange(n_points) + jax.random.uniform(key, (n_points,))
    points = points / n_points
    return points


def sample_fine(key, n_points, probs, ts):
    uniform_samples = jax.random.uniform(key, (n_points,))
    probs = probs + jnp.ones_like(probs) * jnp.finfo(jnp.float32).resolution
    p_cdf = jnp.cumsum(probs)
    p_cdf = p_cdf / p_cdf.max()
    return jnp.interp(uniform_samples, p_cdf, ts)


def calc_transmittance(alphas):
    ret = 1 - alphas
    ret = jnp.clip(ret, 1e-10, 1.0)
    ret = jnp.cumprod(ret)
    ret = jnp.roll(ret, 1)
    ret = ret.at[0].set(1)
    return ret


def calc_alpha(density, delta):
    return 1.0 - jnp.exp(-density * delta)


def calc_w(density, delta):
    alphas = calc_alpha(density, delta)
    T = calc_transmittance(alphas)
    return T * alphas


def dists(ts):
    return jnp.diff(ts, append=1e10)


def render_single_ray(ray, ts, nerf, key, train=False):
    locations = eqx.filter_vmap(ray)(ts)
    direction = ray.direction / jnp.linalg.norm(ray.direction)
    nerf_densities, nerf_rgbs = eqx.filter_vmap(nerf, in_axes=(0, None))(
        locations, direction
    )
    if train:
        nerf_densities = nerf_densities + jax.random.normal(key, nerf_densities.shape)
    nerf_densities = jax.nn.relu(nerf_densities)
    nerf_rgbs = jax.nn.sigmoid(nerf_rgbs)
    deltas = dists(ts)
    w = calc_w(nerf_densities, deltas)
    rgb = jnp.dot(w, nerf_rgbs)
    return rgb, nerf_densities, nerf_rgbs, w


def hierarchical_render_single_ray(key, ray, nerfs, train, renderer_settings):
    coarse_reg_key, fine_reg_key, key = jax.random.split(key, 3)
    coarse_key, fine_key = jax.random.split(key, 2)
    coarse_ts = sample_coarse(coarse_key, renderer_settings["num_coarse_samples"])
    if renderer_settings["t_sampling"] == "inverse":
        coarse_ts = inverse_transform(coarse_ts)
    coarse_rgb, coarse_densities, _, _ = render_single_ray(
        ray,
        coarse_ts,
        nerfs[0],
        coarse_reg_key,
        train,
    )
    coarse_densities = jax.lax.stop_gradient(coarse_densities)

    weights = calc_w(coarse_densities, dists(coarse_ts))

    if renderer_settings["t_sampling"] == "inverse":
        coarse_ts = inverse_transform(coarse_ts)
    fine_ts = sample_fine(
        fine_key, renderer_settings["num_fine_samples"], weights, coarse_ts
    )
    fine_ts = jnp.concatenate((coarse_ts, fine_ts))
    fine_ts = jnp.sort(fine_ts)
    if renderer_settings["t_sampling"] == "inverse":
        fine_ts = inverse_transform(fine_ts)

    fine_rgb, _, _, fine_weights = render_single_ray(
        ray,
        fine_ts,
        nerfs[1],
        fine_reg_key,
        train,
    )

    return coarse_rgb, fine_rgb, fine_weights.dot(fine_ts)


@eqx.filter_jit
def render_line(nerfs, rays, key, renderer_settings):
    keys = jax.random.split(key, rays.origin.shape[0])
    coarse_rgbs, fine_rgbs, depth_map = eqx.filter_vmap(
        hierarchical_render_single_ray, in_axes=(0, 0, None, None, None)
    )(keys, rays, nerfs, False, renderer_settings)
    return coarse_rgbs, fine_rgbs, depth_map


def render_frame(nerfs, camera, key, n_rays_per_chunk, renderer_settings):
    rays = camera.get_rays()
    rays_orig_shape = rays.origin.shape
    total_n_rays = rays_orig_shape[0] * rays_orig_shape[1]
    if total_n_rays > n_rays_per_chunk:
        n_chunks = int(total_n_rays / n_rays_per_chunk)
    else:
        n_chunks = 1
    assert n_chunks * n_rays_per_chunk == total_n_rays
    keys = jax.random.split(key, n_chunks)

    rays = jax.tree_map(lambda x: x.reshape((n_chunks, n_rays_per_chunk, 3)), rays)

    coarse_rgbs, fine_rgbs, depth_map = jax.lax.map(
        lambda ray_key: render_line(nerfs, ray_key[0], ray_key[1], renderer_settings),
        (rays, keys),
    )

    fine_rgbs = jax.tree_map(lambda x: x.reshape((*rays_orig_shape[:-1], 3)), fine_rgbs)
    coarse_rgbs = jax.tree_map(
        lambda x: x.reshape((*rays_orig_shape[:-1], 3)), coarse_rgbs
    )
    depth_map = jax.tree_map(lambda x: x.reshape(rays_orig_shape[:-1]), depth_map)

    return coarse_rgbs, fine_rgbs, depth_map


if __name__ == "__main__":
    test_ts = sample_coarse(jax.random.PRNGKey(0), 8)
    print(f"{sample_coarse(jax.random.PRNGKey(0), 8)=}")
    test_density = jnp.array([0.4, 0.9, 9.5, 0.1, 1.0, 2.0, 4.0, 0.9])
    test_deltas = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    alphas = calc_alpha(test_density, test_deltas)
    print(f"{alphas=}")
    T = calc_transmittance(alphas)
    print(f"{T=}")
    print(f"{T.sum()=}")
    w = calc_w(test_density, test_deltas)
    t = jnp.array([0.1, 0.2, 0.5, 0.9, 1.1, 1.8])
    print(f"{dists(t)=}")
    delta = jnp.diff(t)
    fine_ts = sample_fine(jax.random.PRNGKey(0), 10, test_density, test_ts)
    fine_ts = jnp.sort(fine_ts)
    print(f"{fine_ts=}")
