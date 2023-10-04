#!/usr/bin/env python3

from primitives import camera, encoding, mlp, render
import jax
import jax.numpy as jnp

import jaxlie

def main():
    key = jax.random.PRNGKey(0)
    nerf_key, sampler_key = jax.random.split(key, 2)
    pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0, 0.1, 0),
        jnp.array([0,0,0]),
    )
    cam = camera.PinholeCamera(100.0, (720, 1280), pose, 1.0)
    ray = cam.get_ray(1, 3)
    nerf = mlp.MhallMLP(nerf_key)
    coarse_key, fine_key, sampler_key = jax.random.split(sampler_key, 3)
    coarse_ts = render.sample_coarse(coarse_key, 64)
    xyzs = jax.vmap(ray.at)(coarse_ts)
    location = jax.vmap(lambda x: encoding.positional_encoding(x,10))(xyzs)
    direction = jnp.stack([encoding.positional_encoding(ray.direction, 4) for i in range(location.shape[0])])
    nerf_densities, nerf_rgbs = jax.vmap(nerf)(location, direction)

    coarse_T = render.calc_T(nerf_densities[1:], jnp.diff(coarse_ts))
    coarse_w = render.calc_w(nerf_densities[1:], jnp.diff(coarse_ts))
    alpha = coarse_T * coarse_w
    coarse_color = (alpha @ nerf_rgbs[1:])

    fine_ts = render.sample_fine(fine_key, 128, nerf_densities, coarse_ts)
    fine_ts = jnp.concatenate((coarse_ts, fine_ts))
    xyzs = jax.vmap(ray.at)(fine_ts)
    location = jax.vmap(lambda x: encoding.positional_encoding(x,10))(xyzs)
    direction = jnp.tile(encoding.positional_encoding(ray.direction, 4), (location.shape[0], 1))
    nerf_densities, nerf_rgbs = jax.vmap(nerf)(location, direction)

    nerf_densities = nerf_densities[1:]
    nerf_rgbs = nerf_rgbs[1:]

    fine_T = render.calc_T(nerf_densities, jnp.diff(fine_ts))
    fine_w = render.calc_w(nerf_densities, jnp.diff(fine_ts))
    alpha = fine_T * fine_w
    fine_color = (alpha @ nerf_rgbs)
    print(f"{coarse_color=}")
    print(f"{fine_color=}")
    return

if __name__ == "__main__":
    main()
