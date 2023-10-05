#!/usr/bin/env python3

from primitives import camera, mlp, render
import jax
import jax.numpy as jnp
import equinox as eqx
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

    coarse_rgb, fine_rgb = render.hierarchical_render_single_ray(sampler_key, ray, nerf)

    rays = [cam.get_ray(1, i) for i in range(10)]
    rays = jax.vmap(cam.get_ray)(jnp.arange(10), jnp.arange(10,20))
    sampler_keys = jax.random.split(sampler_key, 10)
    coarse_rgbs, fine_rgbs = eqx.filter_vmap(
        render.hierarchical_render_single_ray,
        in_axes=(0, eqx.if_array(0), None)
    ) (sampler_keys, rays, nerf)

    print(f"{coarse_rgbs=}")
    print(f"{fine_rgbs=}")

    return

if __name__ == "__main__":
    main()
