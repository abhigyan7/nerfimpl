#!/usr/bin/env python3

from primitives import camera, encoding, mlp, render
import jax
import jax.numpy as jnp

import jaxlie

def main():
    key = jax.random.PRNGKey(0)
    nerf_key, sampler_key = jax.random.split(key, 2)
    pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(0,0,0),
        jnp.array([0,0,0]),
    )
    cam = camera.PinholeCamera(100.0, (720, 1280), pose, 1.0)
    ray = cam.get_ray(0, 0)
    direction = encoding.positional_encoding(ray.direction, 4)
    location = encoding.positional_encoding(ray.origin, 10)
    nerf = mlp.MhallMLP(nerf_key)
    print(f"{nerf(location, direction)=}")
    sampler = render.HierarchicalSampler(sampler_key, 3, 10)
    print(f"{sampler.get_coarse_points()=}")
    print(f"{sampler.get_coarse_points()=}")
    print(f"{sampler.get_coarse_points()=}")
    print(f"{sampler.get_coarse_points()=}")
    return

if __name__ == "__main__":
    main()
