#!/usr/bin/env python3

import jax.numpy as jnp
import jax

def positional_encoding(p, L, scale=1.0):
    p = p / scale
    theta = jnp.outer(p, jnp.pi * (2.0 ** jnp.arange(L)))
    sines = jnp.sin(theta)
    cosines = jnp.cos(theta)
    periodic_fns = jnp.concatenate([sines.reshape(-1), cosines.reshape(-1)])
    return periodic_fns

if __name__ == "__main__":
    a = jax.random.uniform(jax.random.PRNGKey(0), [3,])
    pe = positional_encoding(a, 10)
    print(f"{pe=}")
    print(f"{a.shape=}")
    print(f"{pe.shape=}")
