#!/usr/bin/env python3

import equinox as eqx
import jax.numpy as jnp
import jax


class PositionalEncoding(eqx.Module):
    L: float
    scale: float = 1.0

    def __call__(self, p):
        p = p / self.scale
        theta = jnp.outer(p, jnp.pi * (2.0 ** jnp.arange(self.L)))
        sines = jnp.sin(theta)
        cosines = jnp.cos(theta)
        periodic_fns = jnp.concatenate([sines.reshape(-1), cosines.reshape(-1)])
        return periodic_fns


if __name__ == "__main__":
    a = jax.random.uniform(
        jax.random.PRNGKey(0),
        [
            3,
        ],
    )
    pe = PositionalEncoding(10)(a)
    print(f"{pe=}")
    print(f"{a.shape=}")
    print(f"{pe.shape=}")
