#!/usr/bin/env python3

import jax.numpy as jnp
import jax
from functools import partial

@partial(jax.jit, static_argnums=1)
def positional_encoding(p, L):
    theta = jnp.outer(p, jnp.pi * (2.0 ** jnp.arange(L))).reshape(-1)
    periodic_fns = jnp.concatenate([jnp.sin(theta), jnp.cos(theta)])
    return periodic_fns

