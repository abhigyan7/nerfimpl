#!/usr/bin/env python3

from typing import Any
from jaxlie import SE3, SO3
import jax.numpy as jnp


def lerp(start: Any, end: Any, t: float):
    return start + (end - start) * t


def slerp(start_pose: SO3, end_pose: SO3, t) -> SO3:
    q1 = start_pose.wxyz
    q2 = end_pose.wxyz
    cos_theta_by_two = q1.dot(q2)
    theta_by_two = jnp.arccos(cos_theta_by_two)
    q_ret = q1 * jnp.sin((1 - t) * theta_by_two) + q2 * jnp.sin(t * theta_by_two)
    return SO3(q_ret).normalize()


class LinearTrajectory:
    def __init__(self, start_pose: SE3, end_pose: SE3, time_length: float):
        self.end_pose = end_pose
        self.start_pose = start_pose
        self.time_length = time_length

    def __call__(self, t):
        return SE3.from_rotation_and_translation(
            slerp(self.start_pose.rotation(), self.end_pose.rotation(), t),
            lerp(self.start_pose.translation(), self.end_pose.translation(), t),
        )

    def duration(self):
        return self.time_length


def main():
    pose1 = SE3.identity()
    pose2 = SE3.from_rotation_and_translation(
        SO3.from_rpy_radians(0.3, 1.0, -0.5), jnp.array([1, 2, 3])
    )
    lt = LinearTrajectory(pose1, pose2, 1.0)
    print(pose1)
    print(pose2)
    for i in range(11):
        j = i / 10
        print(f"{j=} {lt(j)=}")


if __name__ == "__main__":
    main()
