#!/usr/bin/env python3

from typing import Any, List
from jaxlie import SE3, SO3
from itertools import chain
import jax.numpy as jnp


def ease_in_out_bezier(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def lerp(start: Any, end: Any, t: float):
    return start + (end - start) * t


def slerp(start_rot: SO3, end_rot: SO3, t) -> SO3:
    q1 = start_rot.wxyz
    q2 = end_rot.wxyz
    cos_theta_by_two = q1.dot(q2)
    theta_by_two = jnp.arccos(cos_theta_by_two)
    q_ret = q1 * jnp.sin((1 - t) * theta_by_two) + q2 * jnp.sin(t * theta_by_two)
    return SO3(q_ret).normalize()


class LinearTrajectory:
    def __init__(self, start_pose: SE3, end_pose: SE3, duration: int):
        # duration in frames
        self.end_pose = end_pose
        self.start_pose = start_pose
        self.duration = duration
        self.current_frame = 0

    def interp(self, frame):
        return SE3.from_rotation_and_translation(
            slerp(
                self.start_pose.rotation(),
                self.end_pose.rotation(),
                frame / (self.duration - 1),
            ),
            lerp(
                self.start_pose.translation(),
                self.end_pose.translation(),
                frame / (self.duration - 1),
            ),
        )

    def __iter__(self):
        return self

    def __next__(self) -> SE3:
        if self.current_frame == self.duration:
            raise StopIteration
        ret = self.interp(self.current_frame)
        self.current_frame += 1
        return ret

    def __len__(self) -> int:
        return self.duration


class MetaTrajectory:
    def __init__(self, trajectories: List[LinearTrajectory]):
        self.trajectories = trajectories

    def __iter__(self):
        return chain(*self.trajectories)

    def __len__(self):
        return sum((x.duration for x in self.trajectories))

    @staticmethod
    def from_poses(poses: List[SE3], duration: List[int]):
        ret = MetaTrajectory([])
        for i in range(len(poses) - 1):
            ret.trajectories.append(
                LinearTrajectory(poses[i], poses[i + 1], duration[i])
            )
        return ret


def main():
    pose1 = SE3.identity()
    pose2 = SE3.from_rotation_and_translation(
        SO3.from_rpy_radians(0.3, 1.0, -0.5), jnp.array([1, 2, 3])
    )
    pose3 = SE3.identity()
    lt1 = LinearTrajectory(pose1, pose2, 25)
    lt2 = LinearTrajectory(pose2, pose3, 42)
    m = MetaTrajectory([lt1, lt2])
    print(pose1)
    print(pose2)
    print(pose3)
    print(len(m))
    for i, pose in enumerate(m):
        print(f"{i=}, {pose=}")

    m = MetaTrajectory.from_poses([pose1, pose2, pose3], [25, 25])
    for i, pose in enumerate(m):
        print(f"{i=}, {pose=}")


if __name__ == "__main__":
    main()
