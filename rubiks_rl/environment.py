import random
from enum import Enum
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame, ObsType


# Simplified action space
# can only move in two directions
# reverse rotations are equivalent to
# 3 turns in the opposite direction anyway
class Action(Enum):
    ROTATE_TOP_LEFT = 0
    ROTATE_MIDDLE_LEFT = 1
    ROTATE_BOTTOM_LEFT = 2
    ROTATE_LEFT_DOWN = 3
    ROTATE_MIDDLE_DOWN = 4
    ROTATE_RIGHT_DOWN = 5


# Yes this is way over engineered
# No magic numbers!
# Only works for cubes!
class Face(Enum):
    TOP = 0
    FRONT = 1
    BOTTOM = 2
    LEFT = 3
    BACK = 4
    RIGHT = 5


# this could be modular tbh
class Row(Enum):
    TOP = 0
    MIDDLE = 1
    BOTTOM = 2


class Col(Enum):
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2


class RubiksCube(gym.Env):
    observation_space = gym.spaces.Box(
        shape=(len(Face), len(Row), len(Col)),
        low=Face.TOP.value,
        high=Face.RIGHT.value,
        dtype=np.uint8,
    )
    action_space = gym.spaces.Discrete(5)

    def __init__(self, shuffle: bool = True):
        self.shuffle = shuffle

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        self.cube = np.tile(
            np.reshape(np.arange(len(Face), dtype=np.uint8), (len(Face), 1, 1)),
            (1, len(Row), len(Col)),
        )

        if self.shuffle:
            if not seed:
                seed = random.randint(0, 128)
            seed = seed % 128
            for _ in range(seed):
                self.step(self.action_space.sample())

        return self.cube, {}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = Action(action)
        if action == Action.ROTATE_TOP_LEFT:
            self.cube[Face.TOP.value] = np.rot90(self.cube[Face.TOP.value])
            self.rotate_row(Row.TOP)
        elif action == Action.ROTATE_MIDDLE_LEFT:
            self.rotate_row(Row.MIDDLE)
        elif action == Action.ROTATE_BOTTOM_LEFT:
            self.cube[Face.BOTTOM.value] = np.rot90(self.cube[Face.BOTTOM.value])
            self.rotate_row(Row.BOTTOM)
        elif action == Action.ROTATE_LEFT_DOWN:
            self.cube[Face.LEFT.value] = np.rot90(self.cube[Face.LEFT.value])
            self.rotate_col(Col.LEFT)
        elif action == Action.ROTATE_MIDDLE_DOWN:
            self.rotate_col(Col.MIDDLE)
        elif action == Action.ROTATE_RIGHT_DOWN:
            self.cube[Face.TOP.value] = np.rot90(self.cube[Face.RIGHT.value])
            self.rotate_col(Col.RIGHT)

        done = self.reward == len(Face) * 9

        # TODO: Truncated
        return self.cube, self.reward(), done, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        grid = np.zeros((9, 12), dtype=np.uint8) - 1
        grid[:3, :3] = self.cube[Face.TOP.value]
        grid[6:, :3] = self.cube[Face.BOTTOM.value]
        for i, face in enumerate([Face.FRONT, Face.RIGHT, Face.BACK, Face.LEFT]):
            grid[3:6, 3 * i : 3 * (i + 1)] = self.cube[face.value]
        return grid

    def reward(self):
        # The reward is calculated by checking how much of each face matches the center of each face
        return np.sum(
            np.reshape(self.cube[:, Row.MIDDLE.value, Col.MIDDLE.value], (-1, 1, 1)) == self.cube
        )

    # there's a more mathy way of doing this tbh
    def rotate_row(self, row: Row):
        for old_face, new_face in (
            (Face.FRONT, Face.RIGHT),
            (Face.RIGHT, Face.BACK),
            (Face.BACK, Face.LEFT),
        ):
            # Could be a more efficient swap
            old = self.cube[old_face.value, row.value, :].copy()
            new = self.cube[new_face.value, row.value, :].copy()
            self.cube[old_face.value, row.value, :] = new
            self.cube[new_face.value, row.value, :] = old

    def rotate_col(self, col: Col):
        for old_face, new_face in (
            (Face.FRONT, Face.TOP),
            (Face.TOP, Face.BACK),
            (Face.BACK, Face.BOTTOM),
        ):
            old = self.cube[old_face.value, :, col.value].copy()
            new = self.cube[new_face.value, :, col.value].copy()
            self.cube[old_face.value, :, col.value] = new
            self.cube[new_face.value, :, col.value] = old


if __name__ == "__main__":
    env = RubiksCube(shuffle=False)
    env.reset()

    print("Welcome to Rubiks Env")
    print(env.render())
    print()

    while True:
        action = Action(int(input("Action (0-5): ")))
        env.step(action)
        print("Performing action:", action)
        print("State: ")
        print(env.render())
        print()
