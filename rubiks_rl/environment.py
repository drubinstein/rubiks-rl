import argparse
import random
from enum import Enum
from functools import cached_property
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, RenderFrame


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


class RubiksCube(gym.Env):
    def __init__(self, shuffle: bool = True, n_rows: int = 3):
        self.shuffle = shuffle
        self.n_rows = n_rows

    @cached_property
    def observation_space(self):
        return gym.spaces.Box(
            shape=(len(Face), self.n_rows, self.n_rows),
            low=Face.TOP.value,
            high=Face.RIGHT.value,
            dtype=np.uint8,
        )

    # The action space maps to either moving a row to the left or a column down
    # SB3 doesn't support dictionary action spaces
    # Encoding is
    #  - [0, n_rows) means move row left
    #  - [n_rows, n_rows+n_rows) menas move column down
    @cached_property
    def action_space(self):
        return gym.spaces.Discrete(self.n_rows + self.n_rows)

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        self.cube = np.tile(
            np.reshape(np.arange(len(Face), dtype=np.uint8), (len(Face), 1, 1)),
            (1, self.n_rows, self.n_rows),
        )

        if self.shuffle:
            if not seed:
                seed = random.randint(0, 128)
            seed = seed % 128
            for _ in range(seed):
                self.run_action(self.action_space.sample())
        self.current_reward = self.reward()

        return self.cube, {"total_reward": self.current_reward}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.run_action(action)
        reward = self.reward()
        done = reward == len(Face) * (self.n_rows**2)
        step_reward = reward - self.current_reward
        self.current_reward = reward

        # TODO: Truncated
        return self.cube, step_reward, done, False, {"total_reward": self.current_reward}

    def run_action(self, action: int):
        if action < self.n_rows:
            if action == 0:
                self.cube[Face.TOP.value] = np.rot90(self.cube[Face.TOP.value])
            if action == self.n_rows - 1:
                self.cube[Face.BOTTOM.value] = np.rot90(self.cube[Face.BOTTOM.value])
            self.rotate_row(action)
        else:
            action -= self.n_rows
            if action == 0:
                self.cube[Face.LEFT.value] = np.rot90(self.cube[Face.LEFT.value])
            if action == self.n_rows - 1:
                self.cube[Face.RIGHT.value] = np.rot90(self.cube[Face.RIGHT.value])
            self.rotate_col(action)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        grid = np.zeros((3 * self.n_rows, 4 * self.n_rows), dtype=np.uint8) - 1
        grid[: self.n_rows, : self.n_rows] = self.cube[Face.TOP.value]
        grid[2 * self.n_rows :, : self.n_rows] = self.cube[Face.BOTTOM.value]
        for i, face in enumerate([Face.FRONT, Face.RIGHT, Face.BACK, Face.LEFT]):
            grid[self.n_rows : 2 * self.n_rows, self.n_rows * i : self.n_rows * (i + 1)] = (
                self.cube[face.value]
            )
        return grid

    def reward(self):
        # The reward is calculated by checking how much of each face matches the upper left corner of
        # each face
        return np.sum(np.reshape(self.cube[:, 0, 0], (-1, 1, 1)) == self.cube)

    # there's a more mathy way of doing this tbh
    def rotate_row(self, row: int):
        for old_face, new_face in (
            (Face.FRONT, Face.RIGHT),
            (Face.RIGHT, Face.BACK),
            (Face.BACK, Face.LEFT),
        ):
            # Could be a more efficient swap
            old = self.cube[old_face.value, row, :].copy()
            new = self.cube[new_face.value, row, :].copy()
            self.cube[old_face.value, row, :] = new
            self.cube[new_face.value, row, :] = old

    def rotate_col(self, col: int):
        for old_face, new_face in (
            (Face.FRONT, Face.TOP),
            (Face.TOP, Face.BACK),
            (Face.BACK, Face.BOTTOM),
        ):
            old = self.cube[old_face.value, :, col].copy()
            new = self.cube[new_face.value, :, col].copy()
            self.cube[old_face.value, :, col] = new
            self.cube[new_face.value, :, col] = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Rubik's Cube Environment Simulator")
    parser.add_argument(
        "--n-rows", type=int, default=3, help="Number of rows and columns in the cube."
    )
    args = parser.parse_args()

    env = RubiksCube(shuffle=False, n_rows=args.n_rows)
    env.reset()

    print("Welcome to Rubiks Env")
    print(env.render())
    print()

    while True:
        action = int(input(f"Action (0-{env.action_space.n-1}): "))

        env.step(action)
        print("Performing action:", action)
        print("State: ")
        print(env.render())
        print()
