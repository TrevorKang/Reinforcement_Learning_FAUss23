import copy
from typing import Any, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]
        self.start_station = [0, 0]
        # TODO: Define your action_space and observation_space here
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=np.shape(self.map)[0], shape=(2,), dtype=np.int32)
        self.agent_position = self.start_station    # start position

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # TODO: Write your implementation here
        self.agent_position = self.start_station
        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # TODO: Write your implementation here
        # 0:y
        # 1:x
        # 0     Go up
        # 1     Go right
        # 2     Go down
        # 3     Go left
        if action == 0:
            self.agent_position[0] -= 1
        elif action == 1:
            self.agent_position[1] += 1
        elif action == 2:
            self.agent_position[0] += 1
        else:
            self.agent_position[1] -= 1

        # to make sure it is within the boundary
        self.agent_position[0] = clamp(v=self.agent_position[0], minimal_value=0, maximal_value=3)
        self.agent_position[1] = clamp(v=self.agent_position[1], minimal_value=0, maximal_value=3)
        observation = self._observe()

        # count the reward and determine if it is trapped or meets terminal state
        reward = 0  # normal reward
        done = False    # flag for terminal state

        if self.map[self.agent_position[0]][self.agent_position[1]] == 't':     # trap
            done = True
            reward = -1
        if self.map[self.agent_position[0]][self.agent_position[1]] == 'g':     # goal
            done = True
            reward = 1
        return observation, reward, done, False, {}

    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass

    def _observe(self):
        return np.array(self.agent_position)
