# --------------------------------------------------------------------------------------------------
# This environement is similar to singleagent_oneoven.py. A few variables have been adjusted to match the dual-task variant
# For example, the upper bound in observation space is increased to the dual-task variant.
# This enables incremental learning. That is, first train the agent on single task (timing task). Then train it on dual task. To make sure the dual-task variant has learned the timing task correctly
# ------------------------------------------------------------------------------------------------------

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np
import random


class OverCookedSingleAgentIncrementallearning(gym.Env):

    # object code - 1- Agent, 3- Onion, 4- delivery counter,   5 - oven, 7 -plate (test also with one-hot encoding)
    ### channel 1 - map # There is no orientation in the basic version so make sure the agent has only one object to interact at a given time

    #    [0., 3, 4],
    #    [0., 0., 0.],
    #    [5, 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 7]

    ### channel 2 - Agent position
    #    [0., 0, 0],
    #    [0., 1, 0.],
    #    [0., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 3 - carrying onion
    #    [0., 0, 0],
    #    [0., 1, 0.],
    #    [0., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 4 - carrying soup
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [0., 0., 0],
    #    [0., 0., 0],
    #    [0., 0, 0.]

    ### channel 5 - oven states
    #    [0., 0, 0],
    #    [0., 0, 0.],
    #    [1., 0., 0],
    #    [0., 0., 1],
    #    [0., 0, 0.]


    def __init__(self, oven_duration=8, n_actions=6, max_timesteps=200, grid_size=(5, 3), n_channels=6):
        self.onion_pos = (0, 1)
        self.delivery_pos = (0, 2)
        self.oven1_pos = (2, 0)

        self.max_timesteps = max_timesteps
        self.oven_duration = oven_duration
        # self.grid_map[4,1,0] = 5
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.n_actions = n_actions

        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=0, high=20, shape=(self.grid_map.shape),
                                     dtype=np.uint8)  # Upper bound 20 chosen to match dual task variant

    def get_obs(self, ):
        return self.grid_map

    def reset_oven_timers(self):
        self.oven1_timer = -1
        # self.oven2_timer = -1

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.reset_oven_timers()
        self.num_soups_delivered = 0
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.grid_map[self.onion_pos[0], self.onion_pos[1], 0] = 3
        self.grid_map[self.delivery_pos[0], self.delivery_pos[1], 0] = 4
        self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 0] = 5

        self.free_positions = [
            (i, j) for i in range(self.grid_map.shape[0])
            for j in range(self.grid_map.shape[1])
            if self.grid_map[i, j, 0] == 0
        ]

        # Randomly select a free position for the agent
        self.agent_position = random.choice(self.free_positions)
        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 1
        self.oven1_state = 0
        return self.get_obs(), {}

    def move_agent(self, action) -> int:
        reward = 0
        carrying_onion = carrying_soup = False
        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 0

        # check if agent is carrying onion
        if self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1:
            carrying_onion = True
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0
        if 1 in self.grid_map[:, :, 2]:
            raise ValueError("onion carrying channel contains incorrect information")

        # check if agent is carrying soup
        if self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1:
            carrying_soup = True
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
        if 1 in self.grid_map[:, :, 3]:
            raise ValueError("soup carrying channel contains incorrect information")

        if action == 1:  # down
            new_pos = (self.agent_position[0] + 1, self.agent_position[1])
            if (new_pos in self.free_positions) and (new_pos[0] < self.grid_map.shape[0]):
                self.agent_position = new_pos

        elif action == 2:  # up
            new_pos = (self.agent_position[0] - 1, self.agent_position[1])
            if (new_pos in self.free_positions) and (new_pos[0] >= 0):
                self.agent_position = new_pos

        elif action == 3:  # right
            new_pos = (self.agent_position[0], self.agent_position[1] + 1)
            if (new_pos in self.free_positions) and (new_pos[1] < self.grid_map.shape[1]):
                self.agent_position = new_pos

        elif action == 4:  # left
            new_pos = (self.agent_position[0], self.agent_position[1] - 1)
            if (new_pos in self.free_positions) and (new_pos[1] >= 0):
                self.agent_position = new_pos

        self.grid_map[self.agent_position[0], self.agent_position[1], 1] = 1
        if carrying_onion:
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 1
        elif carrying_soup:
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 1

        return reward

    def get_adjacent_cells(self, pos):
        """
        Get adjacent non-diagonal cell values from grid map for a given position.
        """
        rows, cols = self.grid_map[:, :, 0].shape
        r, c = pos
        adjacent = {}

        # Up
        if r > 0:
            adjacent[(r - 1, c)] = self.grid_map[r - 1, c, 0]
        else:
            adjacent[(r - 1, c)] = None

        # Down
        if r < rows - 1:
            adjacent[(r + 1, c)] = self.grid_map[r + 1, c, 0]
        else:
            adjacent[(r + 1, c)] = None

        # Left
        if c > 0:
            adjacent[(r, c - 1)] = self.grid_map[r, c - 1, 0]
        else:
            adjacent[(r, c - 1)] = None

        # Right
        if c < cols - 1:
            adjacent[(r, c + 1)] = self.grid_map[r, c + 1, 0]
        else:
            adjacent[(r, c + 1)] = None

        return adjacent

    def interact(self) -> int:
        reward = 0
        adjacent_cells = self.get_adjacent_cells(self.agent_position)
        # check if there are more than one objects to interact at a time
        interact_obj_count = sum(1 for val in adjacent_cells.values() if val not in [None, 0.0])
        if interact_obj_count > 1:
            raise ValueError("more than 1 object to interact with", self.grid_map[:, :, 0])

        # interacts with onion dispenser and not carrying soup or onion already
        if (self.onion_pos in adjacent_cells.keys()) and (
                all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0)):  # carry onion
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 1


        # oven on: carrying onion and interacts with oven that is not on
        elif (self.oven1_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 2] == 1) and (self.oven1_timer == -1):
            self.oven1_timer = 0
            self.oven1_state = 1
            self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 4] = 1
            self.grid_map[self.agent_position[0], self.agent_position[1], 2] = 0

        # oven off: not carrying onion or soup and interacts with oven when oven time more than or equal to target duration
        # TODO: add overcooked condition in step function - reset oven timer if overcooked
        elif (self.oven1_pos in adjacent_cells.keys()) and (
                all(self.grid_map[self.agent_position[0], self.agent_position[1], 2:4] == 0)) and (
                self.oven1_timer >= self.oven_duration):
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 1
            self.oven1_timer = -1
            self.oven1_state = 0
            self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 4] = 0


        # interacts with delivery counter and carrying soup
        elif (self.delivery_pos in adjacent_cells.keys()) and (
                self.grid_map[self.agent_position[0], self.agent_position[1], 3] == 1):
            self.grid_map[self.agent_position[0], self.agent_position[1], 3] = 0
            self.num_soups_delivered += 1
            reward = 1

        return reward

    # actions: 0:down, 1-up, 2-right, 3-left, 4- interact
    # reward calculation: 1 for soup delivery and 0 otherwise
    def step(self, action):
        self.timestep += 1
        if action != 5:  # navigation action
            reward = self.move_agent(action)
        else:
            # interact with object
            reward = self.interact()
        if self.oven1_timer != -1:  # increment oven timer if oven already on
            self.oven1_timer += 1

        if self.timestep > self.max_timesteps:  # terminate episode after max timesteps
            terminated = True
        else:
            terminated = False

        return self.get_obs(), reward, terminated, False, {}
