# --------------------------------------------------------------------------------------------------
# This environement is similar to singleagent_dualtask.py but with a delayed reward instead of an immediate reward for correctly interacting with the number game counter
# It enables matching the dualtask variation with the single task since all reward is delivered to agent only after soup delivery
# ------------------------------------------------------------------------------------------------------

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np
import random


class OverCookedSingleAgentDualtaskDelayedReward(gym.Env):
    # objects - oven, onion, delivery counter, cl block - classify number > 100 or < 100
    # object state - on-time (oven), location, ready (soup), soup overcooked, picked (onion, plate, soup)
    # players - a1,
    # player state - location, picked object,

    # object code - 1- Agent, 3- Onion, 4- delivery counter,   5 - oven, 8 -cl block
    ### channel 1 - map

    #    [0., 3., 4],
    #    [0., 0., 0.],
    #    [5, 0., 0.],
    #    [0., 0., 0.],
    #    [0.,8, 0.]

    ### channel 2 - Agent position
    #    [0., 0., 0.],
    #    [0., 1., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.]

    ### channel 3 - carrying onion
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.]

    ### channel 4 - carrying soup
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.]

    ### channel 5 - oven states 1 = on, 2 = overcooked (omit for v0) , 0 = off
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [1., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.]

    ### channel 6 - cl block
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 0., 0.],
    #    [0., 15, 0.]

    # cl block - activated once soup is placed in oven for 4 timesteps. Agent must press 5 if number less than 5 and more than zero (other action reward 0), press 0 key if more than or equal to 5 (reward for other keys is zero)

    def __init__(self, oven_duration=8, overcooked_threshold=4, n_actions=6, max_timesteps=200, grid_size=(5, 3),
                 n_channels=6):
        self.onion_pos = (0, 1)
        self.delivery_pos = (0, 2)
        self.oven1_pos = (2, 0)
        self.cl_block_pos = (4, 1)  # add cl block

        self.max_timesteps = max_timesteps
        self.oven_duration = oven_duration
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.n_actions = n_actions

        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=0, high=20, shape=(self.grid_map.shape),
                                     dtype=np.uint8)  # Upper bound 20 chosen to allow CL block random values (0–20); may be narrowed for stability.

    def get_obs(self, ):
        return self.grid_map

    def reset_oven_timers(self):
        self.oven1_timer = -1

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.clblock_rewards = 0
        self.reset_oven_timers()
        self.num_soups_delivered = 0
        self.grid_map = np.zeros((self.grid_size[0], self.grid_size[1], self.n_channels), dtype=np.uint8)
        self.grid_map[self.onion_pos[0], self.onion_pos[1], 0] = 3
        self.grid_map[self.delivery_pos[0], self.delivery_pos[1], 0] = 4
        self.grid_map[self.oven1_pos[0], self.oven1_pos[1], 0] = 5
        self.grid_map[self.cl_block_pos[0], self.cl_block_pos[1], 0] = 8  # add cl block
        # self.grid_map[self.cl_block_pos[0],self.cl_block_pos[1],5] =  random.choice(range(0,20))
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

    # Move agent and calculate reward.
    # reward calculation - 0 for movement actions (including correctly interacting with cl_block).
    # If agent correctly interacts with cl_block a delayed reward is stored which is added to the total reward after soup delivery

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

        if action == 0:  # check if agent waits for the right input in cl block
            adjacent_cells = self.get_adjacent_cells(self.agent_position)
            if (self.cl_block_pos in adjacent_cells.keys()) and (
                    self.grid_map[self.cl_block_pos[0], self.cl_block_pos[1], 5] >= 5):
                # reward = 1  # can add surprise factor by increasing this reward?
                self.clblock_rewards += 1

        elif action == 1:  # down
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

        # oven off: not carrying onion or soup and interacts with oven when oven time more than or equal to target duratio
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
            self.num_soups_delivered += 1  # add a number to cl block, set cl_block_played flag to false
            reward = 1

        # interact with cl block and cl block number less than 5
        elif (self.cl_block_pos in adjacent_cells.keys()) and (
                (self.grid_map[self.cl_block_pos[0], self.cl_block_pos[1], 5] < 5) and (
                self.grid_map[self.cl_block_pos[0], self.cl_block_pos[1], 5] > 0)):
            self.clblock_rewards += 1

        return reward

    # actions: 0:down, 1-up, 2-right, 3-left, 4- interact
    def step(self, action):
        self.timestep += 1

        if action != 5:
            reward = self.move_agent(action)
        else:
            # interact with object
            reward = self.interact()
            if reward == 1:
                reward += self.clblock_rewards
                self.clblock_rewards = 0
        if self.oven1_timer != -1:  # increment oven timer if oven already on
            self.oven1_timer += 1

        if (self.oven1_timer >= 0) and (self.oven1_timer <= 4):
            # print("activated cl block")
            self.grid_map[self.cl_block_pos[0], self.cl_block_pos[1], 5] = random.choice(range(0, 10))
        else:
            self.grid_map[self.cl_block_pos[0], self.cl_block_pos[1], 5] = 0

        if self.timestep > self.max_timesteps:  # terminate episode after max timesteps
            terminated = True
        else:
            terminated = False

        return self.get_obs(), reward, terminated, False, {}
