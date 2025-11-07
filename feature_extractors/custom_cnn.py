import torch as th
import torch.nn as nn

# from gym.spaces import Box

from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: Box, features_dim: int = 256, n_channels: int = 6):
        super().__init__(observation_space, features_dim)

        # n_input_channels = observation_space.shape[0] # does not work because input is not chaneel first
        n_input_channels = n_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, n_input_channels, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = observation_space.sample()[None]
            obs = th.tensor(obs, dtype=th.float32).permute(0, 3, 1, 2)
            n_flatten = self.cnn(obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.detach().clone().float().permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))
