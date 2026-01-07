import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class RocketLeaguePolicy(BaseFeaturesExtractor):
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        input_size = observation_space.shape[0]

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 512),
            nn.ReLU(),
            
            # Layer 2
            nn.Linear(512, 512),
            nn.ReLU(),

            # Layer 3
            nn.Linear(512, features_dim),
            nn.ReLU(),

        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
